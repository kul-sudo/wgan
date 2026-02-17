use crate::consts::CHANNELS;
use crate::netperceptual::{PerceptualNet, PerceptualNetConfig};
use burn::{
    config::Config,
    module::Module,
    nn::{
        GaussianNoise, GaussianNoiseConfig, InstanceNorm, InstanceNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Tensor,
        activation::{leaky_relu, tanh},
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

#[derive(Module, Debug)]
pub struct DiscriminatorBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: InstanceNorm<B>,
}

#[derive(Config, Debug)]
pub struct DiscriminatorBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
}

impl DiscriminatorBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DiscriminatorBlock<B> {
        DiscriminatorBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm: InstanceNormConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> DiscriminatorBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        leaky_relu(x, 0.2)
    }
}

#[derive(Module, Debug)]
pub struct GeneratorConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: InstanceNorm<B>,
}

#[derive(Config, Debug)]
pub struct GeneratorConvBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
}

impl GeneratorConvBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> GeneratorConvBlock<B> {
        GeneratorConvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm: InstanceNormConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> GeneratorConvBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        leaky_relu(x, 0.2)
    }
}

#[derive(Module, Debug)]
pub struct GeneratorDeconvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: InstanceNorm<B>,
}

#[derive(Config, Debug)]
pub struct GeneratorDeconvBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl GeneratorDeconvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeneratorDeconvBlock<B> {
        GeneratorDeconvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm: InstanceNormConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> GeneratorDeconvBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _channels, h, w] = input.dims();

        let x = interpolate(
            input,
            [h * 2, w * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );

        let x = self.conv.forward(x);
        let x = self.norm.forward(x);
        leaky_relu(x, 0.2)
    }
}

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    pub enc1: GeneratorConvBlock<B>,
    pub enc2: GeneratorConvBlock<B>,
    pub enc3: GeneratorConvBlock<B>,
    pub enc4: GeneratorConvBlock<B>,
    pub dec4: GeneratorDeconvBlock<B>,
    pub dec1: GeneratorDeconvBlock<B>,
    pub dec2: GeneratorDeconvBlock<B>,
    pub dec3: GeneratorDeconvBlock<B>,
    pub final_conv: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct GeneratorConfig {
    hidden_channels: usize,
}

impl GeneratorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        let c = self.hidden_channels;

        Generator {
            enc1: GeneratorConvBlockConfig::new(CHANNELS, c, 2).init(device),
            enc2: GeneratorConvBlockConfig::new(c, c * 2, 2).init(device),
            enc3: GeneratorConvBlockConfig::new(c * 2, c * 4, 2).init(device),
            enc4: GeneratorConvBlockConfig::new(c * 4, c * 8, 2).init(device),
            dec4: GeneratorDeconvBlockConfig::new(c * 8, c * 4).init(device),
            dec1: GeneratorDeconvBlockConfig::new(c * 8, c * 2).init(device),
            dec2: GeneratorDeconvBlockConfig::new(c * 4, c).init(device),
            dec3: GeneratorDeconvBlockConfig::new(c * 2, c).init(device),
            final_conv: Conv2dConfig::new([c, CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }
}

impl<B: Backend> Generator<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let s1 = self.enc1.forward(input);
        let s2 = self.enc2.forward(s1.clone());
        let s3 = self.enc3.forward(s2.clone());
        let s4 = self.enc4.forward(s3.clone());

        let x = self.dec4.forward(s4);
        let x = Tensor::cat(vec![x, s3], 1);

        let x = self.dec1.forward(x);
        let x = Tensor::cat(vec![x, s2], 1);

        let x = self.dec2.forward(x);
        let x = Tensor::cat(vec![x, s1], 1);

        let x = self.dec3.forward(x);
        let x = self.final_conv.forward(x);

        tanh(x)
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    pub noise: GaussianNoise,
    pub conv1: DiscriminatorBlock<B>,
    pub conv2: DiscriminatorBlock<B>,
    pub conv3: DiscriminatorBlock<B>,
    pub final_layer: Conv2d<B>,
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.noise.forward(images);

        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);

        let x = self.final_layer.forward(x);
        x.mean_dim(2).mean_dim(3).flatten(1, 3)
    }
}

#[derive(Config, Debug)]
pub struct NetworkConfig {
    #[config(default = 64)]
    pub hidden_channels: usize,
}

impl NetworkConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> (Generator<B>, Discriminator<B>, PerceptualNet<B>) {
        let c = self.hidden_channels;

        let generator = GeneratorConfig::new(c).init(device);
        let discriminator = Discriminator {
            noise: GaussianNoiseConfig::new(0.05).init(),
            conv1: DiscriminatorBlockConfig::new(CHANNELS, c, 2).init(device),
            conv2: DiscriminatorBlockConfig::new(c, c * 2, 2).init(device),
            conv3: DiscriminatorBlockConfig::new(c * 2, c * 4, 2).init(device),
            final_layer: Conv2dConfig::new([c * 4, 1], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        };
        let perceptual_net = PerceptualNetConfig::new(CHANNELS, c).init(device);

        (generator, discriminator, perceptual_net)
    }
}
