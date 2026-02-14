use crate::consts::CHANNELS;
use burn::{
    config::Config,
    module::Module,
    nn::{
        GaussianNoise, GaussianNoiseConfig, InstanceNorm, InstanceNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Tensor,
        activation::{leaky_relu, mish, tanh},
        backend::Backend,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

#[derive(Module, Debug)]
pub struct DiscriminatorBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: InstanceNorm<B>,
}

impl<B: Backend> DiscriminatorBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn = InstanceNormConfig::new(out_channels).init(device);

        Self { conv, bn }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.bn.forward(x);
        leaky_relu(x, 0.2)
    }
}

#[derive(Module, Debug)]
pub struct GeneratorConvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: InstanceNorm<B>,
}

impl<B: Backend> GeneratorConvBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn = InstanceNormConfig::new(out_channels).init(device);

        Self { conv, bn }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.bn.forward(x);
        mish(x)
    }
}

#[derive(Module, Debug)]
pub struct GeneratorDeconvBlock<B: Backend> {
    conv: Conv2d<B>,
    bn: InstanceNorm<B>,
}

impl<B: Backend> GeneratorDeconvBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn = InstanceNormConfig::new(out_channels).init(device);

        Self { conv, bn }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, _channels, h, w] = input.dims();

        let x = interpolate(
            input,
            [h * 2, w * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );

        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        mish(x)
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

    pub fn forward_with_features(
        &self,
        images: Tensor<B, 4>,
    ) -> (Tensor<B, 2>, Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.noise.forward(images);

        let f1 = self.conv1.forward(x);
        let f2 = self.conv2.forward(f1.clone());
        let x = self.conv3.forward(f2.clone());

        let x = self.final_layer.forward(x);
        let score = x.mean_dim(2).mean_dim(3).flatten(1, 3);

        (score, f1, f2)
    }
}

#[derive(Config, Debug)]
pub struct NetworkConfig {
    #[config(default = 64)]
    pub base_channels: usize,
}

impl NetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Generator<B>, Discriminator<B>) {
        let c = self.base_channels;

        let generator = Generator {
            enc1: GeneratorConvBlock::new(CHANNELS, c, 2, device),
            enc2: GeneratorConvBlock::new(c, c * 2, 2, device),
            enc3: GeneratorConvBlock::new(c * 2, c * 4, 2, device),
            enc4: GeneratorConvBlock::new(c * 4, c * 8, 2, device),
            dec4: GeneratorDeconvBlock::new(c * 8, c * 4, device),
            dec1: GeneratorDeconvBlock::new(c * 8, c * 2, device),
            dec2: GeneratorDeconvBlock::new(c * 4, c, device),
            dec3: GeneratorDeconvBlock::new(c * 2, c, device),
            final_conv: Conv2dConfig::new([c, CHANNELS], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        };

        let discriminator = Discriminator {
            noise: GaussianNoiseConfig::new(0.05).init(),
            conv1: DiscriminatorBlock::new(CHANNELS, c, 2, device),
            conv2: DiscriminatorBlock::new(c, c * 2, 2, device),
            conv3: DiscriminatorBlock::new(c * 2, c * 4, 2, device),
            final_layer: Conv2dConfig::new([c * 4, 1], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        };

        (generator, discriminator)
    }
}
