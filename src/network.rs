use crate::consts::CHANNELS;
use crate::nets::{
    flashattention::FlashAttentionV3,
    perceptual::{PerceptualNet, PerceptualNetConfig},
};
use burn::{
    config::Config,
    module::{Initializer, Module, RunningState},
    nn::{
        GaussianNoise, GaussianNoiseConfig, InstanceNorm, InstanceNormConfig, LayerNorm,
        LayerNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Distribution, Tensor,
        activation::{leaky_relu, tanh},
        backend::Backend,
        linalg::{Norm, vector_norm, vector_normalize},
        module::{conv2d, interpolate},
        ops::ConvOptions,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

const NEGATIVE_SLOPE: f64 = 0.2;

fn spectral_norm<B: Backend>(
    weight: Tensor<B, 4>,
    u: &Tensor<B, 2>,
) -> (Tensor<B, 4>, Tensor<B, 2>) {
    let [oc, ic, kh, kw] = weight.dims();
    let w_mat = weight.clone().reshape([oc, ic * kh * kw]);

    let v = u.clone().matmul(w_mat.clone());
    let v = vector_normalize(v, Norm::L2, 1, 1e-12);

    let u_new = w_mat.clone().matmul(v.clone().transpose()).transpose();
    let u_new = vector_normalize(u_new, Norm::L2, 1, 1e-12);

    let sigma = u_new.clone().matmul(w_mat).matmul(v.transpose());
    let weight_sn = weight.div(sigma.reshape([1, 1, 1, 1]));

    (weight_sn, u_new.detach())
}

#[derive(Module, Debug)]
pub struct DiscriminatorBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: Option<LayerNorm<B>>,
    u: RunningState<Tensor<B, 2>>,
}

#[derive(Config, Debug)]
pub struct DiscriminatorBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    use_norm: bool,
}

impl DiscriminatorBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DiscriminatorBlock<B> {
        let leaky_gain = (2.0 / (1.0 + NEGATIVE_SLOPE.powi(2))).sqrt();

        DiscriminatorBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false) // before InstanceNorm
                .with_initializer(Initializer::KaimingNormal {
                    gain: leaky_gain,
                    fan_out_only: false,
                })
                .init(device),
            norm: self
                .use_norm
                .then(|| LayerNormConfig::new(self.out_channels).init(device)),
            u: RunningState::new(Tensor::<B, 2>::random(
                [1, self.out_channels],
                Distribution::Default,
                device,
            )),
        }
    }
}

impl<B: Backend> DiscriminatorBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let weight = self.conv.weight.val();
        let u_current = self.u.value();

        let (sn_weight, u_next) = spectral_norm(weight, &u_current);

        self.u.update(u_next);

        let x = conv2d(
            input,
            sn_weight,
            self.conv.bias.as_ref().map(|b| b.val()),
            ConvOptions::new(
                self.conv.stride,
                [1, 1],
                self.conv.dilation,
                self.conv.groups,
            ),
        );

        let x = self.norm.as_ref().map_or(x.clone(), |norm| {
            let x = x.swap_dims(1, 3).swap_dims(1, 2);
            let x = norm.forward(x);
            x.swap_dims(1, 2).swap_dims(1, 3)
        });
        leaky_relu(x, NEGATIVE_SLOPE)
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
        let leaky_gain = (2.0 / (1.0 + NEGATIVE_SLOPE.powi(2))).sqrt();

        GeneratorConvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false) // before InstanceNorm
                .with_initializer(Initializer::KaimingNormal {
                    gain: leaky_gain,
                    fan_out_only: false,
                })
                .init(device),
            norm: InstanceNormConfig::new(self.out_channels).init(device),
        }
    }
}

impl<B: Backend> GeneratorConvBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        leaky_relu(x, NEGATIVE_SLOPE)
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
        let leaky_gain = (2.0 / (1.0 + NEGATIVE_SLOPE.powi(2))).sqrt();

        GeneratorDeconvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false) // before InstanceNorm
                .with_initializer(Initializer::KaimingNormal {
                    gain: leaky_gain,
                    fan_out_only: false,
                })
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
        leaky_relu(x, NEGATIVE_SLOPE)
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
                .with_bias(true)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
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
        let x = FlashAttentionV3::forward(x, s3.clone(), s3.clone(), None, false);
        let x = Tensor::cat(vec![x, s3], 1);

        let x = self.dec1.forward(x);
        let x = FlashAttentionV3::forward(x, s2.clone(), s2.clone(), None, false);
        let x = Tensor::cat(vec![x, s2], 1);

        let x = self.dec2.forward(x);
        let x = FlashAttentionV3::forward(x, s1.clone(), s1.clone(), None, false);
        let x = Tensor::cat(vec![x, s1], 1);

        let x = self.dec3.forward(x);
        let x = self.final_conv.forward(x);

        tanh(x)
    }
}

#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    hidden_channels: usize,
}

impl DiscriminatorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        let c = self.hidden_channels;

        Discriminator {
            noise: GaussianNoiseConfig::new(0.2).init(),
            conv1: DiscriminatorBlockConfig::new(CHANNELS, c, 2, false).init(device),
            conv2: DiscriminatorBlockConfig::new(c, c * 2, 2, true).init(device),
            conv3: DiscriminatorBlockConfig::new(c * 2, c * 4, 2, true).init(device),
            final_conv: Conv2dConfig::new([c * 4, 1], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 1.0,
                    fan_out_only: false,
                })
                .init(device),
            final_u: Tensor::<B, 2>::random([1, 1], Distribution::Default, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    pub noise: GaussianNoise,
    pub conv1: DiscriminatorBlock<B>,
    pub conv2: DiscriminatorBlock<B>,
    pub conv3: DiscriminatorBlock<B>,
    pub final_conv: Conv2d<B>,
    pub final_u: Tensor<B, 2>,
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.noise.forward(images);

        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);

        let (sn_weight, _u_final) =
            spectral_norm(self.final_conv.weight.val(), &self.final_u.clone());

        let x = conv2d(
            x,
            sn_weight,
            self.final_conv.bias.as_ref().map(|b| b.val()),
            ConvOptions::new(
                self.final_conv.stride,
                [1, 1],
                self.final_conv.dilation,
                self.final_conv.groups,
            ),
        );

        x.mean_dims(&[2, 3]).squeeze_dims(&[2, 3])
    }
}

#[derive(Config, Debug)]
pub struct NetworkConfig {
    #[config(default = 128)]
    pub hidden_channels: usize,
}

impl NetworkConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> (Generator<B>, Discriminator<B>, PerceptualNet<B>) {
        let c = self.hidden_channels;

        let generator = GeneratorConfig::new(c).init(device);
        let discriminator = DiscriminatorConfig::new(c).init(device);
        let perceptual_net = PerceptualNetConfig::new(CHANNELS, c).init(device);

        (generator, discriminator, perceptual_net)
    }
}
