use crate::consts::CHANNELS;
use crate::nets::perceptual::{PerceptualNet, PerceptualNetConfig};
use burn::{
    config::Config,
    module::{Initializer, Module, RunningState},
    nn::{
        GaussianNoise, GaussianNoiseConfig, GroupNorm, GroupNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Distribution, Tensor,
        activation::{leaky_relu, tanh},
        backend::Backend,
        linalg::{Norm, vector_normalize},
        module::{attention, conv2d, interpolate},
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
    let v = vector_normalize(u.clone().matmul(w_mat.clone()), Norm::L2, 1, 1e-5);
    let u_new = vector_normalize(
        v.clone().matmul(w_mat.clone().transpose()),
        Norm::L2,
        1,
        1e-5,
    );
    let sigma = u_new.clone().matmul(w_mat).matmul(v.transpose());
    let weight_sn = weight.div(sigma.reshape([1, 1, 1, 1]).add_scalar(1e-5));

    (weight_sn, u_new.detach())
}

#[derive(Module, Debug)]
pub struct DiscriminatorBlock<B: Backend> {
    conv: Conv2d<B>,
    u: RunningState<Tensor<B, 2>>,
}

#[derive(Config, Debug)]
pub struct DiscriminatorBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
}

impl DiscriminatorBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DiscriminatorBlock<B> {
        let leaky_gain = (2.0 / (1.0 + NEGATIVE_SLOPE.powi(2))).sqrt();

        DiscriminatorBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(true)
                .with_initializer(Initializer::KaimingNormal {
                    gain: leaky_gain,
                    fan_out_only: false,
                })
                .init(device),
            u: RunningState::new(Tensor::<B, 2>::random(
                [1, self.out_channels],
                Distribution::Default,
                device,
            )),
        }
    }
}

impl<B: Backend> DiscriminatorBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>, activation: bool) -> Tensor<B, 4> {
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

        if activation {
            leaky_relu(x, NEGATIVE_SLOPE)
        } else {
            x
        }
    }
}

#[derive(Module, Debug)]
pub struct GeneratorConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: GroupNorm<B>,
}

#[derive(Config, Debug)]
pub struct GeneratorConvBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    #[config(default = 8)]
    groups: usize,
}

impl GeneratorConvBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> GeneratorConvBlock<B> {
        let leaky_gain = (2.0 / (1.0 + NEGATIVE_SLOPE.powi(2))).sqrt();

        GeneratorConvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false) // before InstanceNorm
                .with_initializer(Initializer::KaimingNormal {
                    gain: leaky_gain,
                    fan_out_only: false,
                })
                .init(device),
            norm: GroupNormConfig::new(self.groups, self.out_channels).init(device),
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
    norm: GroupNorm<B>,
}

#[derive(Config, Debug)]
pub struct GeneratorDeconvBlockConfig {
    in_channels: usize,
    out_channels: usize,
    #[config(default = 8)]
    groups: usize,
}

impl GeneratorDeconvBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeneratorDeconvBlock<B> {
        let leaky_gain = (2.0 / (1.0 + NEGATIVE_SLOPE.powi(2))).sqrt();

        GeneratorDeconvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false) // before InstanceNorm
                .with_initializer(Initializer::KaimingNormal {
                    gain: leaky_gain,
                    fan_out_only: false,
                })
                .init(device),
            norm: GroupNormConfig::new(self.groups, self.out_channels).init(device),
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
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(true)
                .with_initializer(Initializer::XavierUniform { gain: 1.0 })
                .init(device),
        }
    }
}

impl<B: Backend> Generator<B> {
    fn attention_layer(&self, query: Tensor<B, 4>, context: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = query.dims();

        let d_k = 16;
        let num_heads = c / d_k;

        assert_eq!(c % d_k, 0);

        let prep = |t: Tensor<B, 4>| t.reshape([b, num_heads, d_k, h * w]).swap_dims(2, 3);

        let out = attention(prep(query), prep(context.clone()), prep(context), None);

        out.swap_dims(2, 3).reshape([b, c, h, w])
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let s1 = self.enc1.forward(input);
        let s2 = self.enc2.forward(s1.clone());
        let s3 = self.enc3.forward(s2.clone());
        let s4 = self.enc4.forward(s3.clone());

        let x = self.dec4.forward(s4);
        let x = self.attention_layer(x, s3.clone());
        let x = Tensor::cat(vec![x, s3], 1);

        let x = self.dec1.forward(x);
        // let x = self.attention_layer(x, s2.clone());
        let x = Tensor::cat(vec![x, s2], 1);

        let x = self.dec2.forward(x);
        // let x = self.attention_layer(x, s1.clone());
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
            conv1: DiscriminatorBlockConfig::new(CHANNELS, c, 2).init(device),
            conv2: DiscriminatorBlockConfig::new(c, c * 2, 2).init(device),
            conv3: DiscriminatorBlockConfig::new(c * 2, c * 4, 2).init(device),
            final_block: DiscriminatorBlockConfig::new(c * 4, 1, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    pub noise: GaussianNoise,
    pub conv1: DiscriminatorBlock<B>,
    pub conv2: DiscriminatorBlock<B>,
    pub conv3: DiscriminatorBlock<B>,
    pub final_block: DiscriminatorBlock<B>,
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.noise.forward(images);

        let x = self.conv1.forward(x, true);
        let x = self.conv2.forward(x, true);
        let x = self.conv3.forward(x, true);

        let x = self.final_block.forward(x, false);

        x.mean_dims(&[2, 3]).squeeze_dims(&[2, 3])
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
        let discriminator = DiscriminatorConfig::new(c).init(device);
        let perceptual_net = PerceptualNetConfig::new(CHANNELS, c).init(device);

        (generator, discriminator, perceptual_net)
    }
}
