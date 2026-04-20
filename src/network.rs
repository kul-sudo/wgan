use crate::consts::CHANNELS;
use burn::{
    config::Config,
    module::{Initializer, Module, RunningState},
    nn::{
        Dropout, DropoutConfig, GaussianNoise, GaussianNoiseConfig, InstanceNorm,
        InstanceNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    tensor::{
        Distribution, Tensor,
        activation::{leaky_relu, mish, tanh},
        backend::Backend,
        linalg::{Norm, vector_normalize},
        module::conv2d,
        ops::ConvOptions,
    },
};
use std::f64::consts::SQRT_2;

const NEGATIVE_SLOPE: f64 = 0.2;

fn spectral_norm<B: Backend>(
    weight: Tensor<B, 4>,
    u: Tensor<B, 2>,
    v: Tensor<B, 2>,
    n_power_iterations: usize,
    eps: f64,
) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
    let [oc, ic, kh, kw] = weight.dims();
    let weight_mat = weight.clone().detach().reshape([oc, ic * kh * kw]);

    let mut u_vec = u;
    let mut v_vec = v;

    for _ in 0..n_power_iterations {
        v_vec = vector_normalize(u_vec.matmul(weight_mat.clone()), Norm::L2, 1, eps).detach();
        u_vec = vector_normalize(
            v_vec.clone().matmul(weight_mat.clone().transpose()),
            Norm::L2,
            1,
            eps,
        )
        .detach();
    }

    let sigma = u_vec
        .clone()
        .matmul(weight_mat)
        .matmul(v_vec.clone().transpose());

    let sigma_val = sigma.reshape([1, 1, 1, 1]).abs();
    let normalized_weight = weight.div(sigma_val.add_scalar(1e-8));

    (normalized_weight, u_vec.detach(), v_vec.detach())
}

// Generator
#[derive(Module, Debug)]
pub struct GeneratorConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: Option<InstanceNorm<B>>,
}

#[derive(Config, Debug)]
pub struct GeneratorConvBlockConfig {
    in_channels: usize,
    out_channels: usize,
    stride: usize,
}

impl GeneratorConvBlockConfig {
    fn init<B: Backend>(&self, use_norm: bool, device: &B::Device) -> GeneratorConvBlock<B> {
        GeneratorConvBlock {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [4, 4])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(!use_norm)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            norm: use_norm.then(|| InstanceNormConfig::new(self.out_channels).init(device)),
        }
    }
}

impl<B: Backend> GeneratorConvBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = match &self.norm {
            Some(n) => n.forward(x),
            None => x,
        };
        mish(x)
    }
}

#[derive(Module, Debug)]
pub struct GeneratorDeconvBlock<B: Backend> {
    deconv: ConvTranspose2d<B>,
    norm: Option<InstanceNorm<B>>,
}

#[derive(Config, Debug)]
pub struct GeneratorDeconvBlockConfig {
    in_channels: usize,
    out_channels: usize,
}

impl GeneratorDeconvBlockConfig {
    pub fn init<B: Backend>(&self, use_norm: bool, device: &B::Device) -> GeneratorDeconvBlock<B> {
        GeneratorDeconvBlock {
            deconv: ConvTranspose2dConfig::new([self.in_channels, self.out_channels], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_bias(!use_norm)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            norm: use_norm.then(|| InstanceNormConfig::new(self.out_channels).init(device)),
        }
    }
}

impl<B: Backend> GeneratorDeconvBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.deconv.forward(input);
        let x = match &self.norm {
            Some(n) => n.forward(x),
            None => x,
        };
        mish(x)
    }
}

// Generator: Residual net
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    norm1: InstanceNorm<B>,
    norm2: InstanceNorm<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct ResBlockConfig {
    channels: usize,
}

impl ResBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        ResBlock {
            conv1: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            norm1: InstanceNormConfig::new(self.channels).init(device),
            conv2: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false)
                .with_initializer(Initializer::KaimingNormal {
                    gain: SQRT_2,
                    fan_out_only: false,
                })
                .init(device),
            norm2: InstanceNormConfig::new(self.channels).init(device),
            dropout: DropoutConfig::new(0.5).init(),
        }
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input.clone());
        let x = self.norm1.forward(x);
        let x = mish(x);

        let x = self.dropout.forward(x);

        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);

        x + input
    }
}

#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    pub enc1: GeneratorConvBlock<B>,
    pub enc2: GeneratorConvBlock<B>,
    pub enc3: GeneratorConvBlock<B>,
    pub enc4: GeneratorConvBlock<B>,
    pub noise: GaussianNoise,
    pub res_blocks: Vec<ResBlock<B>>,
    pub dec4: GeneratorDeconvBlock<B>,
    pub dec3: GeneratorDeconvBlock<B>,
    pub dec2: GeneratorDeconvBlock<B>,
    pub dec1: GeneratorDeconvBlock<B>,
    pub final_conv: Conv2d<B>,
}

#[derive(Config, Debug)]
pub struct GeneratorConfig {
    hidden_channels: usize,
    #[config(default = 9)]
    res_blocks: usize,
}

impl GeneratorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        let c = self.hidden_channels;

        Generator {
            enc1: GeneratorConvBlockConfig::new(CHANNELS, c, 2).init(false, device),
            enc2: GeneratorConvBlockConfig::new(c, c * 2, 2).init(true, device),
            enc3: GeneratorConvBlockConfig::new(c * 2, c * 4, 2).init(true, device),
            enc4: GeneratorConvBlockConfig::new(c * 4, c * 8, 2).init(true, device),
            noise: GaussianNoiseConfig::new(0.2).init(),
            res_blocks: (0..self.res_blocks)
                .map(|_| ResBlockConfig::new(c * 8).init(device))
                .collect(),
            dec4: GeneratorDeconvBlockConfig::new(c * 16, c * 4).init(true, device),
            dec3: GeneratorDeconvBlockConfig::new(c * 4, c * 2).init(true, device),
            dec2: GeneratorDeconvBlockConfig::new(c * 2, c * 2).init(true, device),
            dec1: GeneratorDeconvBlockConfig::new(c * 2, c).init(false, device),
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
        let s2 = self.enc2.forward(s1);
        let s3 = self.enc3.forward(s2);
        let s4 = self.enc4.forward(s3);

        let x = self
            .res_blocks
            .iter()
            .fold(self.noise.forward(s4.clone()), |acc, block| {
                block.forward(acc)
            });

        let skip_noisy = self.noise.forward(s4);
        let x = Tensor::cat(vec![x, skip_noisy], 1);

        let x = self.dec4.forward(x);
        let x = self.dec3.forward(x);
        let x = self.dec2.forward(x);
        let x = self.dec1.forward(x);

        tanh(self.final_conv.forward(x))
    }
}

// Discriminator
#[derive(Module, Debug)]
pub struct DiscriminatorBlock<B: Backend> {
    conv: Conv2d<B>,
    u: RunningState<Tensor<B, 2>>,
    v: RunningState<Tensor<B, 2>>,
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
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [4, 4])
                .with_stride([self.stride, self.stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(true)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            u: RunningState::new(Tensor::<B, 2>::random(
                [1, self.out_channels],
                Distribution::Normal(0.0, 1.0),
                device,
            )),
            v: RunningState::new(Tensor::<B, 2>::random(
                [1, self.in_channels * 4_usize.pow(2)],
                Distribution::Normal(0.0, 1.0),
                device,
            )),
        }
    }
}

impl<B: Backend> DiscriminatorBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>, use_activation: bool) -> Tensor<B, 4> {
        let weight = self.conv.weight.val();
        let u_current = self.u.value();
        let v_current = self.v.value();

        let (sn_weight, u_next, v_next) = spectral_norm(weight, u_current, v_current, 1, 1e-12);

        if B::ad_enabled() {
            self.u.update(u_next);
            self.v.update(v_next);
        }

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

        if use_activation {
            leaky_relu(x, NEGATIVE_SLOPE)
        } else {
            x
        }
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
            noise: GaussianNoiseConfig::new(0.01).init(),
            block1: DiscriminatorBlockConfig::new(CHANNELS, c, 2).init(device),
            block2: DiscriminatorBlockConfig::new(c, c * 2, 2).init(device),
            block3: DiscriminatorBlockConfig::new(c * 2, c * 4, 2).init(device),
            block4: DiscriminatorBlockConfig::new(c * 4, c * 8, 2).init(device),
            final_block: DiscriminatorBlockConfig::new(c * 8, 1, 1).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    pub noise: GaussianNoise,
    pub block1: DiscriminatorBlock<B>,
    pub block2: DiscriminatorBlock<B>,
    pub block3: DiscriminatorBlock<B>,
    pub block4: DiscriminatorBlock<B>,
    pub final_block: DiscriminatorBlock<B>,
}

impl<B: Backend> Discriminator<B> {
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.noise.forward(images);

        let x = self.block1.forward(x, true);
        let x = self.block2.forward(x, true);
        let x = self.block3.forward(x, true);
        let x = self.block4.forward(x, true);

        self.final_block.forward(x, false)
    }

    pub fn forward_with_features(&self, images: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut features = vec![];
        let x = self.noise.forward(images);

        let x = self.block1.forward(x, true);
        features.push(x.clone());

        let x = self.block2.forward(x, true);
        features.push(x.clone());

        let x = self.block3.forward(x, true);
        features.push(x.clone());

        let x = self.block4.forward(x, true);
        features.push(x.clone());

        let out = self.final_block.forward(x, false);

        (out, features)
    }
}

#[derive(Config, Debug)]
pub struct NetworkConfig {
    #[config(default = 128)]
    pub hidden_channels: usize,
}

impl NetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> (Generator<B>, Discriminator<B>) {
        let c = self.hidden_channels;

        let generator = GeneratorConfig::new(c).init(device);
        let discriminator = DiscriminatorConfig::new(c).init(device);

        (generator, discriminator)
    }
}
