use burn::{
    config::Config,
    module::Initializer,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Tensor, activation::leaky_relu, backend::Backend},
};

const NEGATIVE_SLOPE: f64 = 0.2;

#[derive(Config, Debug)]
pub struct PerceptualNetConfig {
    pub in_channels: usize,
    pub hidden_channels: usize,
}

impl PerceptualNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PerceptualNet<B> {
        let h = self.hidden_channels;

        let leaky_gain = (2.0 / (1.0 + NEGATIVE_SLOPE.powi(2))).sqrt();

        let conv1 = Conv2dConfig::new([self.in_channels, h], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(Initializer::KaimingNormal {
                gain: leaky_gain,
                fan_out_only: false,
            })
            .init(device);
        let conv2 = Conv2dConfig::new([h, h * 2], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(Initializer::KaimingNormal {
                gain: leaky_gain,
                fan_out_only: false,
            })
            .init(device);
        let conv3 = Conv2dConfig::new([h * 2, h * 4], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_initializer(Initializer::KaimingNormal {
                gain: 1.0,
                fan_out_only: false,
            })
            .init(device);

        PerceptualNet {
            conv1,
            conv2,
            conv3,
        }
    }
}

#[derive(Debug, Module)]
pub struct PerceptualNet<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
}

impl<B: Backend> PerceptualNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.conv1.forward(x);
        let f1 = leaky_relu(x, NEGATIVE_SLOPE);

        let x = self.conv2.forward(f1.clone());
        let f2 = leaky_relu(x, NEGATIVE_SLOPE);

        let f3 = self.conv3.forward(f2.clone());

        (f1, f2, f3)
    }
}
