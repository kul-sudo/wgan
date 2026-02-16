use burn::{
    config::Config,
    module::Module,
    nn::PaddingConfig2d,
    nn::conv::{Conv2d, Conv2dConfig},
    tensor::{Tensor, activation::mish, backend::Backend},
};

#[derive(Config, Debug)]
pub struct PerceptualNetConfig {
    pub in_channels: usize,
    pub hidden_channels: usize,
}

impl PerceptualNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PerceptualNet<B> {
        let h = self.hidden_channels;

        let conv1 = Conv2dConfig::new([self.in_channels, h], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv2 = Conv2dConfig::new([h, h * 2], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv3 = Conv2dConfig::new([h * 2, h * 4], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
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
        let f1 = mish(self.conv1.forward(x));
        let f2 = mish(self.conv2.forward(f1.clone()));
        let f3 = self.conv3.forward(f2.clone());

        (f1, f2, f3)
    }
}
