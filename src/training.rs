use crate::dataset::ImageBatcher;
use crate::files::ImagePair;
use crate::network::NetworkConfig;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::{Module, ModuleMapper, Param},
    nn::loss::{MseLoss, Reduction},
    optim::{GradientsParams, Optimizer, RmsPropConfig},
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, module::conv2d, ops::ConvOptions},
};
use image::{GrayImage, Luma};
use std::fs::create_dir_all;

const ARTIFACT_DIR: &str = "artifact";

#[derive(Debug, Config)]
pub struct TrainingConfig {
    pub model: NetworkConfig,
    pub optimizer: RmsPropConfig,
    #[config(default = 500)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 5)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    #[config(default = 5)]
    pub num_critic: usize,
}

#[derive(Module, Clone, Debug)]
pub struct Clip {
    pub min: f32,
    pub max: f32,
}

impl<B: AutodiffBackend> ModuleMapper<B> for Clip {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let is_require_grad = tensor.is_require_grad();

        let mut tensor = Tensor::from_inner(tensor.inner().clamp(self.min, self.max));

        if is_require_grad {
            tensor = tensor.require_grad();
        }
        Param::from_mapped_value(id, tensor, mapper)
    }
}

fn sobel<B: Backend>(x: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 4> {
    let opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);

    let gx = Tensor::<B, 4>::from_data(
        [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
        device,
    );
    let gy = Tensor::<B, 4>::from_data(
        [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]],
        device,
    );

    let x_edges = conv2d(x.clone(), gx, None, opts.clone());
    let y_edges = conv2d(x, gy, None, opts);

    (x_edges.powf_scalar(2.0) + y_edges.powf_scalar(2.0) + 1e-8).sqrt()
}

fn save_samples<B: Backend>(
    epoch: usize,
    iter: usize,
    manual_input: Tensor<B, 4>,
    original_target: Tensor<B, 4>,
    reconstructed: Tensor<B, 4>,
) {
    let [_batch_size, _channels, h, w] = original_target.dims();

    let input_vec = manual_input
        .slice([0..1, 0..1])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let target_vec = original_target
        .slice([0..1, 0..1])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let recon_vec = reconstructed
        .slice([0..1, 0..1])
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let mut combined = GrayImage::new(w as u32 * 3, h as u32);

    let to_u8 = |val: f32| ((val + 1.0) * 127.5).clamp(0.0, 255.0) as u8;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;

            combined.put_pixel(x as u32, y as u32, Luma([to_u8(target_vec[idx])]));
            combined.put_pixel(x as u32 + w as u32, y as u32, Luma([to_u8(input_vec[idx])]));
            combined.put_pixel(
                x as u32 + (w as u32 * 2),
                y as u32,
                Luma([to_u8(recon_vec[idx])]),
            );
        }
    }

    let path = format!("{ARTIFACT_DIR}/comparison_e{}_i{}.png", epoch, iter);
    if let Err(e) = combined.save(&path) {
        eprintln!("Failed to save sample: {}", e);
    }
}

pub fn train<B: AutodiffBackend>(items: &mut [ImagePair], device: B::Device) {
    create_dir_all(ARTIFACT_DIR).unwrap();

    let optimizer_config = RmsPropConfig::new()
        .with_alpha(0.99)
        .with_weight_decay(None);

    let config = TrainingConfig::new(NetworkConfig::new(), optimizer_config);

    let mut clip = Clip {
        min: -0.01,
        max: 0.01,
    };

    config.save(format!("{ARTIFACT_DIR}/config.json")).unwrap();
    B::seed(&device, config.seed);

    let (mut generator, mut discriminator) = config.model.init::<B>(&device);
    let mut optimizer_g = config.optimizer.init();
    let mut optimizer_d = config.optimizer.init();

    let batcher = ImageBatcher::default();

    let dataset = InMemDataset::new(items.to_vec());
    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(2)
        .build(dataset);

    let mse = MseLoss::new();

    for epoch in 0..config.num_epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let edited = batch.edited.detach();
            let original = batch.original.detach();

            let fake_images = generator.forward(edited.clone()).detach();
            for i in 0..config.num_critic {
                let score_fake = discriminator.forward(fake_images.clone());
                let score_real = discriminator.forward(original.clone());

                let loss_d = score_fake.mean() - score_real.mean();

                if i == 0 && iteration % 10 == 0 {
                    println!("Loss D: {}", loss_d.clone().into_scalar().to_f32());
                }

                {
                    let grads = loss_d.backward();
                    let grads = GradientsParams::from_grads(grads, &discriminator);
                    discriminator = optimizer_d.step(config.lr, discriminator, grads);
                }

                discriminator = discriminator.map(&mut clip);
            }

            let reconstructed = generator.forward(edited.clone());
            let score_reconstructed = discriminator.forward(reconstructed.clone());
            let loss_adv = -score_reconstructed.mean();

            let loss_l1 = (reconstructed.clone() - original.clone()).abs().mean();

            let edges_reconstructed = sobel(reconstructed.clone(), &device);
            let edges_original = sobel(original.clone(), &device).detach();
            let loss_sobel = mse.forward(edges_reconstructed, edges_original, Reduction::Mean);

            // const LAMBDA_ADV: f32 = 20.0;
            // const LAMBDA_L1: f32 = 200.0;
            // const LAMBDA_SOBEL: f32 = 800.0;

            // const LAMBDA_ADV: f32 = 20.0;
            // const LAMBDA_L1: f32 = 150.0;
            // const LAMBDA_SOBEL: f32 = 800.0;

            const LAMBDA_ADV: f32 = 4000.0;
            const LAMBDA_L1: f32 = 100.0;
            const LAMBDA_SOBEL: f32 = 1200.0;

            let loss_g = (loss_adv.clone() * LAMBDA_ADV)
                + (loss_l1.clone() * LAMBDA_L1)
                + (loss_sobel.clone() * LAMBDA_SOBEL);

            println!(
                "{} {} {}",
                (loss_adv.into_scalar().to_f32() * LAMBDA_ADV),
                (loss_l1.into_scalar().to_f32() * LAMBDA_L1),
                (loss_sobel.into_scalar().to_f32() * LAMBDA_SOBEL)
            );

            {
                let grads = loss_g.backward();
                let grads = GradientsParams::from_grads(grads, &generator);
                generator = optimizer_g.step(config.lr, generator, grads);
            }

            if iteration % 10 == 0 {
                println!("Loss G: {}", loss_g.clone().into_scalar().to_f32());
                println!("Epoch: {}", epoch);
            }

            if iteration % 100 == 0 {
                save_samples(
                    epoch,
                    iteration,
                    edited.clone().detach(),
                    original.clone().detach(),
                    reconstructed.clone().detach(),
                );
            }
        }

        if epoch % 20 == 0 {
            generator
                .clone()
                .save_file(
                    format!("{ARTIFACT_DIR}/generator_{}", epoch),
                    &CompactRecorder::new(),
                )
                .unwrap()
        }
    }
}
