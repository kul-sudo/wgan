use crate::dataset::ImageBatcher;
use crate::files::ImagePair;
use crate::network::NetworkConfig;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::{Module, ModuleMapper, Param},
    optim::{GradientsParams, Optimizer, RmsPropConfig},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
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
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 5)]
    pub seed: u64,
    #[config(default = 5e-4)]
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

fn save_samples<B: Backend>(
    epoch: usize,
    iter: usize,
    manual_input: Tensor<B, 4>,
    original_target: Tensor<B, 4>,
    reconstructed: Tensor<B, 4>,
) {
    let [_, _, h, w] = original_target.dims();

    let input_vec = manual_input.into_data().to_vec::<f32>().unwrap();
    let target_vec = original_target.into_data().to_vec::<f32>().unwrap();
    let recon_vec = reconstructed.into_data().to_vec::<f32>().unwrap();

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
    combined.save(&path).ok();
}

pub fn train<B: AutodiffBackend>(items: &mut [ImagePair], device: B::Device) {
    create_dir_all(ARTIFACT_DIR).unwrap();

    let optimizer_config = RmsPropConfig::new()
        .with_alpha(0.99)
        .with_epsilon(1e-8)
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
        .num_workers(1)
        .build(dataset);

    for epoch in 0..config.num_epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let fake_images = generator.forward(batch.edited.clone()).detach();
            let score_fake = discriminator.forward(fake_images);
            let score_real = discriminator.forward(batch.original.clone());
            let loss_d = score_fake.mean() - score_real.mean();

            let d_loss_scalar = loss_d.clone().into_scalar();

            let grads = loss_d.backward();
            let grads = GradientsParams::from_grads(grads, &discriminator);
            discriminator = optimizer_d.step(config.lr, discriminator, grads);
            discriminator = discriminator.map(&mut clip);

            if iteration % config.num_critic == 0 {
                let reconstructed = generator.forward(batch.edited.clone());
                let score_recon = discriminator.forward(reconstructed.clone());
                let loss_adv = -score_recon.mean();
                let loss_l1 = (reconstructed.clone() - batch.original.clone())
                    .abs()
                    .mean();

                let loss_g = loss_adv.clone() + (loss_l1.clone() * 100.0);

                let g_loss_scalar = loss_g.clone().into_scalar();

                let grads = loss_g.backward();
                let grads = GradientsParams::from_grads(grads, &generator);
                generator = optimizer_g.step(config.lr, generator, grads);

                let batch_num = (dataloader_train.num_items() as f32 / config.batch_size as f32)
                    .ceil() as usize;

                if iteration % 100 == 0 {
                    save_samples(
                        epoch,
                        iteration,
                        batch.edited.clone(),
                        batch.original.clone(),
                        reconstructed.clone(),
                    );
                }

                println!(
                    "[Epoch {}/{}] [Batch {}/{}] [D: {:.4}] [G: {:.4}]",
                    epoch + 1,
                    config.num_epochs,
                    iteration,
                    batch_num,
                    d_loss_scalar,
                    g_loss_scalar
                );
            }
        }
    }

    generator
        .save_file(format!("{ARTIFACT_DIR}/generator"), &CompactRecorder::new())
        .expect("Failed to save generator");
}
