use crate::dataset::ImageBatcher;
use crate::files::ImagePair;
use crate::network::{Discriminator, NetworkConfig};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{Distribution, Tensor, backend::AutodiffBackend},
};
use image::{GrayImage, Luma};
use std::fs::create_dir_all;

const ARTIFACT_DIR: &str = "artifact";

#[derive(Debug, Config)]
pub struct TrainingConfig {
    pub model: NetworkConfig,
    pub optimizer: AdamConfig,
    #[config(default = 500)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 5)]
    pub seed: u64,
    #[config(default = 1e-5)]
    pub lr: f64,
    #[config(default = 5)]
    pub num_critic: usize,
    #[config(default = 0.1)]
    pub lambda_adv: f32,
    #[config(default = 100.0)]
    pub lambda_l1: f32,
    #[config(default = 600.0)]
    pub lambda_perceptual: f32,
}

fn gradient_penalty<B: AutodiffBackend>(
    discriminator: &Discriminator<B>,
    real: Tensor<B, 4>,
    fake: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 1> {
    let alpha = Tensor::<B, 4>::random_like(&real, Distribution::Uniform(0.0, 1.0));
    let interpolates = real.mul(alpha.clone()) + fake.mul(alpha.neg().add_scalar(1.0));

    let interpolates = interpolates.require_grad();

    let d_output = discriminator.forward(interpolates.clone());
    let grads = d_output.sum().backward();

    let grad_wrt_interp = interpolates.grad(&grads).unwrap();

    let flattened: Tensor<B::InnerBackend, 2> = grad_wrt_interp.flatten(1, 3);
    let grad_norm: Tensor<B::InnerBackend, 2> =
        (flattened.powf_scalar(2.0).sum_dim(1) + 1e-8).sqrt();
    let gradient_penalty: Tensor<B::InnerBackend, 1> =
        grad_norm.sub_scalar(1.0).powf_scalar(2.0).mean();

    gradient_penalty
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

    let optimizer_config = AdamConfig::new().with_beta_1(0.0).with_beta_2(0.9);

    let config = TrainingConfig::new(NetworkConfig::new(), optimizer_config);

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

    for epoch in 0..config.num_epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let fake_images = generator.forward(batch.edited.clone()).detach();
            for i in 0..config.num_critic {
                let score_fake = discriminator.forward(fake_images.clone());
                let score_real = discriminator.forward(batch.original.clone());

                let loss_wasserstein = score_fake.mean() - score_real.mean();

                let gp_inner =
                    gradient_penalty(&discriminator, batch.original.clone(), fake_images.clone());
                let gp = Tensor::<B, 1>::from_inner(gp_inner);

                let loss_d = loss_wasserstein + gp.mul_scalar(10.0);

                if i == 0 && iteration % 10 == 0 {
                    println!("Loss D: {}", loss_d.clone().into_scalar().to_f32());
                }

                let grads = loss_d.backward();
                let grads = GradientsParams::from_grads(grads, &discriminator);
                discriminator = optimizer_d.step(config.lr, discriminator, grads);
            }

            let reconstructed = generator.forward(batch.edited.clone());

            let (_, feat_real1, feat_real2) =
                discriminator.forward_with_features(batch.original.clone());
            let (score_reconstructed, feat_fake1, feat_fake2) =
                discriminator.forward_with_features(reconstructed.clone());

            let loss_adv = -score_reconstructed.mean();

            let loss_l1 = (reconstructed.clone() - batch.original.clone())
                .abs()
                .mean();

            let loss_feat1 = (feat_real1.detach() - feat_fake1).abs().mean();
            let loss_feat2 = (feat_real2.detach() - feat_fake2).abs().mean();
            let loss_perceptual = loss_feat1 + loss_feat2;

            let loss_g = (loss_adv.clone() * config.lambda_adv)
                + (loss_l1.clone() * config.lambda_l1)
                + (loss_perceptual.clone() * config.lambda_perceptual);

            println!(
                "{:.2} {:.2} {:.2}",
                (loss_adv.into_scalar().to_f32() * config.lambda_adv),
                (loss_l1.into_scalar().to_f32() * config.lambda_l1),
                (loss_perceptual.into_scalar().to_f32() * config.lambda_perceptual)
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
                    batch.edited,
                    batch.original,
                    reconstructed,
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
