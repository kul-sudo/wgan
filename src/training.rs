use crate::consts::ARTIFACT_DIR;
use crate::data::{PIXEL_MID, RawImage};
use crate::dataset::ImageBatcher;
use crate::network::NetworkConfig;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{Tensor, activation::relu, backend::AutodiffBackend},
};
use image::{GrayImage, imageops::replace};
use std::fs::create_dir_all;

#[derive(Debug, Config)]
pub struct TrainingConfig {
    pub model: NetworkConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1000)]
    pub num_epochs: usize,
    #[config(default = 8)]
    pub batch_size: usize,
    #[config(default = 0.1)]
    pub label_smoothing: f32,
    #[config(default = 5)]
    pub seed: u64,
    #[config(default = 4e-4)]
    pub discriminator_lr: f64,
    #[config(default = 1e-4)]
    pub generator_lr: f64,
    #[config(default = 5.0)]
    pub lambda_adv: f32,
    #[config(default = 0.0)]
    pub lambda_l1: f32,
    #[config(default = 10.0)]
    pub lambda_perceptual: f32,
}

fn save_samples<B: Backend>(
    epoch: usize,
    iter: usize,
    manual_input: Tensor<B, 4>,
    original_target: Tensor<B, 4>,
    reconstructed: Tensor<B, 4>,
) {
    let [_, _, h, w] = original_target.dims();

    let denorm = |t: Tensor<B, 4>| -> Vec<u8> {
        t.slice([0..1, 0..1])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| ((v + 1.0) * PIXEL_MID).clamp(0.0, 255.0) as u8)
            .collect()
    };

    let target_img = GrayImage::from_raw(w as u32, h as u32, denorm(original_target)).unwrap();
    let input_img = GrayImage::from_raw(w as u32, h as u32, denorm(manual_input)).unwrap();
    let reconstruction_img =
        GrayImage::from_raw(w as u32, h as u32, denorm(reconstructed)).unwrap();

    let mut combined = GrayImage::new(w as u32 * 3, h as u32);

    replace(&mut combined, &target_img, 0, 0);
    replace(&mut combined, &input_img, w as i64, 0);
    replace(&mut combined, &reconstruction_img, w as i64 * 2, 0);

    let path = format!("{ARTIFACT_DIR}/comparison_e{}_i{}.png", epoch, iter);
    if let Err(e) = combined.save(&path) {
        eprintln!("Failed to save sample: {}", e);
    }
}

pub fn train<B: AutodiffBackend>(items: &mut [RawImage], device: B::Device) {
    create_dir_all(ARTIFACT_DIR).unwrap();

    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.0)
        .with_beta_2(0.9)
        // .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .with_weight_decay(None);
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

    for epoch in 1..=config.num_epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let fake = generator.forward(batch.edited.clone()).detach();

            let score_fake = discriminator.forward(fake);
            let score_real = discriminator.forward(batch.original.clone());

            dbg!(
                score_fake.clone().mean().into_scalar().to_f32(),
                score_real.clone().mean().into_scalar().to_f32()
            );
            let loss_d: Tensor<B, 1> = relu(1.0 + score_fake).mean()
                + relu(1.0 - config.label_smoothing - score_real).mean();

            let loss_d_scalar: f32 = loss_d.clone().into_scalar().to_f32();
            println!("Loss D: {}", loss_d_scalar);

            let grads = loss_d.backward();
            let grads = GradientsParams::from_grads(grads, &discriminator);
            discriminator = optimizer_d.step(config.discriminator_lr, discriminator, grads);

            let reconstructed = generator.forward(batch.edited.clone());

            let (score_reconstructed, feat_fake) =
                discriminator.forward_with_features(reconstructed.clone());
            let (_, feat_real) = discriminator.forward_with_features(batch.original.clone());

            // Adv Loss
            let loss_adv = -score_reconstructed.mean();

            // L1 Loss
            let loss_l1 = (reconstructed.clone() - batch.original.clone())
                .abs()
                .mean();

            // Perceptual
            let mut loss_perceptual = Tensor::from_data([0.0], &device);
            for (f_f, f_r) in feat_fake.into_iter().zip(feat_real.into_iter()) {
                loss_perceptual = loss_perceptual + (f_f - f_r.detach()).abs().mean();
            }

            // Total G Loss
            let loss_g = (loss_adv.clone() * config.lambda_adv)
                + (loss_l1.clone() * config.lambda_l1)
                + (loss_perceptual.clone() * config.lambda_perceptual);

            if iteration % 10 == 0 {
                println!(
                    "Epoch: {} Adv: {:.2} L1: {:.2} FeatMatch: {:.2} Total G: {:.4}",
                    epoch,
                    (loss_adv.clone().into_scalar().to_f32() * config.lambda_adv),
                    (loss_l1.clone().into_scalar().to_f32() * config.lambda_l1),
                    (loss_perceptual.clone().into_scalar().to_f32() * config.lambda_perceptual),
                    loss_g.clone().into_scalar().to_f32()
                );
            }

            let grads = loss_g.backward();
            let grads = GradientsParams::from_grads(grads, &generator);
            generator = optimizer_g.step(config.generator_lr, generator, grads);

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
