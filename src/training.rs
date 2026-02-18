use crate::dataset::ImageBatcher;
use crate::files::ImagePair;
use crate::network::{Discriminator, NetworkConfig};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{Distribution, Tensor, backend::AutodiffBackend, module::conv2d, ops::ConvOptions},
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
    #[config(default = 3e-4)]
    pub discriminator_lr: f64,
    #[config(default = 1e-4)]
    pub generator_lr: f64,
    #[config(default = 5)]
    pub num_critic: usize,
    #[config(default = 1.0)]
    pub lambda_adv: f32,
    #[config(default = 20.0)]
    pub lambda_l1: f32,
    #[config(default = 200.0)]
    pub lambda_perceptual: f32,
    #[config(default = 10.0)]
    pub lambda_sobel: f32,
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

fn gradient_penalty<B: AutodiffBackend>(
    discriminator: &Discriminator<B>,
    real_samples: Tensor<B, 4>,
    fake_samples: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let [batch_size, channels, height, width] = real_samples.dims();
    let device = real_samples.device();

    let alpha = Tensor::<B, 4>::random([batch_size, 1, 1, 1], Distribution::Default, &device);

    let interpolates = alpha.clone() * real_samples + ((-alpha + 1.0) * fake_samples);
    let interpolates = interpolates.require_grad();

    let score = discriminator.forward(interpolates.clone());
    let grads = score.sum().backward();

    let gradients = interpolates.grad(&grads).unwrap();

    let norm = gradients
        .square()
        .sum_dims(&[1, 2, 3])
        .add_scalar(1e-8)
        .sqrt();
    let penalty = (norm - 1.0).square().mean();

    Tensor::<B, 1>::from_inner(penalty)
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

    let mse_loss = MseLoss::new();
    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.5)
        .with_beta_2(0.9)
        .with_weight_decay(None);
    let config = TrainingConfig::new(NetworkConfig::new(), optimizer_config);

    config.save(format!("{ARTIFACT_DIR}/config.json")).unwrap();
    B::seed(&device, config.seed);

    let (mut generator, mut discriminator, perceptual_net) = config.model.init::<B>(&device);
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

                let gp =
                    gradient_penalty(&discriminator, batch.original.clone(), fake_images.clone());

                let loss_d = loss_wasserstein + gp.mul_scalar(10.0);

                // if loss_d.clone().is_nan().any().into_scalar().to_bool() {
                //     panic!(
                //         "NaN detected in Loss D! Iteration: {}, Epoch: {}",
                //         iteration, epoch
                //     );
                // }

                if i == 0 && iteration % 10 == 0 {
                    println!("Loss D: {}", loss_d.clone().into_scalar().to_f32());
                }

                let grads = loss_d.backward();
                let grads = GradientsParams::from_grads(grads, &discriminator);

                discriminator = optimizer_d.step(config.discriminator_lr, discriminator, grads);
            }

            let reconstructed = generator.forward(batch.edited.clone());

            // if reconstructed.clone().is_nan().any().into_scalar().to_bool() {
            //     panic!("Grave Error: Generator produced NaNs in the image pixels!");
            // }

            let score_reconstructed = discriminator.forward(reconstructed.clone());
            // if score_reconstructed
            //     .clone()
            //     .is_nan()
            //     .any()
            //     .into_scalar()
            //     .to_bool()
            // {
            //     panic!("Grave Error: Discriminator Score is NaN! Check Gradient Penalty.");
            // }
            let loss_adv = -score_reconstructed.mean();

            let loss_l1 = (reconstructed.clone() - batch.original.clone())
                .abs()
                .mean();

            let (f_real1, f_real2, f_real3) = perceptual_net.forward(batch.original.clone());
            let (f_fake1, f_fake2, f_fake3) = perceptual_net.forward(reconstructed.clone());

            let loss_perceptual = mse_loss.forward(f_fake1, f_real1.detach(), Reduction::Mean)
                + mse_loss.forward(f_fake2, f_real2.detach(), Reduction::Mean)
                + mse_loss.forward(f_fake3, f_real3.detach(), Reduction::Mean);

            let edges_real = sobel(batch.original.clone(), &device);
            let edges_fake = sobel(reconstructed.clone(), &device);
            let loss_sobel = (edges_fake - edges_real).abs().mean();

            let loss_g = (loss_adv.clone() * config.lambda_adv)
                + (loss_l1.clone() * config.lambda_l1)
                + (loss_perceptual.clone() * config.lambda_perceptual)
                + (loss_sobel.clone() * config.lambda_sobel);

            // if loss_g.clone().is_nan().any().into_scalar().to_bool() {
            //     panic!(
            //         "NaN detected in Loss G! Iteration: {}, Epoch: {}",
            //         iteration, epoch
            //     );
            // }

            println!(
                "{:.2} {:.2} {:.2} {:.2}",
                (loss_adv.into_scalar().to_f32() * config.lambda_adv),
                (loss_l1.into_scalar().to_f32() * config.lambda_l1),
                (loss_perceptual.into_scalar().to_f32() * config.lambda_perceptual),
                (loss_sobel.into_scalar().to_f32() * config.lambda_sobel)
            );

            {
                let grads = loss_g.backward();
                let grads = GradientsParams::from_grads(grads, &generator);
                generator = optimizer_g.step(config.generator_lr, generator, grads);
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
