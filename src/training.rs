use crate::consts::ARTIFACT_DIR;
use crate::dataset::ImageBatcher;
use crate::files::{ImagePair, PIXEL_MID};
use crate::network::NetworkConfig;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::InMemDataset},
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::{
        Tensor, activation::relu, backend::AutodiffBackend, module::conv2d, ops::ConvOptions,
    },
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
    #[config(default = 5)]
    pub seed: u64,
    #[config(default = 4e-4)]
    pub discriminator_lr: f64,
    #[config(default = 1e-4)]
    pub generator_lr: f64,
    #[config(default = 1)]
    pub num_critic: usize,
    #[config(default = 20.0)]
    pub lambda_adv: f32,
    #[config(default = 20.0)]
    pub lambda_l1: f32,
    #[config(default = 25.0)]
    pub lambda_perceptual: f32,
    #[config(default = 4.0)]
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

pub fn train<B: AutodiffBackend>(items: &mut [ImagePair], device: B::Device) {
    create_dir_all(ARTIFACT_DIR).unwrap();

    let optimizer_config = AdamConfig::new()
        .with_beta_1(0.0)
        .with_beta_2(0.9)
        // .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
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

    for epoch in 1..=config.num_epochs {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let reconstructed = generator.forward(batch.edited.clone());
            let fake = reconstructed.clone().detach();

            for _ in 0..config.num_critic {
                let score_fake = discriminator.forward(fake.clone());
                let score_real = discriminator.forward(batch.original.clone());

                let loss_d = relu(1.0 + score_fake).mean() + relu(1.0 - score_real).mean();

                let grads = loss_d.backward();
                let grads = GradientsParams::from_grads(grads, &discriminator);
                discriminator = optimizer_d.step(config.discriminator_lr, discriminator, grads);
            }

            let score_reconstructed = discriminator.forward(reconstructed.clone());
            let loss_adv = -score_reconstructed.mean();

            let loss_l1 = (reconstructed.clone() - batch.original.clone())
                .abs()
                .mean();

            let (f_real1, f_real2, f_real3) = perceptual_net.forward(batch.original.clone());
            let (f_fake1, f_fake2, f_fake3) = perceptual_net.forward(reconstructed.clone());

            let loss_perceptual = (f_fake1 - f_real1.detach()).abs().mean()
                + (f_fake2 - f_real2.detach()).abs().mean()
                + (f_fake3 - f_real3.detach()).abs().mean();

            let edges_real = sobel(batch.original.clone(), &device);
            let edges_fake = sobel(reconstructed.clone(), &device);
            let loss_sobel = (edges_fake - edges_real).abs().mean();

            let loss_g = (loss_adv.clone() * config.lambda_adv)
                + (loss_l1.clone() * config.lambda_l1)
                + (loss_perceptual.clone() * config.lambda_perceptual)
                + (loss_sobel.clone() * config.lambda_sobel);

            if iteration % 10 == 0 {
                println!(
                    "Epoch: {} Adv: {:.2} L1: {:.2} Perceptual: {:.2} Sobel: {:.2} Total G: {:.4}",
                    epoch,
                    (loss_adv.clone().into_scalar().to_f32() * config.lambda_adv),
                    (loss_l1.clone().into_scalar().to_f32() * config.lambda_l1),
                    (loss_perceptual.clone().into_scalar().to_f32() * config.lambda_perceptual),
                    (loss_sobel.clone().into_scalar().to_f32() * config.lambda_sobel),
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
