use crate::consts::ARTIFACT_DIR;
use crate::files::distort;
use crate::files::{PIXEL_MID, norm};
use crate::training::TrainingConfig;
use burn::{
    config::Config,
    module::Module,
    record::CompactRecorder,
    tensor::{Tensor, TensorData, backend::Backend},
};
use image::{GrayImage, imageops::replace, open};
use std::fs::{create_dir_all, read_dir};

const TEST_DIR: &str = "test";
const RECONSTRUCTED_DIR: &str = "reconstructed";

pub fn infer<B: Backend>(device: &B::Device) {
    let config =
        TrainingConfig::load(format!("{ARTIFACT_DIR}/config.json")).expect("Config not found");
    let (generator, _, _) = config.model.init::<B>(device);

    let latest_path = read_dir(ARTIFACT_DIR)
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("generator"))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
        .unwrap();

    let generator = generator
        .load_file(latest_path, &CompactRecorder::new(), device)
        .unwrap();

    create_dir_all(RECONSTRUCTED_DIR).ok();
    let entries = read_dir(TEST_DIR).unwrap().flatten();

    for (idx, entry) in entries.enumerate() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        dbg!(&path);

        let file_name = path.file_stem().and_then(|s| s.to_str()).unwrap();
        let is_original = file_name.ends_with("original");

        let img = open(&path).unwrap().to_luma8();
        let (w, h) = img.dimensions();

        let processed_img = if is_original { distort(img) } else { img };

        let input_tensor = Tensor::<B, 4>::from_data(
            TensorData::new(norm(processed_img.as_raw()), [1, 1, h as usize, w as usize]),
            device,
        );

        let reconstructed = generator.forward(input_tensor.clone());
        save_sample::<B>(idx, input_tensor, reconstructed);
    }
}

fn save_sample<B: Backend>(iter: usize, input: Tensor<B, 4>, recon: Tensor<B, 4>) {
    let [_, _, h, w] = input.dims();
    let denorm = |t: Tensor<B, 4>| -> Vec<u8> {
        t.slice([0..1, 0..1])
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|v| ((v + 1.0) * PIXEL_MID).clamp(0.0, 255.0) as u8)
            .collect()
    };

    let mut combined = GrayImage::new(w as u32 * 2, h as u32);
    let img_in = GrayImage::from_raw(w as u32, h as u32, denorm(input)).unwrap();
    let img_out = GrayImage::from_raw(w as u32, h as u32, denorm(recon)).unwrap();

    replace(&mut combined, &img_in, 0, 0);
    replace(&mut combined, &img_out, w as i64, 0);

    combined
        .save(format!("{RECONSTRUCTED_DIR}/output_{iter}.png"))
        .unwrap();
}
