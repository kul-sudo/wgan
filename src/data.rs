use image::{DynamicImage, GrayImage, open};
use rand::{RngExt, rng};
use std::{
    fs::{create_dir_all, read_dir},
    path::Path,
};

#[derive(Clone, Debug)]
pub struct RawImage {
    pub original: GrayImage,
}

const INPUT_DIR: &str = "images";
const OUTPUT_DIR: &str = "edited";

pub const PIXEL_MAX: f32 = 255.0;
pub const PIXEL_MID: f32 = PIXEL_MAX / 2.0;

pub fn norm(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as f32 / PIXEL_MID) - 1.0).collect()
}

pub fn distort(mut luma: GrayImage) -> GrayImage {
    let mut r = rng();
    let a = r.random_range(1.0..3.0);
    let b = r.random_range(0.0..4.0);
    let c = r.random_range(0.0..2.0);
    let d = r.random_range(1.0..8.0);
    let f = r.random_range(1.0..9.0);
    let k = r.random_range(0.94..0.99);
    let threshold = r.random_range(0.08..0.16);

    for p in luma.pixels_mut() {
        let l = p[0] as f32 / PIXEL_MAX;
        let effect = if l < threshold {
            a
        } else {
            (l * f).powf(b).max(1.0)
        };

        let val = l * effect;

        p[0] = if val >= k {
            255
        } else {
            (val * 255.0).min(255.0) as u8
        };
    }

    DynamicImage::ImageLuma8(luma)
        .blur(c)
        .adjust_contrast(d)
        .to_luma8()
}

pub fn files_init() -> Vec<RawImage> {
    create_dir_all(OUTPUT_DIR).unwrap();

    let entries: Vec<_> = read_dir(INPUT_DIR).unwrap().flatten().collect();
    let total_files = entries.len();
    let total_f32 = total_files as f32;

    let loaded_images: Vec<RawImage> = entries
        .into_iter()
        .enumerate()
        .filter_map(|(i, entry)| {
            let path = entry.path();

            let original = open(&path).ok()?.to_luma8();

            distort(original.clone())
                .save(Path::new(OUTPUT_DIR).join(path.file_name()?))
                .ok()?;

            if i % 10 == 0 {
                println!("{:.1}%", (i as f32 / total_f32) * 100.0);
            }

            Some(RawImage { original })
        })
        .collect();

    println!("{}/{}", loaded_images.len(), total_files);

    loaded_images
}
