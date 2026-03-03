use image::{DynamicImage, GrayImage, open};
use rand::{RngExt, rng};
use std::{
    fs::{create_dir_all, read, read_dir, write},
    path::Path,
};
use wincode::{SchemaRead, SchemaWrite, deserialize, serialize};

#[derive(Clone, Debug, SchemaWrite, SchemaRead)]
pub struct ImagePair {
    pub edited: Vec<f32>,
    pub original: Vec<f32>,
}

const IMAGES_CACHE: &str = "images.bin";
const INPUT_DIR: &str = "images";
const OUTPUT_DIR: &str = "edited";

pub const PIXEL_MAX: f32 = 255.0;
pub const PIXEL_MID: f32 = PIXEL_MAX / 2.0;

pub fn norm(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as f32 / PIXEL_MID) - 1.0).collect()
}

pub fn distort(mut luma: GrayImage) -> GrayImage {
    let mut r = rng();
    let a = r.random_range(1.8..2.3);
    let b = r.random_range(1.0..1.5);
    let c = r.random_range(0.1..1.4);
    let d = r.random_range(2.0..7.0);
    let f = r.random_range(4.0..6.0);
    let threshold = r.random_range(0.08..0.16);

    for p in luma.pixels_mut() {
        let l = p[0] as f32 / PIXEL_MAX;
        let effect = if l < threshold {
            a
        } else {
            (l * f).powf(b).max(1.0)
        };

        let val = l * effect;

        p[0] = if val >= 0.99 {
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

pub fn files_init() -> Vec<ImagePair> {
    if let Ok(data) = read(IMAGES_CACHE) {
        return deserialize(&data).unwrap();
    }

    create_dir_all(OUTPUT_DIR).unwrap();
    let entries: Vec<_> = read_dir(INPUT_DIR).unwrap().flatten().collect();
    let mut files = Vec::with_capacity(entries.len());

    for (n, entry) in entries.iter().enumerate() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let luma_original = open(&path).unwrap().to_luma8();
        let original_vec = norm(luma_original.as_raw());

        let luma_edited = distort(luma_original);

        luma_edited
            .save(Path::new(OUTPUT_DIR).join(path.file_name().unwrap()))
            .unwrap();

        files.push(ImagePair {
            edited: norm(luma_edited.as_raw()),
            original: original_vec,
        });

        println!("{:.2}%", (n + 1) as f32 / entries.len() as f32 * 100.0);
    }

    write(IMAGES_CACHE, serialize(&files).unwrap()).unwrap();
    files
}
