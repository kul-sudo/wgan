use image::{DynamicImage, open};
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

const PIXEL_MAX: f32 = 255.0;
const PIXEL_MID: f32 = PIXEL_MAX / 2.0;
const TARGET_WHITE: f32 = 253.0;
const LUMA_THRESHOLD: f32 = 0.12;

fn norm(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&b| (b as f32 / PIXEL_MID) - 1.0).collect()
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

        let mut luma = open(&path).unwrap().to_luma8();
        let original = norm(luma.as_raw());

        for p in luma.pixels_mut() {
            let l = p[0] as f32 / PIXEL_MAX;
            let effect = if l < LUMA_THRESHOLD {
                0.6
            } else {
                (l * 2.5).powf(1.1).max(1.0)
            };
            p[0] = (l * effect * TARGET_WHITE).min(TARGET_WHITE) as u8;
        }

        let final_img = DynamicImage::ImageLuma8(luma)
            .blur(0.5)
            .adjust_contrast(50.0);
        final_img
            .save(Path::new(OUTPUT_DIR).join(path.file_name().unwrap()))
            .unwrap();

        files.push(ImagePair {
            edited: norm(final_img.to_luma8().as_raw()),
            original,
        });

        println!("{:.2}%", (n + 1) as f32 / entries.len() as f32 * 100.0);
    }

    write(IMAGES_CACHE, serialize(&files).unwrap()).unwrap();
    files
}
