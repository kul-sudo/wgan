mod consts;
mod data;
mod dataset;
mod inference;
mod network;
mod training;

use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
use std::env::var;
// use wgan::{model::ModelConfig, training::TrainingConfig};
use data::files_init;
use inference::infer;
use training::train;

//
// pub fn infer_mode<B: burn::tensor::backend::Backend>(device: B::Device) {
//     wgan::infer::generate::<B>("artifact", device);
// }

fn main() {
    let mut files = files_init();

    let device = CudaDevice::default();
    let mode = var("MODE").unwrap_or_else(|_| panic!("No MODE specified."));

    match mode.to_lowercase().as_str() {
        "training" => {
            train::<Autodiff<Cuda>>(&mut files, device);
        }
        "inference" => {
            infer::<Cuda>(&device);
        }
        _ => {
            std::process::exit(1);
        }
    }
}
