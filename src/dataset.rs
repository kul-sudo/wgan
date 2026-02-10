use crate::consts::{CHANNELS, SIZE};
use crate::files::ImagePair;
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use rand::{RngExt, rng};

#[derive(Clone, Debug)]
pub struct ImageBatch<B: Backend> {
    pub edited: Tensor<B, 4>,
    pub original: Tensor<B, 4>,
}

#[derive(Clone, Default)]
pub struct ImageBatcher {}

impl<B: Backend> Batcher<B, ImagePair, ImageBatch<B>> for ImageBatcher {
    fn batch(&self, items: Vec<ImagePair>, device: &B::Device) -> ImageBatch<B> {
        let (w, h) = (SIZE.0 as usize, SIZE.1 as usize);

        let (edited, original): (Vec<_>, Vec<_>) = items
            .into_iter()
            .map(|item| {
                let mut e = Tensor::<B, 3>::from_data(
                    TensorData::new(item.edited, [h, w, CHANNELS]).convert::<B::FloatElem>(),
                    &device,
                )
                .permute([2, 0, 1]);

                let mut o = Tensor::<B, 3>::from_data(
                    TensorData::new(item.original, [h, w, CHANNELS]).convert::<B::FloatElem>(),
                    &device,
                )
                .permute([2, 0, 1]);

                if rng().random_bool(0.5) {
                    e = e.flip([2]);
                    o = o.flip([2]);
                }

                (e, o)
            })
            .unzip();

        ImageBatch {
            edited: Tensor::stack(edited, 0),
            original: Tensor::stack(original, 0),
        }
    }
}
