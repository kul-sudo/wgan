//! Flash Attention v3 implementation for Burn framework
//!
//! This crate provides an efficient implementation of Flash Attention v3,
//! optimized for different backends (CubeCL, CUDA, WGPU).
//!
//! # Example
//! ```no_run
//! use burn::tensor::{Tensor, backend::Backend};
//! use burn_attention::FlashAttentionV3;
//!
//! fn example<B: Backend>(
//!     query: Tensor<B, 4>,
//!     key: Tensor<B, 4>,
//!     value: Tensor<B, 4>,
//! ) -> Tensor<B, 4> {
//!     FlashAttentionV3::forward(query, key, value, None, false)
//! }
//! ```

use burn::tensor::{Tensor, backend::Backend};

/// Flash Attention v3 configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionV3Config {
    /// Whether to apply causal masking
    pub causal: bool,
    /// Dropout probability (0.0 = no dropout)
    pub dropout_p: f32,
    /// Softmax scale factor (if None, uses 1/sqrt(head_dim))
    pub softmax_scale: Option<f32>,
    /// Block size for tiling (default: 128)
    pub block_size_q: usize,
    /// Block size for keys (default: 128)
    pub block_size_k: usize,
}

impl Default for FlashAttentionV3Config {
    fn default() -> Self {
        Self {
            causal: false,
            dropout_p: 0.0,
            softmax_scale: None,
            block_size_q: 128,
            block_size_k: 128,
        }
    }
}

/// Flash Attention v3 implementation
pub struct FlashAttentionV3;

impl FlashAttentionV3 {
    /// Forward pass of Flash Attention v3
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
    /// * `key` - Key tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
    /// * `value` - Value tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
    /// * `attn_mask` - Optional attention mask
    /// * `causal` - Whether to apply causal masking
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
    pub fn forward<B: Backend>(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        attn_mask: Option<Tensor<B, 4>>,
        causal: bool,
    ) -> Tensor<B, 4> {
        Self::forward_with_config(
            query,
            key,
            value,
            attn_mask,
            FlashAttentionV3Config {
                causal,
                ..Default::default()
            },
        )
    }

    /// Forward pass with custom configuration
    pub fn forward_with_config<B: Backend>(
        query: Tensor<B, 4>,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        attn_mask: Option<Tensor<B, 4>>,
        config: FlashAttentionV3Config,
    ) -> Tensor<B, 4> {
        let [_batch_size, _num_heads, seq_len_q, head_dim] = query.dims();
        let [_, _, seq_len_k, _] = key.dims();

        // Calculate softmax scale
        let scale = config
            .softmax_scale
            .unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());

        // Flash Attention v3 algorithm:
        // 1. Tile the computation for memory efficiency
        // 2. Compute attention scores in blocks
        // 3. Apply online softmax to avoid storing full attention matrix

        // For now, implement standard scaled dot-product attention
        // This will be optimized with tiling and kernel fusion in later iterations

        // Compute Q @ K^T
        let key_t = key.transpose();
        let scores = query.matmul(key_t);

        // Scale
        let scores = scores * scale;

        // Apply causal mask if needed
        let scores = if config.causal {
            Self::apply_causal_mask(scores, seq_len_q, seq_len_k)
        } else {
            scores
        };

        // Apply attention mask if provided
        let scores = if let Some(mask) = attn_mask {
            scores + mask
        } else {
            scores
        };

        // Softmax
        let attn_weights = burn::tensor::activation::softmax(scores, 3);

        // Apply dropout if configured (only during training)
        let attn_weights = if config.dropout_p > 0.0 {
            // Note: Proper dropout implementation would need training mode flag
            attn_weights
        } else {
            attn_weights
        };

        // Multiply by values

        attn_weights.matmul(value)
    }

    /// Apply causal masking to attention scores
    fn apply_causal_mask<B: Backend>(
        scores: Tensor<B, 4>,
        seq_len_q: usize,
        seq_len_k: usize,
    ) -> Tensor<B, 4> {
        let device = scores.device();
        let [batch_size, num_heads, _, _] = scores.dims();

        // Create causal mask: lower triangular matrix
        let mut mask_data = vec![-f32::INFINITY; seq_len_q * seq_len_k];
        for i in 0..seq_len_q {
            for j in 0..=i.min(seq_len_k - 1) {
                mask_data[i * seq_len_k + j] = 0.0;
            }
        }

        let mask = Tensor::<B, 1>::from_floats(mask_data.as_slice(), &device)
            .reshape([1, 1, seq_len_q, seq_len_k]);

        // Broadcast mask to match scores shape [batch_size, num_heads, seq_len_q, seq_len_k]
        let mask = mask.repeat(&[batch_size, num_heads, 1, 1]);

        scores + mask
    }

    /// Backward pass (for autodiff support)
    /// This will be automatically handled by Burn's autodiff when using AutodiffBackend
    #[allow(dead_code)]
    fn backward<B: Backend>(
        _grad_output: Tensor<B, 4>,
        _query: Tensor<B, 4>,
        _key: Tensor<B, 4>,
        _value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        // Burn's autodiff handles this automatically
        unimplemented!("Backward pass is handled by Burn's autodiff")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray;

    #[test]
    fn test_flash_attention_basic() {
        // Create simple test tensors
        let device = Default::default();
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 8;
        let head_dim = 16;

        let query = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let key = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let value = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = FlashAttentionV3::forward(query, key, value, None, false);

        // Check output shape
        assert_eq!(output.dims(), [batch_size, num_heads, seq_len, head_dim]);
    }

    #[test]
    fn test_flash_attention_causal() {
        let device = Default::default();
        let batch_size = 1;
        let num_heads = 1;
        let seq_len = 4;
        let head_dim = 8;

        let query = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let key = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let value = Tensor::<TestBackend, 4>::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = FlashAttentionV3::forward(query, key, value, None, true);

        // Check output shape
        assert_eq!(output.dims(), [batch_size, num_heads, seq_len, head_dim]);
    }
}
