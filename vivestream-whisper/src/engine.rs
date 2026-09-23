//! Whisper transformer inference engine implemented with Candle.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m_whisper, audio as m_audio, model::Whisper, Config};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::exporters::{Segment, TranscriptionResult, WordTiming};

pub fn get_config_for_model(name: &str) -> Config {
    match name {
        "tiny" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            suppress_tokens: vec![],
        },
        "base" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            suppress_tokens: vec![],
        },
        "small" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 768,
            encoder_attention_heads: 12,
            encoder_layers: 12,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 12,
            decoder_layers: 12,
            suppress_tokens: vec![],
        },
        "medium" => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 1024,
            encoder_attention_heads: 16,
            encoder_layers: 24,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 16,
            decoder_layers: 24,
            suppress_tokens: vec![],
        },
        _ => Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            vocab_size: 51865,
            max_target_positions: 448,
            decoder_attention_heads: 8,
            decoder_layers: 6,
            suppress_tokens: vec![],
        },
    }
}

pub fn load_embedded_mel_filters() -> Vec<f32> {
    let bytes = include_bytes!("mel_80.bin");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

pub struct WhisperEngine {
    pub model: Whisper,
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub mel_filters: Vec<f32>,
}

impl WhisperEngine {
    pub fn new<P: AsRef<Path>>(
        weights_path: P,
        config: Config,
        tokenizer: Tokenizer,
        device: Device,
    ) -> Result<Self, String> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], DType::F32, &device)
                .map_err(|e| format!("Failed to read safetensors: {}", e))?
        };

        let model = Whisper::load(&vb, config.clone())
            .map_err(|e| format!("Failed to load Whisper model weights: {}", e))?;

        let mel_filters = load_embedded_mel_filters();

        Ok(Self {
            model,
            config,
            tokenizer,
            device,
            mel_filters,
        })
    }

    pub fn transcribe_pcm(&mut self, pcm: &[f32]) -> Result<TranscriptionResult, String> {
        let mel = m_audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let mel_len = mel.len();
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, mel_len / self.config.num_mel_bins),
            &self.device,
        )
        .map_err(|e| format!("Failed to create mel tensor: {}", e))?;

        // Encoder pass
        let encoder_output = self
            .model
            .encoder
            .forward(&mel_tensor, true)
            .map_err(|e| format!("Encoder forward pass error: {}", e))?;

        // Token IDs from tokenizer
        let sot_token = self
            .tokenizer
            .token_to_id(m_whisper::SOT_TOKEN)
            .unwrap_or(50258);
        let transcribe_token = self
            .tokenizer
            .token_to_id(m_whisper::TRANSCRIBE_TOKEN)
            .unwrap_or(50359);
        let eot_token = self
            .tokenizer
            .token_to_id(m_whisper::EOT_TOKEN)
            .unwrap_or(50257);
        let no_timestamps_token = self
            .tokenizer
            .token_to_id(m_whisper::NO_TIMESTAMPS_TOKEN)
            .unwrap_or(50363);

        // Precompute suppress tokens tensor
        let suppress_tokens: Vec<f32> = (0..self.config.vocab_size as u32)
            .map(|i| {
                if self.config.suppress_tokens.contains(&i) {
                    f32::NEG_INFINITY
                } else {
                    0.0f32
                }
            })
            .collect();
        let suppress_t = Tensor::new(suppress_tokens.as_slice(), &self.device)
            .map_err(|e| format!("Suppress tensor error: {}", e))?;

        let mut tokens = vec![sot_token];
        if let Some(en_id) = self.tokenizer.token_to_id("<|en|>") {
            tokens.push(en_id);
        }
        tokens.push(transcribe_token);
        tokens.push(no_timestamps_token);

        let mut segments = Vec::new();
        let mut words = Vec::new();
        let mut current_segment_text = String::new();

        // Autoregressive decoding loop (up to max_target_positions)
        let max_steps = 224;
        let mut prev_word_time = 0.0f64;

        for _step in 0..max_steps {
            let tokens_t = Tensor::new(&tokens[..], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| format!("Tensor creation error: {}", e))?;

            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &encoder_output, true)
                .map_err(|e| format!("Decoder forward error: {}", e))?;

            let (_, seq_len, _) = ys
                .dims3()
                .map_err(|e| format!("Decoder dims3 error: {}", e))?;

            // Project hidden state from final layer to vocab logits
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..)).map_err(|e| format!("{}", e))?)
                .map_err(|e| format!("Decoder final_linear error: {}", e))?
                .i(0)
                .map_err(|e| format!("{}", e))?
                .i(0)
                .map_err(|e| format!("{}", e))?;

            let logits = logits
                .broadcast_add(&suppress_t)
                .map_err(|e| format!("Suppress add error: {}", e))?;

            // Greedy argmax
            let next_token = logits
                .argmax(0)
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| format!("Argmax error: {}", e))?;

            if next_token == eot_token {
                break;
            }

            tokens.push(next_token);

            // Decode token to text
            if let Ok(token_str) = self.tokenizer.decode(&[next_token], true) {
                if !token_str.is_empty() {
                    let word_start = prev_word_time;
                    let word_end = word_start + 0.35; // Estimated word duration
                    prev_word_time = word_end;

                    words.push(WordTiming {
                        word: token_str.clone(),
                        start: word_start,
                        end: word_end,
                        probability: 0.95,
                    });

                    current_segment_text.push_str(&token_str);
                }
            }
        }

        let total_duration = pcm.len() as f64 / 16000.0;
        let full_text = current_segment_text.trim().to_string();

        segments.push(Segment {
            id: 0,
            start: 0.0,
            end: total_duration,
            text: full_text.clone(),
            words,
        });

        Ok(TranscriptionResult {
            text: full_text,
            language: "en".to_string(),
            duration: total_duration,
            segments,
        })
    }
}
