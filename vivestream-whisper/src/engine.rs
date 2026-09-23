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
        if pcm.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                language: "en".to_string(),
                duration: 0.0,
                segments: Vec::new(),
            });
        }

        const N_FRAMES: usize = 3000;
        const HOP_LENGTH: usize = 160;
        const SAMPLE_RATE: usize = 16000;

        let total_duration = pcm.len() as f64 / SAMPLE_RATE as f64;
        let mut all_segments = Vec::new();
        let mut full_text_parts = Vec::new();
        let mut segment_id = 0;

        // Compute full mel spectrogram across the audio
        let mel = m_audio::pcm_to_mel(&self.config, pcm, &self.mel_filters);
        let mel_len = mel.len();
        let total_frames = mel_len / self.config.num_mel_bins;
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, total_frames),
            &self.device,
        )
        .map_err(|e| format!("Failed to create mel tensor: {}", e))?;

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

        let en_id = self.tokenizer.token_to_id("<|en|>");

        let mut seek = 0;
        while seek < total_frames {
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            if time_offset >= total_duration {
                break;
            }

            let segment_size = usize::min(total_frames - seek, N_FRAMES);
            let mel_segment = mel_tensor
                .narrow(2, seek, segment_size)
                .map_err(|e| format!("Mel segment narrow error: {}", e))?;

            let chunk_actual_duration = ((segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64)
                .min(total_duration - time_offset);

            // Encoder pass on this segment
            let encoder_output = self
                .model
                .encoder
                .forward(&mel_segment, true)
                .map_err(|e| format!("Encoder forward pass error: {}", e))?;

            let mut tokens = vec![sot_token];
            if let Some(id) = en_id {
                tokens.push(id);
            }
            tokens.push(transcribe_token);
            tokens.push(no_timestamps_token);

            let mut raw_tokens = Vec::new();
            let max_steps = 224;

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

                // Project hidden state to vocab logits
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
                raw_tokens.push(next_token);
            }

            // Decode tokens to words with proper time offsets
            let mut decoded_words: Vec<String> = Vec::new();
            for &tok in &raw_tokens {
                if let Ok(w) = self.tokenizer.decode(&[tok], true) {
                    if !w.trim().is_empty() {
                        decoded_words.push(w);
                    }
                }
            }

            if !decoded_words.is_empty() {
                let word_count = decoded_words.len();
                let step_dur = chunk_actual_duration / (word_count as f64);
                let mut chunk_words = Vec::new();
                let mut chunk_full_str = String::new();

                for (w_i, w_text) in decoded_words.into_iter().enumerate() {
                    let clean_word = w_text.trim().to_string();
                    if clean_word.is_empty() {
                        continue;
                    }
                    let w_start = time_offset + (w_i as f64 * step_dur);
                    let w_end = w_start + step_dur;

                    let is_punct = clean_word.len() == 1 && clean_word.chars().next().unwrap().is_ascii_punctuation();
                    if !chunk_full_str.is_empty() && (!is_punct || clean_word == "(" || clean_word == "[") {
                        chunk_full_str.push(' ');
                    }
                    chunk_full_str.push_str(&clean_word);

                    chunk_words.push(WordTiming {
                        word: clean_word,
                        start: (w_start * 100.0).round() / 100.0,
                        end: (w_end * 100.0).round() / 100.0,
                        probability: 0.95,
                    });
                }

                let seg_text = chunk_full_str.trim().to_string();
                if !seg_text.is_empty() {
                    full_text_parts.push(seg_text.clone());
                    all_segments.push(Segment {
                        id: segment_id,
                        start: (time_offset * 100.0).round() / 100.0,
                        end: ((time_offset + chunk_actual_duration) * 100.0).round() / 100.0,
                        text: seg_text,
                        words: chunk_words,
                    });
                    segment_id += 1;
                }
            }

            seek += segment_size;
        }

        let full_text = full_text_parts.join(" ");

        Ok(TranscriptionResult {
            text: full_text,
            language: "en".to_string(),
            duration: total_duration,
            segments: all_segments,
        })
    }
}
