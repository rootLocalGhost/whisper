//! Pure Rust multi-format audio decoder and 16,000 Hz resampler using Symphonia.

use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub const WHISPER_SAMPLE_RATE: u32 = 16000;

pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, String> {
    let src = File::open(&path).map_err(|e| format!("Failed to open audio file: {}", e))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| format!("Unsupported audio format or probe failed: {}", e))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "No supported audio tracks found in media".to_string())?;

    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| "Audio track has unknown sample rate".to_string())?;
    let channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to initialize audio decoder: {}", e))?;

    let mut raw_mono_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => {
                eprintln!("Warning: packet decode notice: {}", e);
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                append_mono_samples(&decoded, channels, &mut raw_mono_samples);
            }
            Err(Error::DecodeError(e)) => {
                eprintln!("Warning: decode frame error: {}", e);
                continue;
            }
            Err(e) => {
                eprintln!("Warning: decode error: {}", e);
                break;
            }
        }
    }

    if raw_mono_samples.is_empty() {
        return Err("Decoded audio produced zero samples".to_string());
    }

    // Resample to 16,000 Hz if necessary
    if sample_rate == WHISPER_SAMPLE_RATE {
        Ok(raw_mono_samples)
    } else {
        Ok(linear_resample(&raw_mono_samples, sample_rate, WHISPER_SAMPLE_RATE))
    }
}

fn append_mono_samples(buf: &AudioBufferRef, channels: usize, out: &mut Vec<f32>) {
    match buf {
        AudioBufferRef::F32(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += b.chan(c)[i];
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S16(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += b.chan(c)[i] as f32 / 32768.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::U8(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += (b.chan(c)[i] as f32 - 128.0) / 128.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S32(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += b.chan(c)[i] as f32 / 2147483648.0;
                }
                out.push(sum / channels as f32);
            }
        }
        AudioBufferRef::F64(b) => {
            let frames = b.frames();
            for i in 0..frames {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += b.chan(c)[i] as f32;
                }
                out.push(sum / channels as f32);
            }
        }
        _ => {
            eprintln!("Warning: unhandled audio buffer format, skipping chunk");
        }
    }
}

/// High-quality linear interpolating resampler
fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let target_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let src_idx = i as f64 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        let frac = (src_idx - idx0 as f64) as f32;

        let sample = input[idx0] * (1.0 - frac) + input[idx1] * frac;
        output.push(sample);
    }
    output
}
