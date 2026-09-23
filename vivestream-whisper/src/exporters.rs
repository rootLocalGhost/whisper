//! Exporters for synchronized lyrics (LRC, Enhanced Karaoke LRC), subtitles (SRT, WebVTT), and JSON.

use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Serialize, Clone)]
pub struct WordTiming {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub probability: f64,
}

#[derive(Debug, Serialize, Clone)]
pub struct Segment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<WordTiming>,
}

#[derive(Debug, Serialize, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: String,
    pub duration: f64,
    pub segments: Vec<Segment>,
}

pub fn format_lrc_timestamp(seconds: f64) -> String {
    let s = if seconds < 0.0 { 0.0 } else { seconds };
    let minutes = (s / 60.0) as u32;
    let rem = s % 60.0;
    let secs = rem as u32;
    let hundredths = ((rem - (secs as f64)) * 100.0) as u32;
    format!("{:02}:{:02}.{:02}", minutes, secs, hundredths)
}

pub fn format_srt_timestamp(seconds: f64) -> String {
    let s = if seconds < 0.0 { 0.0 } else { seconds };
    let hours = (s / 3600.0) as u32;
    let minutes = ((s % 3600.0) / 60.0) as u32;
    let secs = (s % 60.0) as u32;
    let millis = ((s - (s as u64 as f64)) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, secs, millis)
}

pub fn format_vtt_timestamp(seconds: f64) -> String {
    let s = if seconds < 0.0 { 0.0 } else { seconds };
    let hours = (s / 3600.0) as u32;
    let minutes = ((s % 3600.0) / 60.0) as u32;
    let secs = (s % 60.0) as u32;
    let millis = ((s - (s as u64 as f64)) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
}

pub fn to_lrc(result: &TranscriptionResult, title: &str, artist: &str) -> String {
    let mut out = String::new();
    if !title.is_empty() {
        out.push_str(&format!("[ti:{}]\n", title));
    }
    if !artist.is_empty() {
        out.push_str(&format!("[ar:{}]\n", artist));
    }
    out.push_str("[by:Vivestream-Revived-Native]\n");

    for seg in &result.segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let ts = format_lrc_timestamp(seg.start);
        out.push_str(&format!("[{}] {}\n", ts, text));
    }
    out
}

pub fn to_enhanced_lrc(result: &TranscriptionResult, title: &str, artist: &str) -> String {
    let mut out = String::new();
    if !title.is_empty() {
        out.push_str(&format!("[ti:{}]\n", title));
    }
    if !artist.is_empty() {
        out.push_str(&format!("[ar:{}]\n", artist));
    }
    out.push_str("[by:Vivestream-Revived-Karaoke]\n");

    for seg in &result.segments {
        let seg_ts = format_lrc_timestamp(seg.start);
        if !seg.words.is_empty() {
            let mut word_parts = Vec::new();
            for w in &seg.words {
                let w_text = w.word.trim();
                if w_text.is_empty() {
                    continue;
                }
                let w_ts = format_lrc_timestamp(w.start);
                word_parts.push(format!("<{}> {}", w_ts, w_text));
            }
            if !word_parts.is_empty() {
                out.push_str(&format!("[{}] {}\n", seg_ts, word_parts.join(" ")));
            } else if !seg.text.trim().is_empty() {
                out.push_str(&format!("[{}] {}\n", seg_ts, seg.text.trim()));
            }
        } else if !seg.text.trim().is_empty() {
            out.push_str(&format!("[{}] {}\n", seg_ts, seg.text.trim()));
        }
    }
    out
}

pub fn to_srt(result: &TranscriptionResult) -> String {
    let mut out = String::new();
    let mut idx = 1;
    for seg in &result.segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let start_str = format_srt_timestamp(seg.start);
        let end_str = format_srt_timestamp(seg.end);
        out.push_str(&format!("{}\n{} --> {}\n{}\n\n", idx, start_str, end_str, text));
        idx += 1;
    }
    out
}

pub fn to_vtt(result: &TranscriptionResult) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in &result.segments {
        let text = seg.text.trim();
        if text.is_empty() {
            continue;
        }
        let start_str = format_vtt_timestamp(seg.start);
        let end_str = format_vtt_timestamp(seg.end);
        out.push_str(&format!("{} --> {}\n{}\n\n", start_str, end_str, text));
    }
    out
}

pub fn save_files(
    result: &TranscriptionResult,
    prefix_path: &Path,
    formats: &[String],
    title: &str,
    artist: &str,
) -> std::io::Result<Vec<(String, String)>> {
    let mut written = Vec::new();

    if let Some(parent) = prefix_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    for fmt in formats {
        let f_lower = fmt.to_lowercase();
        match f_lower.as_str() {
            "lrc" => {
                let path = format!("{}.lrc", prefix_path.display());
                let content = to_lrc(result, title, artist);
                File::create(&path)?.write_all(content.as_bytes())?;
                written.push(("lrc".to_string(), path));
            }
            "elrc" => {
                let path = format!("{}.enhanced.lrc", prefix_path.display());
                let content = to_enhanced_lrc(result, title, artist);
                File::create(&path)?.write_all(content.as_bytes())?;
                written.push(("elrc".to_string(), path));
            }
            "srt" => {
                let path = format!("{}.srt", prefix_path.display());
                let content = to_srt(result);
                File::create(&path)?.write_all(content.as_bytes())?;
                written.push(("srt".to_string(), path));
            }
            "vtt" => {
                let path = format!("{}.vtt", prefix_path.display());
                let content = to_vtt(result);
                File::create(&path)?.write_all(content.as_bytes())?;
                written.push(("vtt".to_string(), path));
            }
            "json" => {
                let path = format!("{}.json", prefix_path.display());
                let content = serde_json::to_string_pretty(result)?;
                File::create(&path)?.write_all(content.as_bytes())?;
                written.push(("json".to_string(), path));
            }
            _ => {}
        }
    }
    Ok(written)
}
