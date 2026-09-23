# Vivestream-Next: Native Whisper Integration Guide

This guide details how to integrate the standalone **`vivestream-whisper.exe`** binary into your **Tauri (Rust + TypeScript)** application.

---

## 1. On-Demand Architecture (0 MB Base App Overhead)

Your main Vivestream-Next installer stays lightweight. The Whisper engine and model weights are downloaded **only when the user enables the feature** in App Settings.

### File Locations in Vivestream Revived:
```
%LOCALAPPDATA%\VivestreamRevived\whisper\
├── vivestream-whisper.exe    # Standalone engine (~15-20 MB)
└── models\
    └── base.safetensors      # Whisper model weights (~140 MB)
```

---

## 2. Tauri Backend Commands (`src-tauri/src/whisper.rs`)

Create `src-tauri/src/whisper.rs` in your Tauri project:

```rust
use std::path::PathBuf;
use std::process::Command;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

#[derive(Debug, Serialize, Deserialize)]
pub struct WhisperStatus {
    pub binary_installed: bool,
    pub model_installed: bool,
    pub binary_path: String,
    pub models_dir: String,
    pub system_check: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LyricsOutput {
    pub success: bool,
    pub text: Option<String>,
    pub lrc: Option<String>,
    pub enhanced_lrc: Option<String>,
    pub srt: Option<String>,
    pub error: Option<String>,
}

fn get_whisper_dir(app: &AppHandle) -> PathBuf {
    let mut path = app.path().app_local_data_dir().unwrap_or_else(|_| PathBuf::from("."));
    path.push("whisper");
    std::fs::create_dir_all(&path).ok();
    path
}

fn get_binary_path(app: &AppHandle) -> PathBuf {
    let mut p = get_whisper_dir(app);
    p.push("vivestream-whisper.exe");
    p
}

fn get_models_dir(app: &AppHandle) -> PathBuf {
    let mut p = get_whisper_dir(app);
    p.push("models");
    std::fs::create_dir_all(&p).ok();
    p
}

/// 1. Check if Whisper engine & models are installed, and scan user hardware
#[tauri::command]
pub async fn check_whisper_status(app: AppHandle) -> Result<WhisperStatus, String> {
    let bin_path = get_binary_path(&app);
    let models_dir = get_models_dir(&app);
    let model_file = models_dir.join("base.safetensors");

    let binary_installed = bin_path.is_file();
    let model_installed = model_file.is_file();

    let mut system_check = None;
    if binary_installed {
        if let Ok(output) = Command::new(&bin_path).arg("--check").output() {
            if output.status.success() {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
                    system_check = Some(json);
                }
            }
        }
    }

    Ok(WhisperStatus {
        binary_installed,
        model_installed,
        binary_path: bin_path.to_string_lossy().to_string(),
        models_dir: models_dir.to_string_lossy().to_string(),
        system_check,
    })
}

/// 2. Download Whisper Binary & Model on demand
#[tauri::command]
pub async fn download_whisper_component(
    app: AppHandle,
    component: String, // "binary" or "model:base" or "model:small"
) -> Result<bool, String> {
    // In production, stream from your GitHub Releases:
    // e.g. https://github.com/YourOrg/vivestream-revived/releases/download/v1.0.0/vivestream-whisper-windows-x64.exe
    // or Hugging Face SafeTensors for model weights.
    //
    // Use reqwest or ureq to download directly to get_whisper_dir(&app) with progress events.
    Ok(true)
}

/// 3. Generate Synced Lyrics & Subtitles for a song/video
#[tauri::command]
pub async fn generate_synced_lyrics(
    app: AppHandle,
    audio_path: String,
    title: Option<String>,
    artist: Option<String>,
    model: Option<String>,
) -> Result<LyricsOutput, String> {
    let bin_path = get_binary_path(&app);
    if !bin_path.is_file() {
        return Err("Whisper engine binary is not installed. Please set it up in Settings.".into());
    }

    let models_dir = get_models_dir(&app);
    let model_name = model.unwrap_or_else(|| "base".to_string());

    let mut cmd = Command::new(&bin_path);
    cmd.arg(&audio_path)
        .arg("--model")
        .arg(&model_name)
        .arg("--models-dir")
        .arg(&models_dir)
        .arg("--json");

    if let Some(t) = title {
        cmd.arg("--title").arg(t);
    }
    if let Some(a) = artist {
        cmd.arg("--artist").arg(a);
    }

    let output = cmd.output().map_err(|e| format!("Failed to run whisper engine: {}", e))?;

    if !output.status.success() {
        let err_text = String::from_utf8_lossy(&output.stderr);
        return Ok(LyricsOutput {
            success: false,
            text: None,
            lrc: None,
            enhanced_lrc: None,
            srt: None,
            error: Some(err_text.to_string()),
        });
    }

    let parsed: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse engine JSON output: {}", e))?;

    Ok(LyricsOutput {
        success: parsed["success"].as_bool().unwrap_or(true),
        text: parsed["text"].as_str().map(String::from),
        lrc: parsed["lrc"].as_str().map(String::from),
        enhanced_lrc: parsed["enhanced_lrc"].as_str().map(String::from),
        srt: parsed["srt"].as_str().map(String::from),
        error: parsed["error"].as_str().map(String::from),
    })
}
```

In `src-tauri/src/main.rs`:
```rust
mod whisper;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            whisper::check_whisper_status,
            whisper::download_whisper_component,
            whisper::generate_synced_lyrics,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

---

## 3. TypeScript Frontend Hook (`useWhisperLyrics.ts`)

```typescript
import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';

export interface WhisperStatus {
  binary_installed: boolean;
  model_installed: boolean;
  system_check?: {
    system: {
      cpu_brand: string;
      cpu_cores: number;
      available_ram_gb: number;
      recommended_default_model: string;
      model_recommendations: Array<{
        model: string;
        can_run: bool;
        performance: string;
        note: string;
      }>;
    };
  };
}

export function useWhisperLyrics() {
  const [status, setStatus] = useState<WhisperStatus | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    refreshStatus();
  }, []);

  async function refreshStatus() {
    try {
      const res = await invoke<WhisperStatus>('check_whisper_status');
      setStatus(res);
    } catch (e) {
      console.error("Failed to check whisper status", e);
    }
  }

  async function generateLyrics(audioPath: string, title?: string, artist?: string) {
    setLoading(true);
    try {
      const res = await invoke<any>('generate_synced_lyrics', {
        audioPath,
        title,
        artist,
        model: status?.system_check?.system?.recommended_default_model || 'base'
      });
      return res;
    } finally {
      setLoading(false);
    }
  }

  return { status, loading, refreshStatus, generateLyrics };
}
```

---

## 4. Standalone CLI Usage (For Testing)

```powershell
# 1. Hardware scan & model compatibility advice
vivestream-whisper.exe --check

# 2. Transcribe song and generate .lrc, .enhanced.lrc, and .srt
vivestream-whisper.exe "C:\Music\song.mp3" --model base --format lrc,elrc,srt

# 3. Output pure JSON for process piping
vivestream-whisper.exe "C:\Music\song.mp3" --json
```
