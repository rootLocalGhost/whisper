"""Whisper Lyrics & Subtitle Studio (Intel XPU & CPU)
A feature-rich Gradio web application for generating synchronized lyrics (.lrc, .enhanced.lrc),
subtitles (.srt, .vtt), and structured JSON with Intel Arc A770 acceleration and CPU fallback.
All models are downloaded and stored locally in the ./models directory.
"""

import os
import sys
import time
import json
import logging
import tempfile
from typing import Optional, List, Dict, Any, Tuple

import gradio as gr
import torch
import whisper

from whisper_lyrics.engine import LyricsEngine, get_available_devices, resolve_device
from whisper_lyrics.exporters import to_lrc, to_enhanced_lrc, to_srt, to_vtt, to_json

# Setup workspace models directory
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Set environment variable so all whisper routines use ./models
os.environ["WHISPER_MODELS_DIR"] = MODELS_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("whisper_app")

AVAILABLE_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v3",
    "large-v3-turbo",
    "turbo",
]

# Supported language codes map
LANGUAGE_OPTIONS = ["Auto-Detect"] + [
    f"{code} - {name.title()}" for code, name in sorted(whisper.tokenizer.LANGUAGES.items())
]

# Global cache for preloaded engine
CACHED_ENGINE: Optional[LyricsEngine] = None
CURRENT_MODEL_NAME = "small"
CURRENT_DEVICE = "auto"


def get_local_models_status() -> List[List[str]]:
    """Scan the local ./models directory and report downloaded weights."""
    rows = []
    for model_name in AVAILABLE_MODELS:
        # Check standard filename conventions
        target_name = f"{model_name}.pt"
        target_path = os.path.join(MODELS_DIR, target_name)
        if os.path.isfile(target_path):
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            status = f"Ready ({size_mb:.1f} MB)"
        else:
            status = "Not Downloaded"
        rows.append([model_name, status, MODELS_DIR])
    return rows


def get_system_banner() -> str:
    """Generate Markdown banner showing active hardware devices."""
    devices = get_available_devices()
    xpu_info = devices["xpu"]
    cuda_info = devices["cuda"]

    if xpu_info["available"]:
        gpu_badge = f"**Hardware Acceleration:** ⚡ **Intel XPU Enabled** ({xpu_info.get('device_name', 'Intel Arc GPU')})"
    elif cuda_info["available"]:
        gpu_badge = f"**Hardware Acceleration:** ⚡ **NVIDIA CUDA Enabled** ({cuda_info.get('device_name', 'CUDA GPU')})"
    else:
        gpu_badge = "**Hardware Acceleration:** 💻 **CPU Mode (No GPU detected)**"

    return f"""
<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 16px 20px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 12px; color: #f8fafc;">
    <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 6px;">🎙️ Whisper Synced Lyrics & Subtitle Studio</div>
    <div style="font-size: 0.9rem; color: #94a3b8;">
        {gpu_badge} &nbsp;|&nbsp; 📁 Local Models Path: <code style="color: #38bdf8;">{MODELS_DIR}</code> &nbsp;|&nbsp; PyTorch: <code style="color: #a78bfa;">{torch.__version__}</code>
    </div>
</div>
"""


def ensure_engine(model_name: str, device: str) -> LyricsEngine:
    """Get or instantiate cached Whisper engine."""
    global CACHED_ENGINE, CURRENT_MODEL_NAME, CURRENT_DEVICE
    resolved = resolve_device(device)
    if CACHED_ENGINE is None or CURRENT_MODEL_NAME != model_name or CURRENT_DEVICE != resolved:
        logger.info("Initializing engine for model '%s' on device '%s'...", model_name, resolved)
        CACHED_ENGINE = LyricsEngine(
            model_name=model_name,
            device=resolved,
            download_root=MODELS_DIR,
        )
        CURRENT_MODEL_NAME = model_name
        CURRENT_DEVICE = resolved
    return CACHED_ENGINE


def download_model_action(model_name: str) -> Tuple[str, List[List[str]]]:
    """Explicitly trigger model download into ./models/."""
    try:
        url = whisper._MODELS.get(model_name)
        if not url:
            return f"Error: Unknown model '{model_name}'", get_local_models_status()
        
        target_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
        if os.path.exists(target_path):
            return f"Model '{model_name}' is already downloaded in {MODELS_DIR}!", get_local_models_status()

        logger.info("Downloading model %s to %s...", model_name, MODELS_DIR)
        whisper._download(url, root=MODELS_DIR, in_memory=False)
        return f"Successfully downloaded '{model_name}' to {MODELS_DIR}!", get_local_models_status()
    except Exception as e:
        logger.exception("Failed to download model")
        return f"Download failed: {str(e)}", get_local_models_status()


def process_audio(
    audio_upload: Optional[str],
    audio_filepath: Optional[str],
    model_name: str,
    device: str,
    task: str,
    language_choice: str,
    word_timestamps: bool,
    title: str,
    artist: str,
    temperature: float,
    progress=gr.Progress(),
) -> Tuple[
    str,  # Standard LRC
    str,  # Enhanced LRC
    str,  # SRT
    str,  # WebVTT
    str,  # JSON
    List[List[Any]],  # Segments table
    Optional[str],  # Download LRC path
    Optional[str],  # Download ELRC path
    Optional[str],  # Download SRT path
    Optional[str],  # Download JSON path
    str,  # Status message
]:
    # Determine audio source
    input_file = audio_upload or audio_filepath
    if not input_file or not os.path.exists(input_file):
        empty_table = []
        return (
            "", "", "", "", "", empty_table,
            None, None, None, None,
            "⚠️ Please upload an audio file or provide a valid file path.",
        )

    # Parse language
    lang_code = None
    if language_choice and language_choice != "Auto-Detect":
        lang_code = language_choice.split(" - ")[0].strip()

    task_name = "translate" if "Translate" in task else "transcribe"

    progress(0.1, desc="Preparing Whisper model in local directory...")
    t0 = time.time()

    try:
        engine = ensure_engine(model_name=model_name, device=device)

        progress(0.3, desc=f"Transcribing on {engine.target_device.upper()}...")
        result = engine.transcribe(
            audio_path=input_file,
            word_timestamps=word_timestamps,
            language=lang_code,
            task=task_name,
            temperature=temperature,
        )

        progress(0.85, desc="Formatting lyrics and subtitles...")
        
        # Build text outputs
        lrc_text = to_lrc(result, title=title, artist=artist)
        elrc_text = to_enhanced_lrc(result, title=title, artist=artist)
        srt_text = to_srt(result)
        vtt_text = to_vtt(result)
        json_obj = to_json(result)
        json_text = json.dumps(json_obj, ensure_ascii=False, indent=2)

        # Build segments table
        segments_table = []
        for s in json_obj.get("segments", []):
            duration = round(s["end"] - s["start"], 2)
            words_count = len(s.get("words", []))
            segments_table.append([
                s["id"],
                f"{s['start']:.2f}s",
                f"{s['end']:.2f}s",
                f"{duration:.2f}s",
                words_count,
                s["text"],
            ])

        # Save files to disk for download
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp_prefix = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(OUTPUTS_DIR, f"{base_name}_{timestamp_prefix}")

        saved = engine.export_files(
            result=result,
            output_prefix=prefix,
            formats=["lrc", "elrc", "srt", "vtt", "json"],
            title=title,
            artist=artist,
        )

        elapsed = time.time() - t0
        device_used = engine.target_device.upper()
        detected_lang = result.get("language", "unknown")

        status_msg = (
            f"✅ **Complete in {elapsed:.2f}s** | Device: **{device_used}** | "
            f"Detected Language: **{detected_lang}** | Segments: **{len(segments_table)}**"
        )

        progress(1.0, desc="Done!")
        return (
            lrc_text,
            elrc_text,
            srt_text,
            vtt_text,
            json_text,
            segments_table,
            saved.get("lrc"),
            saved.get("elrc"),
            saved.get("srt"),
            saved.get("json"),
            status_msg,
        )

    except Exception as e:
        logger.exception("Error processing audio")
        empty_table = []
        return (
            "", "", "", "", "", empty_table,
            None, None, None, None,
            f"❌ Error during transcription: {str(e)}",
        )


def process_batch(
    folder_path: str,
    model_name: str,
    device: str,
    word_timestamps: bool,
    formats: List[str],
    progress=gr.Progress(),
) -> str:
    """Batch transcribe all audio/video files in a directory."""
    if not folder_path or not os.path.isdir(folder_path):
        return f"Error: '{folder_path}' is not a valid directory."

    valid_exts = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mkv", ".webm"}
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    if not audio_files:
        return f"No supported audio files found in '{folder_path}'."

    engine = ensure_engine(model_name=model_name, device=device)
    total = len(audio_files)
    logs = [f"Starting batch of {total} files using {engine.target_device.upper()} [model={model_name}]...\n"]

    for idx, fpath in enumerate(audio_files, 1):
        fname = os.path.basename(fpath)
        progress((idx - 1) / total, desc=f"Processing {idx}/{total}: {fname}")
        t0 = time.time()
        try:
            res = engine.transcribe(fpath, word_timestamps=word_timestamps)
            out_prefix = os.path.splitext(fpath)[0]
            engine.export_files(res, out_prefix, formats=formats)
            dur = time.time() - t0
            logs.append(f"[{idx}/{total}] ✅ {fname} ({dur:.1f}s) -> Exported: {', '.join(formats)}")
        except Exception as e:
            logs.append(f"[{idx}/{total}] ❌ {fname} -> Error: {str(e)}")

    progress(1.0, desc="Batch complete!")
    logs.append(f"\nAll {total} files completed!")
    return "\n".join(logs)


# ==========================================
# Gradio UI Layout
# ==========================================
custom_css = """
.container { max-width: 1200px; margin: auto; }
.output-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 0.9rem; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
"""

with gr.Blocks(title="Whisper Lyrics & Subtitle Studio (Intel XPU)") as demo:
    gr.HTML(get_system_banner())

    with gr.Tabs():
        # --- TAB 1: Single Audio Transcription ---
        with gr.TabItem("🎵 Generate Synced Lyrics & Subtitles"):
            with gr.Row():
                # Left Column: Inputs & Settings
                with gr.Column(scale=5):
                    with gr.Group():
                        gr.Markdown("### 1. Select Audio or Video")
                        audio_upload = gr.Audio(
                            label="Upload Audio / Recording",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        audio_filepath = gr.Textbox(
                            label="Or Enter Local File Path",
                            placeholder=r"C:\Music\my_song.mp3",
                            info="Direct path to any audio/video on your PC (ideal for Vivestream Revived downloads)",
                        )

                    with gr.Group():
                        gr.Markdown("### 2. Whisper Model & Compute Settings")
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                label="Whisper Model",
                                choices=AVAILABLE_MODELS,
                                value="small",
                                info="Downloads automatically into ./models/ on first use",
                            )
                            device_dropdown = gr.Dropdown(
                                label="Compute Device",
                                choices=["auto", "xpu", "cpu"],
                                value="auto",
                                info="XPU = Intel Arc A770 (FP16), CPU = Host CPU (FP32)",
                            )

                        with gr.Row():
                            task_radio = gr.Radio(
                                label="Task",
                                choices=["Transcribe (Original)", "Translate to English"],
                                value="Transcribe (Original)",
                            )
                            language_dropdown = gr.Dropdown(
                                label="Spoken Language",
                                choices=LANGUAGE_OPTIONS,
                                value="Auto-Detect",
                            )

                        word_timestamps_cb = gr.Checkbox(
                            label="Enable Word-Level Timestamps (Required for Enhanced Karaoke LRC)",
                            value=True,
                        )

                        with gr.Accordion("Advanced & Metadata Settings", open=False):
                            with gr.Row():
                                song_title = gr.Textbox(label="Song Title (for LRC [ti:..])", placeholder="Song Name")
                                song_artist = gr.Textbox(label="Artist Name (for LRC [ar:..])", placeholder="Artist")
                            temperature_slider = gr.Slider(
                                label="Sampling Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.1,
                                info="0.0 is deterministic and most accurate",
                            )

                    generate_btn = gr.Button("🚀 Generate Lyrics & Subtitles", variant="primary", size="lg")

                # Right Column: Outputs
                with gr.Column(scale=7):
                    status_markdown = gr.Markdown("Ready. Upload an audio file and click **Generate**.")

                    with gr.Tabs():
                        with gr.TabItem("📜 Standard LRC (Line-by-Line)"):
                            gr.Markdown("*Standard synchronized lyrics format `[mm:ss.xx] Line` for music players.*")
                            out_lrc = gr.Textbox(label="Standard LRC Lyrics", lines=14, elem_classes="output-box")
                            file_lrc = gr.File(label="Download .lrc File")

                        with gr.TabItem("🎤 Enhanced Karaoke LRC (Word-by-Word)"):
                            gr.Markdown("*Word-level timestamped karaoke format `<mm:ss.xx> Word`.*")
                            out_elrc = gr.Textbox(label="Enhanced Karaoke LRC", lines=14, elem_classes="output-box")
                            file_elrc = gr.File(label="Download .enhanced.lrc File")

                        with gr.TabItem("🎬 Subtitles (SRT & VTT)"):
                            with gr.Row():
                                with gr.Column():
                                    out_srt = gr.Textbox(label="SRT Format", lines=12, elem_classes="output-box")
                                    file_srt = gr.File(label="Download .srt File")
                                with gr.Column():
                                    out_vtt = gr.Textbox(label="WebVTT Format", lines=12, elem_classes="output-box")

                        with gr.TabItem("📊 Segment Inspector"):
                            gr.Markdown("*Interactive table of transcribed speech segments and word counts.*")
                            out_table = gr.Dataframe(
                                headers=["ID", "Start", "End", "Duration", "Words", "Text"],
                                datatype=["number", "str", "str", "str", "number", "str"],
                                interactive=False,
                            )

                        with gr.TabItem("💻 JSON (Tauri / Rust Data)"):
                            gr.Markdown("*Clean structured JSON payload with word bounds and probabilities.*")
                            out_json = gr.Textbox(label="Structured JSON", lines=14, elem_classes="output-box")
                            file_json = gr.File(label="Download .json File")

        # --- TAB 2: Batch Folder Processing ---
        with gr.TabItem("📂 Batch Folder Processing"):
            gr.Markdown("### Transcribe Entire Music Albums or Playlists")
            gr.Markdown("Process all audio files in a local folder and save synchronized `.lrc` and `.srt` files directly into that folder.")
            with gr.Row():
                batch_folder_input = gr.Textbox(
                    label="Local Folder Path",
                    placeholder=r"C:\Music\DownloadedPlaylist",
                    info="Path to directory containing audio files",
                )
                batch_formats = gr.CheckboxGroup(
                    label="Output Formats to Save",
                    choices=["lrc", "elrc", "srt", "vtt", "json"],
                    value=["lrc", "elrc", "srt"],
                )
            batch_btn = gr.Button("⚡ Start Batch Processing", variant="primary")
            batch_logs = gr.Textbox(label="Batch Processing Logs", lines=12, elem_classes="output-box")

        # --- TAB 3: Local Models Manager ---
        with gr.TabItem("💾 Local Models Manager"):
            gr.Markdown("### Whisper Model Weights Directory (`./models/`)")
            gr.Markdown(
                f"All Whisper models are stored inside `<workspace>/models/` (**`{MODELS_DIR}`**), "
                "keeping your primary Windows user directory clean."
            )
            refresh_models_btn = gr.Button("🔄 Refresh Model Status")
            models_table = gr.Dataframe(
                headers=["Model Name", "Local Status", "Storage Location"],
                value=get_local_models_status(),
                interactive=False,
            )
            with gr.Row():
                download_target_model = gr.Dropdown(
                    label="Pre-download Model into ./models/",
                    choices=AVAILABLE_MODELS,
                    value="base",
                )
                download_btn = gr.Button("⬇️ Download Selected Model Now")
            download_status = gr.Textbox(label="Download Status", interactive=False)

    # Wire up Single Transcription event
    generate_btn.click(
        fn=process_audio,
        inputs=[
            audio_upload,
            audio_filepath,
            model_dropdown,
            device_dropdown,
            task_radio,
            language_dropdown,
            word_timestamps_cb,
            song_title,
            song_artist,
            temperature_slider,
        ],
        outputs=[
            out_lrc,
            out_elrc,
            out_srt,
            out_vtt,
            out_json,
            out_table,
            file_lrc,
            file_elrc,
            file_srt,
            file_json,
            status_markdown,
        ],
    )

    # Wire up Batch Processing event
    batch_btn.click(
        fn=process_batch,
        inputs=[
            batch_folder_input,
            model_dropdown,
            device_dropdown,
            word_timestamps_cb,
            batch_formats,
        ],
        outputs=[batch_logs],
    )

    # Wire up Local Models Manager events
    refresh_models_btn.click(fn=get_local_models_status, outputs=[models_table])
    download_btn.click(
        fn=download_model_action,
        inputs=[download_target_model],
        outputs=[download_status, models_table],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"Starting Whisper Lyrics & Subtitle Studio (Intel XPU)")
    print(f"Local Models Directory: {MODELS_DIR}")
    print(f"Server URL:             http://127.0.0.1:{args.port}")
    print(f"=======================================================\n")

    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share, theme=gr.themes.Soft(), css=custom_css)
