"""Command-line interface for whisper_lyrics.
Designed for standalone usage and Tauri/Rust child-process integration.
"""

import os
import sys
import json
import logging
import argparse
from typing import List

from .engine import LyricsEngine, get_available_devices, resolve_device
from .exporters import to_lrc, to_enhanced_lrc, to_srt, to_vtt, to_json


def main():
    parser = argparse.ArgumentParser(
        description="Whisper Synced Lyrics & Subtitles Generator (Intel XPU + CPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", type=str, help="Path to input audio or video file")
    parser.add_argument(
        "--model",
        "-m",
        default="small",
        help="Whisper model name: tiny, base, small, medium, large-v3, turbo, etc.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="auto",
        choices=["auto", "xpu", "cuda", "cpu"],
        help="Compute device. 'auto' selects XPU if available, else CPU.",
    )
    parser.add_argument(
        "--format",
        "-f",
        nargs="+",
        default=["lrc", "elrc", "srt"],
        help="List of formats to export (comma-separated or space-separated): lrc, elrc, srt, vtt, json, or all",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory (defaults to same folder as audio file)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        help="Spoken language code (e.g. en, ja, es). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="'transcribe' for original lyrics, 'translate' to translate to English.",
    )
    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="Disable word-level timestamps (faster, but disables word-by-word karaoke)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional song title for LRC metadata tags",
    )
    parser.add_argument(
        "--artist",
        default="",
        help="Optional artist name for LRC metadata tags",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print structured transcription JSON to stdout (ideal for Tauri IPC child process)",
    )
    parser.add_argument(
        "--check-devices",
        action="store_true",
        help="Print detected hardware devices (XPU, CUDA, CPU) and exit",
    )

    args = parser.parse_args()

    if args.check_devices:
        devs = get_available_devices()
        print(json.dumps(devs, indent=2))
        return

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        sys.stderr.write(f"Error: Audio file not found: {audio_path}\n")
        sys.exit(1)

    # Parse formats (support both space-separated and comma-separated tokens)
    raw_formats = args.format if isinstance(args.format, list) else [args.format]
    parsed_formats = []
    for item in raw_formats:
        for f in item.split(","):
            f_clean = f.strip().lower()
            if f_clean:
                parsed_formats.append(f_clean)

    if "all" in parsed_formats:
        formats = ["lrc", "elrc", "srt", "vtt", "json"]
    else:
        formats = parsed_formats

    if args.json_stdout:
        logging.getLogger().handlers = []
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    engine = LyricsEngine(model_name=args.model, device=args.device)

    result = engine.transcribe(
        audio_path=audio_path,
        word_timestamps=not args.no_word_timestamps,
        language=args.language,
        task=args.task,
        verbose=False if args.json_stdout else None,
    )

    # Determine output path prefix
    audio_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else audio_dir
    output_prefix = os.path.join(out_dir, base_name)

    written = engine.export_files(
        result=result,
        output_prefix=output_prefix,
        formats=formats,
        title=args.title,
        artist=args.artist,
    )

    if args.json_stdout:
        print(json.dumps(to_json(result), ensure_ascii=False, indent=2))
    else:
        print("\n--- Transcription Complete ---")
        print(f"Language detected: {result.get('language')}")
        print(f"Device utilized:   {engine.target_device.upper()}")
        print("Generated files:")
        for fmt, path in written.items():
            print(f"  [{fmt.upper()}]: {path}")


if __name__ == "__main__":
    main()
