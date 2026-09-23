"""Vivestream Revived Whisper Service Entrypoint.
Provides a unified interface for Vivestream Revived:
  - --server : Start background HTTP daemon on port 5005 (zero cold-start IPC)
  - --cli    : Transcribe a file via CLI and return JSON or save lyrics/subtitles
  - --gui    : Launch Gradio studio on port 7860
  - --check  : Print hardware and model diagnostics in JSON for Tauri
"""

import os
import sys
import json
import argparse

# Ensure local directories
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
os.environ["WHISPER_MODELS_DIR"] = MODELS_DIR

from whisper_lyrics.engine import LyricsEngine, get_available_devices
from whisper_lyrics.server import run_server
from whisper_lyrics.cli import main as cli_main


def run_diagnostics():
    """Print hardware and model status for Vivestream Revived Tauri app."""
    devices = get_available_devices()
    downloaded_models = []
    if os.path.isdir(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pt"):
                size_mb = os.path.getsize(os.path.join(MODELS_DIR, f)) / (1024 * 1024)
                downloaded_models.append({
                    "model": f.replace(".pt", ""),
                    "filename": f,
                    "size_mb": round(size_mb, 1),
                })

    status = {
        "status": "ready",
        "devices": devices,
        "models_dir": MODELS_DIR,
        "installed_models": downloaded_models,
        "default_device": "xpu" if devices["xpu"]["available"] else ("cuda" if devices["cuda"]["available"] else "cpu"),
    }
    print(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Vivestream Whisper Service")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--server", action="store_true", help="Run local HTTP daemon for Vivestream Revived")
    group.add_argument("--gui", action="store_true", help="Launch Gradio Web Studio")
    group.add_argument("--check", action="store_true", help="Print diagnostics JSON and exit")
    group.add_argument("--transcribe", type=str, help="Transcribe audio file via CLI")

    parser.add_argument("--port", type=int, default=5005, help="HTTP Server port (default: 5005)")
    parser.add_argument("--model", type=str, default="small", help="Default Whisper model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "xpu", "cuda", "cpu"], help="Compute device")

    # If no flags passed, pass through to CLI or show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Check mode
    if "--check" in sys.argv:
        run_diagnostics()
        sys.exit(0)

    # GUI mode
    if "--gui" in sys.argv:
        from app import demo
        demo.launch(server_name="127.0.0.1", server_port=7860)
        sys.exit(0)

    # Server mode
    if "--server" in sys.argv:
        args, _ = parser.parse_known_args()
        run_server(host="127.0.0.1", port=args.port, model=args.model, device=args.device)
        sys.exit(0)

    # Otherwise forward directly to CLI runner
    cli_main()


if __name__ == "__main__":
    main()
