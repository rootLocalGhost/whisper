"""Lightweight local HTTP JSON server for whisper_lyrics.
Allows Vivestream Revived to keep the model warm in Arc A770 VRAM
and transcribe songs on-demand without cold-start overhead.
Uses Python's standard library http.server (no extra dependencies required).
"""

import os
import sys
import json
import logging
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .engine import LyricsEngine, get_available_devices
from .exporters import to_lrc, to_enhanced_lrc, to_srt, to_vtt, to_json

logger = logging.getLogger("whisper_lyrics.server")
GLOBAL_ENGINE: Optional[LyricsEngine] = None


class WhisperRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health" or self.path == "/":
            global GLOBAL_ENGINE
            status = {
                "status": "ready" if GLOBAL_ENGINE else "initializing",
                "devices": get_available_devices(),
            }
            if GLOBAL_ENGINE:
                status.update(GLOBAL_ENGINE.get_status())
            self._send_json(200, status)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        global GLOBAL_ENGINE
        if self.path != "/transcribe":
            self._send_json(404, {"error": "Endpoint not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Missing JSON request body"})
            return

        try:
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {str(e)}"})
            return

        audio_file = body.get("file")
        if not audio_file or not os.path.exists(audio_file):
            self._send_json(400, {"error": f"Audio file not found: {audio_file}"})
            return

        model_name = body.get("model")
        device = body.get("device")
        word_timestamps = body.get("word_timestamps", True)
        language = body.get("language")
        task = body.get("task", "transcribe")
        export_formats = body.get("formats", ["lrc", "elrc", "srt"])
        title = body.get("title", "")
        artist = body.get("artist", "")

        try:
            # Switch model/device if requested
            if GLOBAL_ENGINE is None:
                GLOBAL_ENGINE = LyricsEngine(model_name=model_name or "small", device=device or "auto")
            elif (model_name and model_name != GLOBAL_ENGINE.model_name) or (device and device != GLOBAL_ENGINE.target_device):
                GLOBAL_ENGINE.load_model(model_name=model_name, device=device)

            result = GLOBAL_ENGINE.transcribe(
                audio_path=audio_file,
                word_timestamps=word_timestamps,
                language=language,
                task=task,
            )

            # Build response
            response = {
                "success": True,
                "language": result.get("language"),
                "text": result.get("text", "").strip(),
                "device": GLOBAL_ENGINE.target_device,
                "lrc": to_lrc(result, title=title, artist=artist) if "lrc" in export_formats else None,
                "enhanced_lrc": to_enhanced_lrc(result, title=title, artist=artist) if "elrc" in export_formats else None,
                "srt": to_srt(result) if "srt" in export_formats else None,
                "vtt": to_vtt(result) if "vtt" in export_formats else None,
                "structured": to_json(result) if "json" in export_formats else None,
            }

            # Optional file export if requested
            if body.get("save_files", False):
                audio_dir = os.path.dirname(audio_file)
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                out_dir = body.get("output_dir") or audio_dir
                out_prefix = os.path.join(out_dir, base_name)
                saved = GLOBAL_ENGINE.export_files(result, out_prefix, export_formats, title=title, artist=artist)
                response["saved_files"] = saved

            self._send_json(200, response)

        except Exception as e:
            logger.exception("Error processing transcription request")
            self._send_json(500, {"success": False, "error": str(e)})


def run_server(host: str = "127.0.0.1", port: int = 5005, model: str = "small", device: str = "auto"):
    global GLOBAL_ENGINE
    logger.info("Initializing Whisper Lyrics Server on http://%s:%d...", host, port)
    GLOBAL_ENGINE = LyricsEngine(model_name=model, device=device)

    server = HTTPServer((host, port), WhisperRequestHandler)
    logger.info("Whisper Lyrics Server is running on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Whisper Lyrics Server...")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Whisper Lyrics Local HTTP Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    parser.add_argument("--port", "-p", type=int, default=5005, help="Port to listen on")
    parser.add_argument("--model", "-m", default="small", help="Default Whisper model to preload")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "xpu", "cuda", "cpu"], help="Compute device")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, model=args.model, device=args.device)


if __name__ == "__main__":
    main()
