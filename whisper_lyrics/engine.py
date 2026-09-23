"""Lyrics and Subtitle Generation Engine.
Manages Whisper model loading on Intel XPU (Arc A770) and CPU,
audio transcription with word-level alignment, and output generation.
"""

import os
import sys
import json
import logging
from typing import Optional, Union, Dict, Any, List

import torch
import whisper

from .exporters import to_lrc, to_enhanced_lrc, to_srt, to_vtt, to_json

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("whisper_lyrics")


def get_available_devices() -> Dict[str, Any]:
    """Inspect and report available PyTorch compute devices."""
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda_available = torch.cuda.is_available()
    
    xpu_device_name = None
    if xpu_available:
        try:
            xpu_device_name = torch.xpu.get_device_name(0)
        except Exception:
            xpu_device_name = "Intel XPU Accelerator"

    cuda_device_name = None
    if cuda_available:
        try:
            cuda_device_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_device_name = "CUDA Accelerator"

    return {
        "xpu": {
            "available": xpu_available,
            "device_name": xpu_device_name,
            "device_count": torch.xpu.device_count() if xpu_available else 0,
        },
        "cuda": {
            "available": cuda_available,
            "device_name": cuda_device_name,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        },
        "cpu": {
            "available": True,
            "device_name": "Host CPU",
        },
    }


def resolve_device(requested: str = "auto") -> str:
    """
    Resolve requested device ('auto', 'xpu', 'cuda', 'cpu') to a valid device.
    Priority in 'auto' mode: xpu -> cuda -> cpu.
    """
    req = (requested or "auto").lower().strip()
    devices = get_available_devices()

    if req == "auto":
        if devices["xpu"]["available"]:
            return "xpu"
        elif devices["cuda"]["available"]:
            return "cuda"
        return "cpu"

    if req == "xpu":
        if devices["xpu"]["available"]:
            return "xpu"
        logger.warning("XPU requested but not available. Falling back to CPU.")
        return "cpu"

    if req == "cuda":
        if devices["cuda"]["available"]:
            return "cuda"
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    if req == "cpu":
        return "cpu"

    # Default fallback
    logger.warning("Unknown device '%s'. Falling back to CPU.", requested)
    return "cpu"


class LyricsEngine:
    """
    High-level Whisper engine for generating synchronized lyrics and subtitles.
    Supports Intel XPU acceleration and CPU fallback.
    """

    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        download_root: Optional[str] = None,
    ):
        self.model_name = model_name
        self.target_device = resolve_device(device)
        if download_root is None:
            self.download_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
            )
        else:
            self.download_root = download_root
        os.makedirs(self.download_root, exist_ok=True)
        self.model: Optional[whisper.Whisper] = None
        self.device_info = get_available_devices()
        self.load_model(self.model_name, self.target_device)

    def load_model(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Load or reload the Whisper model onto the specified device."""
        if model_name:
            self.model_name = model_name
        if device:
            self.target_device = resolve_device(device)

        dev_name = "CPU"
        if self.target_device == "xpu" and self.device_info["xpu"]["available"]:
            dev_name = self.device_info["xpu"]["device_name"] or "Intel XPU"
        elif self.target_device == "cuda" and self.device_info["cuda"]["available"]:
            dev_name = self.device_info["cuda"]["device_name"] or "CUDA GPU"

        logger.info(
            "Loading Whisper model '%s' on %s (%s)...",
            self.model_name,
            self.target_device.upper(),
            dev_name,
        )

        self.model = whisper.load_model(
            self.model_name,
            device=self.target_device,
            download_root=self.download_root,
        )

        logger.info(
            "Successfully initialized Whisper '%s' on %s.",
            self.model_name,
            self.target_device.upper(),
        )

    def transcribe(
        self,
        audio_path: str,
        word_timestamps: bool = True,
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        initial_prompt: Optional[str] = None,
        verbose: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file and extract segment & word-level timestamps.
        
        Args:
            audio_path: Path to the audio or video file.
            word_timestamps: Enable word-level alignment for karaoke lyrics.
            language: Optional language code (e.g. 'en', 'es', 'ja', 'bn'). None for auto-detection.
            task: 'transcribe' or 'translate' (translates non-English to English).
            temperature: Sampling temperature (0.0 for greedy/most deterministic).
            initial_prompt: Optional context or lyric hints.
            verbose: Print segment progress to stdout.
            
        Returns:
            Dict containing transcript, language, and segments with word timestamps.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.model is None:
            self.load_model()

        # Intel Arc GPUs (XPU) and CUDA support FP16. CPU uses FP32.
        use_fp16 = (self.target_device in ("xpu", "cuda"))

        logger.info(
            "Transcribing '%s' [device=%s, fp16=%s, word_timestamps=%s, language=%s]",
            os.path.basename(audio_path),
            self.target_device,
            use_fp16,
            word_timestamps,
            language or "auto",
        )

        result = whisper.transcribe(
            model=self.model,
            audio=audio_path,
            word_timestamps=word_timestamps,
            language=language,
            task=task,
            temperature=temperature,
            initial_prompt=initial_prompt,
            verbose=verbose,
            fp16=use_fp16,
        )

        # Clear transient memory on XPU if supported
        if self.target_device == "xpu" and hasattr(torch.xpu, "empty_cache"):
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass

        return result

    def export_files(
        self,
        result: Dict[str, Any],
        output_prefix: str,
        formats: Optional[List[str]] = None,
        title: str = "",
        artist: str = "",
    ) -> Dict[str, str]:
        """
        Export transcription results to specified lyric and subtitle formats.
        
        Supported formats: 'lrc', 'elrc', 'srt', 'vtt', 'json'.
        """
        if formats is None:
            formats = ["lrc", "elrc", "srt", "vtt", "json"]

        written_files = {}

        # Ensure directory exists
        out_dir = os.path.dirname(output_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        for fmt in formats:
            fmt_lower = fmt.lower().strip()
            
            if fmt_lower == "lrc":
                content = to_lrc(result, title=title, artist=artist)
                path = f"{output_prefix}.lrc"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                written_files["lrc"] = path

            elif fmt_lower == "elrc":
                content = to_enhanced_lrc(result, title=title, artist=artist)
                path = f"{output_prefix}.enhanced.lrc"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                written_files["elrc"] = path

            elif fmt_lower == "srt":
                content = to_srt(result)
                path = f"{output_prefix}.srt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                written_files["srt"] = path

            elif fmt_lower == "vtt":
                content = to_vtt(result)
                path = f"{output_prefix}.vtt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                written_files["vtt"] = path

            elif fmt_lower == "json":
                structured = to_json(result)
                path = f"{output_prefix}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(structured, f, ensure_ascii=False, indent=2)
                written_files["json"] = path

        return written_files

    def get_status(self) -> Dict[str, Any]:
        """Return engine runtime state and hardware specs."""
        return {
            "model_name": self.model_name,
            "device": self.target_device,
            "available_devices": self.device_info,
            "torch_version": torch.__version__,
        }
