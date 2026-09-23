"""Whisper Lyrics & Subtitle Generation Engine
Optimized for Intel XPU (Arc A770) and CPU inference.
Designed for integration with Vivestream Revived.
"""

from .engine import LyricsEngine
from .exporters import to_lrc, to_enhanced_lrc, to_srt, to_vtt, to_json

__version__ = "1.0.0"
__all__ = [
    "LyricsEngine",
    "to_lrc",
    "to_enhanced_lrc",
    "to_srt",
    "to_vtt",
    "to_json",
]
