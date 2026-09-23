"""Exporters for synchronized lyrics (LRC, Enhanced Karaoke LRC),
Subtitles (SRT, WebVTT), and structured JSON.
"""

from typing import Dict, Any, List


def format_lrc_timestamp(seconds: float) -> str:
    """Format seconds into LRC timestamp [mm:ss.xx] (hundredths of a second)."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    hundredths = int((remaining_seconds - int(remaining_seconds)) * 100)
    secs = int(remaining_seconds)
    return f"{minutes:02d}:{secs:02d}.{hundredths:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds into WebVTT timestamp HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def to_lrc(result: Dict[str, Any], title: str = "", artist: str = "") -> str:
    """
    Generate standard line-by-line synchronized LRC lyrics.
    Format: [mm:ss.xx] Lyric line
    """
    lines: List[str] = []
    
    # Optional metadata headers
    if title:
        lines.append(f"[ti:{title}]")
    if artist:
        lines.append(f"[ar:{artist}]")
    lines.append("[by:Whisper-XPU-Vivestream]")
    
    segments = result.get("segments", [])
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start_time = seg.get("start", 0.0)
        ts = format_lrc_timestamp(start_time)
        lines.append(f"[{ts}] {text}")

    return "\n".join(lines) + "\n"


def to_enhanced_lrc(result: Dict[str, Any], title: str = "", artist: str = "") -> str:
    """
    Generate enhanced word-by-word Karaoke LRC lyrics.
    Format: [mm:ss.xx] <mm:ss.xx> Word1 <mm:ss.xx> Word2 ...
    Falls back to line-by-line if word timestamps are not present.
    """
    lines: List[str] = []

    if title:
        lines.append(f"[ti:{title}]")
    if artist:
        lines.append(f"[ar:{artist}]")
    lines.append("[by:Whisper-XPU-Vivestream-Enhanced]")

    segments = result.get("segments", [])
    for seg in segments:
        words = seg.get("words", [])
        start_time = seg.get("start", 0.0)
        seg_ts = format_lrc_timestamp(start_time)

        if words:
            word_parts = []
            for w in words:
                w_text = w.get("word", "").strip()
                if not w_text:
                    continue
                w_start = format_lrc_timestamp(w.get("start", start_time))
                word_parts.append(f"<{w_start}> {w_text}")
            
            if word_parts:
                lines.append(f"[{seg_ts}] " + " ".join(word_parts))
            else:
                text = seg.get("text", "").strip()
                if text:
                    lines.append(f"[{seg_ts}] {text}")
        else:
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"[{seg_ts}] {text}")

    return "\n".join(lines) + "\n"


def to_srt(result: Dict[str, Any]) -> str:
    """Generate standard SubRip (.srt) subtitles."""
    blocks: List[str] = []
    segments = result.get("segments", [])

    idx = 1
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start_str = format_srt_timestamp(seg.get("start", 0.0))
        end_str = format_srt_timestamp(seg.get("end", 0.0))
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
        idx += 1

    return "\n".join(blocks)


def to_vtt(result: Dict[str, Any]) -> str:
    """Generate WebVTT (.vtt) subtitles."""
    lines: List[str] = ["WEBVTT", ""]
    segments = result.get("segments", [])

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start_str = format_vtt_timestamp(seg.get("start", 0.0))
        end_str = format_vtt_timestamp(seg.get("end", 0.0))
        lines.append(f"{start_str} --> {end_str}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def to_json(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return cleaned, structured dictionary suitable for JSON serialization in Tauri/Rust."""
    cleaned_segments = []
    for s in result.get("segments", []):
        words = []
        for w in s.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": round(w.get("start", 0.0), 3),
                "end": round(w.get("end", 0.0), 3),
                "probability": round(w.get("probability", 1.0), 3),
            })
        cleaned_segments.append({
            "id": s.get("id", 0),
            "start": round(s.get("start", 0.0), 3),
            "end": round(s.get("end", 0.0), 3),
            "text": s.get("text", "").strip(),
            "words": words,
        })

    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", ""),
        "segments": cleaned_segments,
    }
