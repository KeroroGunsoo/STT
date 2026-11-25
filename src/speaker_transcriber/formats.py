from __future__ import annotations

import json
from pathlib import Path

from .pipeline import Transcript


def _ts_txt(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _ts_srt(seconds: float) -> str:
    return _ts_txt(seconds).replace(".", ",")


def to_txt(transcript: Transcript) -> str:
    lines = [
        f"# source: {transcript.source}",
        f"# duration: {_ts_txt(transcript.duration)}",
        f"# language: {transcript.language or 'unknown'}",
        f"# speakers: {', '.join(transcript.speakers)}",
        "",
    ]
    for seg in transcript.segments:
        lines.append(f"[{_ts_txt(seg.start)}] {seg.speaker}: {seg.text}")
    return "\n".join(lines) + "\n"


def to_srt(transcript: Transcript) -> str:
    blocks = []
    for i, seg in enumerate(transcript.segments, start=1):
        blocks.append(
            f"{i}\n{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}\n[{seg.speaker}] {seg.text}\n"
        )
    return "\n".join(blocks)


def to_json(transcript: Transcript) -> str:
    return json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False) + "\n"


_WRITERS = {"txt": to_txt, "srt": to_srt, "json": to_json}


def write_outputs(
    transcript: Transcript,
    output_dir: str | Path,
    formats: tuple[str, ...] = ("txt", "json", "srt"),
) -> dict[str, Path]:
    """Write the transcript in each requested format; returns {format: path}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(transcript.source).stem
    written: dict[str, Path] = {}
    for fmt in formats:
        writer = _WRITERS[fmt]
        path = output_dir / f"{stem}.{fmt}"
        path.write_text(writer(transcript), encoding="utf-8")
        written[fmt] = path
    return written
