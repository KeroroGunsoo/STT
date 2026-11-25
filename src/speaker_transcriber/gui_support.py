from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .formats import _ts_txt
from .pipeline import Transcript

SPEAKER_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
]


@dataclass
class GuiOptions:
    input_path: str = ""
    output_dir: str = ""
    context_dir: str = ""
    model: str = "base"
    language: str = ""
    device: str = "cpu"
    num_speakers: int = 0
    cluster_threshold: float = 0.35
    formats: tuple[str, ...] = ("txt", "json", "srt")


def options_to_kwargs(opts: GuiOptions) -> dict:
    """Map GUI options to ``transcribe_file`` keyword arguments."""
    kwargs = {
        "model": opts.model,
        "device": opts.device,
        "cluster_threshold": opts.cluster_threshold,
    }
    language = opts.language.strip()
    if language:
        kwargs["language"] = language
    if opts.num_speakers and opts.num_speakers > 0:
        kwargs["num_speakers"] = int(opts.num_speakers)
    if opts.context_dir.strip():
        kwargs["context_dir"] = opts.context_dir.strip()
    return kwargs


def resolve_output_dir(opts: GuiOptions) -> str:
    if opts.output_dir.strip():
        return opts.output_dir.strip()
    if opts.input_path:
        return str(Path(opts.input_path).resolve().parent)
    return ""


def speaker_order(transcript: Transcript) -> list[str]:
    """Speakers in order of first appearance in the transcript."""
    seen: list[str] = []
    for seg in transcript.segments:
        if seg.speaker not in seen:
            seen.append(seg.speaker)
    return seen or list(transcript.speakers)


def speaker_color(speaker: str, order: list[str]) -> str:
    return SPEAKER_COLORS[order.index(speaker) % len(SPEAKER_COLORS)]


def transcript_lines(transcript: Transcript) -> list[tuple[str, str, str]]:
    """(timestamp, speaker, text) per segment, for rendering."""
    return [(_ts_txt(seg.start), seg.speaker, seg.text) for seg in transcript.segments]
