from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def transcribe(
    waveform: np.ndarray,
    *,
    model: str = "base",
    device: str = "cpu",
    compute_type: str | None = None,
    language: str | None = None,
    word_timestamps: bool = True,
    beam_size: int = 5,
    vad_filter: bool = True,
    models_dir: str | Path | None = None,
) -> tuple[list[Segment], str | None]:
    """Transcribe a 16 kHz mono waveform with faster-whisper.

    Returns (segments, detected_language). Segments carry word-level
    timestamps when ``word_timestamps`` is enabled.
    """
    from faster_whisper import WhisperModel

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"
    kwargs: dict = {"device": device, "compute_type": compute_type}
    if models_dir is not None:
        kwargs["download_root"] = str(Path(models_dir) / "asr")
    whisper = WhisperModel(model, **kwargs)
    segments, info = whisper.transcribe(
        np.asarray(waveform, dtype=np.float32),
        language=language,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    out: list[Segment] = []
    for seg in segments:
        words = (
            [Word(float(w.start), float(w.end), w.word) for w in seg.words]
            if word_timestamps and seg.words
            else []
        )
        out.append(Segment(float(seg.start), float(seg.end), seg.text.strip(), words))
    return out, info.language
