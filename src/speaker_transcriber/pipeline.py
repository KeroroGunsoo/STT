from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .align import assign_speakers
from .asr import Segment, transcribe as asr_transcribe
from .audio import load_audio
from .diarization import diarize
from .embeddings import SpeakerEmbedder
from .vad import SileroVAD, get_vad_model_path

DEFAULT_MODELS_DIR = Path(
    os.environ.get("SPEAKER_TRANSCRIBER_HOME", Path.home() / ".cache" / "speaker_transcriber")
)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str
    words: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transcript:
    source: str
    language: str | None
    duration: float
    speakers: list[str]
    segments: list[TranscriptSegment]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "language": self.language,
            "duration": self.duration,
            "speakers": self.speakers,
            "segments": [s.to_dict() for s in self.segments],
        }

    @property
    def text(self) -> str:
        return "\n".join(f"[{s.speaker}] {s.text}" for s in self.segments)


def _noop(*args, **kwargs) -> None:
    pass


def transcribe_file(
    path: str | Path,
    *,
    model: str = "base",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str | None = None,
    num_speakers: int | None = None,
    cluster_threshold: float = 0.30,
    vad_threshold: float = 0.5,
    min_silence_s: float = 0.3,
    min_speech_s: float = 0.25,
    window_s: float = 3.0,
    hop_s: float = 1.5,
    word_timestamps: bool = True,
    beam_size: int = 5,
    sample_rate: int = 16000,
    models_dir: str | Path | None = None,
    vad_model_path: str | Path | None = None,
    log: Callable[..., None] = _noop,
) -> Transcript:
    """Transcribe an audio file with speaker labels.

    Pipeline: decode -> ASR (faster-whisper) + diarization (Silero VAD +
    ECAPA-TDNN embeddings + agglomerative clustering) -> align speakers to
    transcript segments.
    """
    import logging

    logging.getLogger("speechbrain").setLevel(logging.WARNING)

    models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading audio: {path}")
    waveform = load_audio(path, sample_rate=sample_rate)
    duration = len(waveform) / sample_rate
    log(f"Audio: {duration:.1f}s at {sample_rate} Hz mono")

    log(f"Transcribing with faster-whisper model '{model}' ({device})...")
    segments, detected_language = asr_transcribe(
        waveform,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        models_dir=models_dir,
    )
    log(f"ASR produced {len(segments)} segment(s), language={detected_language}")

    log("Running speaker diarization (VAD + embeddings + clustering)...")
    vad = SileroVAD(vad_model_path or get_vad_model_path(), sample_rate=sample_rate)
    embedder = SpeakerEmbedder(models_dir, device=device)
    turns = diarize(
        waveform,
        sample_rate,
        vad,
        embedder,
        vad_threshold=vad_threshold,
        min_silence_s=min_silence_s,
        min_speech_s=min_speech_s,
        window_s=window_s,
        hop_s=hop_s,
        num_speakers=num_speakers,
        cluster_threshold=cluster_threshold,
    )
    speakers = sorted({speaker for _, _, speaker in turns}) or ["SPEAKER_00"]
    log(f"Detected {len(speakers)} speaker(s): {', '.join(speakers)}")

    labels = assign_speakers(segments, turns)
    t_segments = [
        TranscriptSegment(
            start=seg.start,
            end=seg.end,
            speaker=label,
            text=seg.text,
            words=[{"start": w.start, "end": w.end, "text": w.text} for w in seg.words],
        )
        for seg, label in zip(segments, labels)
    ]
    return Transcript(
        source=str(path),
        language=detected_language,
        duration=duration,
        speakers=speakers,
        segments=t_segments,
    )
