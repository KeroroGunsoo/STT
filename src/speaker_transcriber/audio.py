from __future__ import annotations

from pathlib import Path

import av
import numpy as np


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode any audio/video file to mono float32 PCM at ``sample_rate``.

    Uses PyAV (bundled ffmpeg), so mp3, m4a, wav, flac, ogg, mp4, ... all work.
    """
    path = str(path)
    container = av.open(path)
    try:
        streams = container.streams.audio
        if not streams:
            raise ValueError(f"No audio stream found in {path}")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(streams[0]):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    finally:
        container.close()
    if not chunks:
        raise ValueError(f"Could not decode any audio from {path}")
    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    return pcm


def save_wav(path: str | Path, waveform: np.ndarray, sample_rate: int = 16000) -> Path:
    """Write a mono float32 waveform to a 16-bit PCM wav file."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(waveform, dtype=np.float32), sample_rate, subtype="PCM_16")
    return path
