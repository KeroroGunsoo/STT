from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SILERO_REPO = "onnx-community/silero-vad"
SILERO_FILE = "onnx/model.onnx"


def get_vad_model_path() -> Path:
    """Return a local path to the Silero VAD onnx model, downloading it once."""
    override = os.environ.get("SPEAKER_TRANSCRIBER_VAD_ONNX")
    if override:
        return Path(override)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(SILERO_REPO, SILERO_FILE))


class SileroVAD:
    """Streaming wrapper around the Silero VAD onnx model (16 kHz, 512-sample windows)."""

    def __init__(self, model_path: str | Path, sample_rate: int = 16000, window: int = 512):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.sample_rate = sample_rate
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        """Speech probability for one ``window``-sample chunk (zero-padded if short)."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[0] < self.window:
            chunk = np.pad(chunk, (0, self.window - chunk.shape[0]))
        elif chunk.shape[0] > self.window:
            chunk = chunk[: self.window]
        out, self._state = self.session.run(
            ["output", "stateN"],
            {
                "input": chunk[None, :],
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            },
        )
        return float(out[0, 0])


def speech_segments(
    waveform: np.ndarray,
    vad: SileroVAD,
    *,
    threshold: float = 0.5,
    min_speech_s: float = 0.25,
    min_silence_s: float = 0.3,
    smoothing_windows: int = 3,
) -> list[tuple[float, float]]:
    """Detect speech regions in a 16 kHz mono waveform; returns (start, end) in seconds."""
    sr, window = vad.sample_rate, vad.window
    n_chunks = max(1, int(np.ceil(len(waveform) / window)))
    probs = np.empty(n_chunks, dtype=np.float32)
    vad.reset()
    for i in range(n_chunks):
        probs[i] = vad.prob(waveform[i * window : (i + 1) * window])
    if smoothing_windows > 1 and n_chunks > 1:
        kernel = np.ones(smoothing_windows, dtype=np.float32) / smoothing_windows
        probs = np.convolve(probs, kernel, mode="same")

    speech = probs >= threshold
    runs: list[list[float]] = []
    i = 0
    while i < n_chunks:
        if speech[i]:
            j = i
            while j + 1 < n_chunks and speech[j + 1]:
                j += 1
            runs.append([i * window / sr, (j + 1) * window / sr])
            i = j + 1
        else:
            i += 1

    merged: list[list[float]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < min_silence_s:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(float(a), float(b)) for a, b in merged if b - a >= min_speech_s]
