from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import SpeakerEmbedder
from .vad import SileroVAD, speech_segments


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: int  # raw cluster id; use `relabel` for SPEAKER_xx names


def build_windows(
    segments: list[tuple[float, float]],
    window_s: float = 3.0,
    hop_s: float = 1.5,
) -> list[tuple[float, float]]:
    """Slice speech segments into overlapping embedding windows, covering the tail."""
    windows: list[tuple[float, float]] = []
    for start, end in segments:
        if end - start <= window_s:
            windows.append((start, end))
            continue
        s = start
        while s + window_s < end - 1e-6:
            windows.append((s, s + window_s))
            s += hop_s
        windows.append((max(start, end - window_s), end))
    return windows


def cluster_embeddings(
    emb: np.ndarray,
    num_speakers: int | None = None,
    threshold: float = 0.35,
) -> np.ndarray:
    """Cluster L2-normalized embeddings with agglomerative clustering (cosine).

    If ``num_speakers`` is given it is used as a fixed cluster count, otherwise
    ``threshold`` is the cosine-distance cut-off.
    """
    from sklearn.cluster import AgglomerativeClustering

    emb = np.asarray(emb, dtype=np.float64)
    n = emb.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    if num_speakers is not None:
        k = max(1, min(int(num_speakers), n))
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average"
        )
    return model.fit_predict(emb)


def merge_turns(
    windows: list[tuple[float, float]],
    labels: np.ndarray,
    merge_gap_s: float = 0.75,
) -> list[SpeakerTurn]:
    """Merge consecutive same-speaker windows into speaker turns."""
    turns: list[SpeakerTurn] = []
    for (start, end), label in zip(windows, labels):
        if turns and turns[-1].speaker == int(label) and start - turns[-1].end <= merge_gap_s + 1e-6:
            turns[-1].end = max(turns[-1].end, end)
        else:
            turns.append(SpeakerTurn(float(start), float(end), int(label)))
    return turns


def relabel(turns: list[SpeakerTurn]) -> list[tuple[float, float, str]]:
    """Rename cluster ids to SPEAKER_00, SPEAKER_01, ... in order of first appearance."""
    names: dict[int, str] = {}
    out = []
    for t in turns:
        if t.speaker not in names:
            names[t.speaker] = f"SPEAKER_{len(names):02d}"
        out.append((t.start, t.end, names[t.speaker]))
    return out


def diarize(
    waveform: np.ndarray,
    sample_rate: int,
    vad: SileroVAD,
    embedder: SpeakerEmbedder,
    *,
    vad_threshold: float = 0.5,
    min_silence_s: float = 0.3,
    min_speech_s: float = 0.25,
    window_s: float = 3.0,
    hop_s: float = 1.5,
    num_speakers: int | None = None,
    cluster_threshold: float = 0.35,
    merge_gap_s: float = 0.75,
) -> list[tuple[float, float, str]]:
    """Full diarization: VAD -> embedding windows -> clustering -> speaker turns."""
    segments = speech_segments(
        waveform,
        vad,
        threshold=vad_threshold,
        min_speech_s=min_speech_s,
        min_silence_s=min_silence_s,
    )
    if not segments:
        return []
    windows = build_windows(segments, window_s=window_s, hop_s=hop_s)
    wavs = [waveform[int(s * sample_rate) : int(e * sample_rate)] for s, e in windows]
    emb = embedder.embed(wavs)
    labels = cluster_embeddings(emb, num_speakers=num_speakers, threshold=cluster_threshold)
    return relabel(merge_turns(windows, labels, merge_gap_s=merge_gap_s))
