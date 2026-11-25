from __future__ import annotations

from .asr import Segment

Turn = tuple[float, float, str]  # (start, end, speaker)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[str]:
    """Assign a speaker label to each ASR segment.

    Uses word-level timestamps when available (majority of overlapped word
    duration per speaker); falls back to segment/turn interval overlap, then
    to the previous segment's speaker, then to the nearest turn.
    """
    labels: list[str] = []
    prev: str | None = None
    for seg in segments:
        spans = [(w.start, w.end) for w in seg.words] or [(seg.start, seg.end)]
        scores: dict[str, float] = {}
        for t_start, t_end, speaker in turns:
            overlap = sum(_overlap(a, b, t_start, t_end) for a, b in spans)
            if overlap > 0:
                scores[speaker] = scores.get(speaker, 0.0) + overlap
        if scores:
            label = max(scores.items(), key=lambda kv: kv[1])[0]
        elif prev is not None:
            label = prev
        elif turns:
            center = (seg.start + seg.end) / 2.0
            label = min(turns, key=lambda t: abs((t[0] + t[1]) / 2.0 - center))[2]
        else:
            label = "SPEAKER_00"
        labels.append(label)
        prev = label
    return labels
