"""Generate a synthetic two-speaker conversation (16 kHz mono wav) for tests/demo.

Uses piper-tts with two distinct English voices so diarization has something
real to separate. Usage:

    python scripts/make_test_audio.py [output.wav]
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

VOICES = {
    "alice": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    ),
    "bob": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),
}

LINES = [
    ("alice", "Hi Bob, this is Alice. I wanted to talk about the project meeting tomorrow."),
    ("bob", "Hey Alice, sure. What time works best for you in the morning?"),
    ("alice", "How about nine thirty? We can review the slides together."),
    ("bob", "Nine thirty sounds perfect. I will bring the budget report."),
    ("alice", "Great, see you then. Thanks for your help with the presentation."),
    ("bob", "Anytime. Talk to you tomorrow morning, Alice."),
]

CACHE = Path.home() / ".cache" / "speaker_transcriber" / "piper"
TARGET_SR = 16000


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "speaker-transcriber-test"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def load_voice(name: str):
    from piper import PiperVoice

    onnx_url, json_url = VOICES[name]
    model = download(onnx_url, CACHE / f"{name}.onnx")
    config = download(json_url, CACHE / f"{name}.onnx.json")
    return PiperVoice.load(model, config)


def synth(voice, text: str) -> tuple[np.ndarray, int]:
    chunks = list(voice.synthesize(text))
    pcm = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32) / 32768.0
    return pcm, chunks[0].sample_rate


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    n_out = int(round(len(wav) * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/assets/conversation.wav")
    gap = np.zeros(int(0.5 * TARGET_SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    voices = {name: load_voice(name) for name in VOICES}
    for speaker, text in LINES:
        print(f"[{speaker}] {text}", file=sys.stderr)
        pcm, sr = synth(voices[speaker], text)
        parts.append(resample(pcm, sr, TARGET_SR))
        parts.append(gap)
    wav = np.concatenate(parts)
    from speaker_transcriber.audio import save_wav

    save_wav(out, wav, TARGET_SR)
    print(f"wrote {out} ({len(wav) / TARGET_SR:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
