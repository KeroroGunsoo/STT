# speaker-transcriber

Transcribe an audio recording **and label who said what**. The app takes any
recording (mp3, wav, m4a, mp4, flac, ogg, ...), identifies the different
speakers (diarization), and produces a timestamped transcription with speaker
labels.

## How it works

```
audio file ──▶ decode to 16 kHz mono (PyAV/ffmpeg)
                  ├─▶ ASR: faster-whisper (word-level timestamps)
                  └─▶ Diarization: Silero VAD → ECAPA-TDNN speaker embeddings
                                   (speechbrain, VoxCeleb) → agglomerative
                                   clustering (cosine) → speaker turns
                  └─▶ align: each transcript segment/word gets the speaker
                             whose turn overlaps it most
```

All models are public on Hugging Face — no API keys or gated-model tokens
needed. Models are downloaded once and cached (default
`~/.cache/speaker_transcriber`, override with `--models-dir` or
`SPEAKER_TRANSCRIBER_HOME`).

## Install

```bash
# CPU-only torch first (much smaller than the default CUDA wheel)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .          # from this directory
pip install -e '.[test]'  # additionally installs pytest + piper-tts for tests
```

Requires Python ≥ 3.10. For GPU, install the CUDA torch build instead and pass
`--device cuda`.

## Usage

```bash
speaker-transcribe meeting.mp3                 # transcript to stdout + .txt/.json/.srt files
speaker-transcribe meeting.mp3 -o out/ --model small
speaker-transcribe call.m4a --num-speakers 3 --language en
python -m speaker_transcriber recording.wav    # same thing
```

Outputs (written next to the input, or into `-o DIR`):

- `<name>.txt` — `[00:00:05.860] SPEAKER_01: Hey Alice, sure. ...`
- `<name>.srt` — subtitle file with `[SPEAKER_xx]` prefixes
- `<name>.json` — full structured result incl. word-level timestamps

Useful flags: `--model` (tiny/base/small/medium/large-v3, default `base`),
`--language` (default auto-detect), `--num-speakers` (force a count; default
estimated by clustering with `--cluster-threshold 0.35`), `--vad-threshold`,
`--min-silence`, `--no-word-timestamps`, `--formats txt,srt`, `-q`.

## Context-aware transcription (PhD project folder)

Point the app at your project folder and it will use your documents to get
domain-specific words right:

```bash
speaker-transcribe defense.mp3 --context-dir "C:\Users\You\phd-thesis"
```

(or the "Context folder" field in the GUI). The folder is scanned for text
documents (`.md`, `.txt`, `.tex`, `.bib`, `.py`, `.docx`, `.odt`, …); a
vocabulary of technical terms, acronyms and names is built (case-insensitive,
stop-words removed, most frequent first, capped at 300 terms). Every
transcribed word that is **not** itself in the vocabulary is fuzzy-matched
against it (ratio ≥ `--vocab-threshold`, default 0.92, min length 5) and
replaced with the closest domain term — so a mis-heard "jenneration" becomes
"generation", "bert" becomes "BERT", etc., preserving punctuation and casing.
This runs after decoding (word-level timestamps are kept intact), which keeps
segment boundaries stable.

The JSON/txt outputs record which context was used (`context` field / `# context:`
header), so you can see how many terms were picked up from your folder.

## Desktop GUI

```bash
speaker-transcribe-gui        # or: python -m speaker_transcriber.gui
```

Tkinter-based desktop app: pick an audio file, adjust options (model, language,
speaker count, cluster threshold, output formats), hit **Transcribe**. The
transcript appears with each speaker in a different color, and the
`.txt`/`.srt`/`.json` files are written to the output folder. Tk ships with the
python.org installers on Windows/macOS; on Linux install the `python3-tk`
package. First run downloads the models (see log pane).

### Python API

```python
from speaker_transcriber import transcribe_file

tr = transcribe_file("meeting.mp3", model="base")
for seg in tr.segments:
    print(f"[{seg.start:7.2f}] {seg.speaker}: {seg.text}")
print(tr.speakers)   # ['SPEAKER_00', 'SPEAKER_01', ...]
```

## Tests

```bash
python -m pytest tests -m "not slow"   # fast unit tests
python -m pytest tests                 # everything, incl. end-to-end
```

The slow tests synthesize a two-speaker conversation with
[piper-tts](scripts/make_test_audio.py) (`python scripts/make_test_audio.py
out.wav`) and verify that the pipeline detects both speakers and transcribes
them correctly. A ready-made sample is in `tests/assets/conversation.wav` /
`.mp3`, with example output in `demo_out/`.

## Notes

- Runs on CPU; `base` is a good speed/accuracy default, `small` for better
  accuracy. int8 quantization is used on CPU automatically.
- Speaker labels are relative (`SPEAKER_00`, ...) — the same label means the
  same voice throughout the file. Relabeling to real names is out of scope.
- `--num-speakers` helps when clustering under/over-segments; otherwise tune
  `--cluster-threshold` (lower = more speakers).
- Windows note: model files are copied into the cache with plain file copies
  (never symlinked/hard-linked), which avoids `WinError 1314` from
  speechbrain's default symlink strategy.
