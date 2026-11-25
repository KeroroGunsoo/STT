from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .formats import to_txt, write_outputs
from .pipeline import transcribe_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speaker-transcribe",
        description=(
            "Transcribe an audio recording and label each segment with the "
            "speaker (diarization). Writes .txt/.srt/.json next to the input "
            "(or into --output-dir) and prints the transcript to stdout."
        ),
    )
    p.add_argument("input", help="Audio file (mp3, wav, m4a, mp4, flac, ogg, ...)")
    p.add_argument("-o", "--output-dir", default=None, help="Directory for output files (default: alongside the input)")
    p.add_argument("--model", default="base", help="faster-whisper model: tiny, base, small, medium, large-v3, ... (default: base)")
    p.add_argument("--language", default=None, help="Language code (e.g. 'en'); default: auto-detect")
    p.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument("--compute-type", default=None, help="faster-whisper compute type (default: int8 on cpu, float16 on cuda)")
    p.add_argument("--num-speakers", type=int, default=None, help="Force this many speakers (default: estimated by clustering)")
    p.add_argument("--cluster-threshold", type=float, default=0.30, help="Cosine-distance threshold for speaker clustering (default: 0.30)")
    p.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD speech probability threshold (default: 0.5)")
    p.add_argument("--min-silence", type=float, default=0.3, help="Merge speech regions separated by less than this many seconds (default: 0.3)")
    p.add_argument("--no-word-timestamps", action="store_true", help="Disable word-level timestamps (faster, coarser speaker assignment)")
    p.add_argument("--formats", default="txt,json,srt", help="Comma-separated output formats: txt,json,srt (default: txt,json,srt)")
    p.add_argument("--models-dir", default=None, help="Where to cache downloaded models (default: ~/.cache/speaker_transcriber)")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress progress on stderr and the transcript on stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else (lambda *a, **k: print(*a, file=sys.stderr))

    transcript = transcribe_file(
        args.input,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        num_speakers=args.num_speakers,
        cluster_threshold=args.cluster_threshold,
        vad_threshold=args.vad_threshold,
        min_silence_s=args.min_silence,
        word_timestamps=not args.no_word_timestamps,
        models_dir=args.models_dir,
        log=log,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).resolve().parent
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    written = write_outputs(transcript, output_dir, formats=formats)

    if not args.quiet:
        sys.stdout.write(to_txt(transcript))
        for fmt, path in written.items():
            print(f"wrote {fmt}: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
