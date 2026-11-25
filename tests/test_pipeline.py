import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

ASSETS = Path(__file__).resolve().parent / "assets"
CONVERSATION = ASSETS / "conversation.wav"


@pytest.fixture(scope="module")
def conversation_wav():
    if not CONVERSATION.exists():
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent.parent / "scripts" / "make_test_audio.py"), str(CONVERSATION)],
            check=True,
        )
    assert CONVERSATION.exists()
    return CONVERSATION


@pytest.fixture(scope="module")
def transcript(conversation_wav, tmp_path_factory):
    from speaker_transcriber import transcribe_file

    models_dir = tmp_path_factory.mktemp("models")
    return transcribe_file(
        conversation_wav,
        model="base",
        num_speakers=2,
        models_dir=models_dir,
        log=lambda *a: None,
    )


def test_vad_finds_speech_in_conversation(conversation_wav):
    from speaker_transcriber.audio import load_audio
    from speaker_transcriber.vad import SileroVAD, get_vad_model_path, speech_segments

    wav = load_audio(conversation_wav, 16000)
    vad = SileroVAD(get_vad_model_path(), sample_rate=16000)
    segs = speech_segments(wav, vad)
    assert len(segs) >= 4  # six spoken lines, possibly split
    speech = sum(e - s for s, e in segs)
    assert speech > 0.5 * len(wav) / 16000


def test_transcript_has_segments_and_text(transcript):
    assert len(transcript.segments) >= 3
    text = " ".join(s.text for s in transcript.segments).lower()
    assert any(word in text for word in ("alice", "bob", "meeting", "morning"))


def test_two_speakers_detected(transcript):
    assert transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]
    labels = {s.speaker for s in transcript.segments}
    assert labels == {"SPEAKER_00", "SPEAKER_01"}


def test_segments_are_time_ordered_and_labeled(transcript):
    prev_end = 0.0
    for seg in transcript.segments:
        assert seg.start >= prev_end - 1e-6
        assert seg.end > seg.start
        assert seg.speaker.startswith("SPEAKER_")
        assert seg.text.strip()
        prev_end = seg.end


def test_pipeline_with_context_dir(conversation_wav, tmp_path):
    from speaker_transcriber import transcribe_file

    ctx = tmp_path / "phd"
    ctx.mkdir()
    (ctx / "thesis.md").write_text(
        "# Thesis\nWe study retrieval augmented generation and BERT fine-tuning. "
        "The budget report and the presentation slides are due tomorrow morning.\n"
    )
    tr = transcribe_file(
        conversation_wav,
        model="tiny",
        context_dir=ctx,
        models_dir=tmp_path / "models",
        log=lambda *a: None,
    )
    assert tr.context is not None
    terms = [t.lower() for t in tr.context["terms"]]
    assert any(t in terms for t in ("budget", "retrieval", "bert", "slides"))
    assert len(tr.segments) >= 1


def test_cli_writes_outputs(conversation_wav, tmp_path, capsys):
    from speaker_transcriber.cli import main

    ctx = tmp_path / "phd"
    ctx.mkdir()
    (ctx / "notes.txt").write_text("PhD on retrieval augmented generation, BERT, budget reports.\n")
    outdir = tmp_path / "out"
    rc = main([
        str(conversation_wav),
        "-o", str(outdir),
        "--model", "tiny",
        "--num-speakers", "2",
        "--context-dir", str(ctx),
        "--models-dir", str(tmp_path / "models"),
    ])
    assert rc == 0
    for fmt in ("txt", "srt", "json"):
        assert (outdir / f"conversation.{fmt}").exists()
    data = json.loads((outdir / "conversation.json").read_text())
    assert data["segments"], "expected non-empty transcript"
    assert set(data["speakers"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert data["context"] and data["context"]["num_terms"] > 0
    stdout = capsys.readouterr().out
    assert "SPEAKER_" in stdout
