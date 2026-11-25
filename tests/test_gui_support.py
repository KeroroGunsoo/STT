from speaker_transcriber.gui_support import (
    GuiOptions,
    options_to_kwargs,
    resolve_output_dir,
    speaker_color,
    speaker_order,
    transcript_lines,
)
from speaker_transcriber.pipeline import Transcript, TranscriptSegment


def test_options_to_kwargs_defaults():
    kw = options_to_kwargs(GuiOptions(input_path="a.mp3"))
    assert kw == {"model": "base", "device": "cpu", "cluster_threshold": 0.30}


def test_options_to_kwargs_overrides():
    opts = GuiOptions(input_path="a.mp3", model="small", language=" en ", device="cuda", num_speakers=3, cluster_threshold=0.25)
    kw = options_to_kwargs(opts)
    assert kw["model"] == "small"
    assert kw["device"] == "cuda"
    assert kw["language"] == "en"
    assert kw["num_speakers"] == 3
    assert kw["cluster_threshold"] == 0.25


def test_options_to_kwargs_speakers_zero_means_auto():
    kw = options_to_kwargs(GuiOptions(num_speakers=0))
    assert "num_speakers" not in kw


def test_resolve_output_dir(tmp_path):
    assert resolve_output_dir(GuiOptions(input_path=str(tmp_path / "a.mp3"), output_dir="")) == str(tmp_path)
    assert resolve_output_dir(GuiOptions(input_path="x", output_dir="/tmp/out")) == "/tmp/out"


def _transcript():
    return Transcript(
        source="a.wav",
        language="en",
        duration=4.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            TranscriptSegment(0.0, 2.0, "SPEAKER_00", "Hello.", []),
            TranscriptSegment(2.0, 4.0, "SPEAKER_01", "Hi there.", []),
            TranscriptSegment(4.0, 5.0, "SPEAKER_00", "Bye.", []),
        ],
    )


def test_speaker_order_and_color():
    tr = _transcript()
    order = speaker_order(tr)
    assert order == ["SPEAKER_00", "SPEAKER_01"]
    assert speaker_color("SPEAKER_00", order) != speaker_color("SPEAKER_01", order)


def test_transcript_lines():
    lines = transcript_lines(_transcript())
    assert lines[0] == ("00:00:00.000", "SPEAKER_00", "Hello.")
    assert lines[1] == ("00:00:02.000", "SPEAKER_01", "Hi there.")
