import json

from speaker_transcriber.formats import to_json, to_srt, to_txt
from speaker_transcriber.pipeline import Transcript, TranscriptSegment


def _transcript():
    return Transcript(
        source="meeting.mp3",
        language="en",
        duration=65.5,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=[
            TranscriptSegment(0.0, 2.5, "SPEAKER_00", "Hello Bob.", []),
            TranscriptSegment(3.0, 65.5, "SPEAKER_01", "Hi Alice, long time no see.", []),
        ],
    )


def test_to_txt():
    out = to_txt(_transcript())
    assert "[00:00:00.000] SPEAKER_00: Hello Bob." in out
    assert "[00:00:03.000] SPEAKER_01: Hi Alice, long time no see." in out
    assert "# speakers: SPEAKER_00, SPEAKER_01" in out


def test_to_srt():
    out = to_srt(_transcript())
    assert "1\n00:00:00,000 --> 00:00:02,500\n[SPEAKER_00] Hello Bob." in out
    assert "2\n00:00:03,000 --> 00:01:05,500\n[SPEAKER_01] Hi Alice" in out


def test_to_json_roundtrip():
    data = json.loads(to_json(_transcript()))
    assert data["language"] == "en"
    assert data["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert data["segments"][1]["speaker"] == "SPEAKER_01"
    assert data["segments"][0]["text"] == "Hello Bob."
