from speaker_transcriber.align import assign_speakers
from speaker_transcriber.asr import Segment, Word


def test_assign_uses_word_overlap():
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    segments = [
        Segment(0.2, 1.8, "hello there", [Word(0.2, 0.9, "hello"), Word(1.0, 1.8, "there")]),
        Segment(2.1, 3.9, "hi alice", [Word(2.1, 2.8, "hi"), Word(2.9, 3.9, "alice")]),
    ]
    assert assign_speakers(segments, turns) == ["SPEAKER_00", "SPEAKER_01"]


def test_assign_falls_back_to_segment_and_previous():
    turns = [(0.0, 1.0, "SPEAKER_00"), (5.0, 6.0, "SPEAKER_01")]
    segments = [
        Segment(0.1, 0.9, "one", []),
        Segment(3.0, 4.0, "gap speech", []),  # no overlap -> previous speaker
    ]
    assert assign_speakers(segments, turns) == ["SPEAKER_00", "SPEAKER_00"]


def test_assign_no_turns_single_speaker():
    segments = [Segment(0.0, 1.0, "hello", [])]
    assert assign_speakers(segments, []) == ["SPEAKER_00"]
