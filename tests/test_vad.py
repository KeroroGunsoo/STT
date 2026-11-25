import numpy as np
import pytest

from speaker_transcriber.vad import SileroVAD, get_vad_model_path, speech_segments

SR = 16000


@pytest.fixture(scope="module")
def vad():
    return SileroVAD(get_vad_model_path(), sample_rate=SR)


def _tone(freq, seconds):
    t = np.arange(int(SR * seconds)) / SR
    return 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_silence_yields_no_segments(vad):
    wav = np.zeros(int(SR * 2), dtype=np.float32)
    assert speech_segments(wav, vad) == []


def test_pure_tones_do_not_trigger_vad(vad):
    # Silero is speech-trained: steady tones must not be flagged as speech.
    wav = np.concatenate([_tone(440, 1.0), np.zeros(SR, dtype=np.float32), _tone(220, 2.0)])
    assert speech_segments(wav, vad, threshold=0.5) == []
