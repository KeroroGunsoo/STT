import numpy as np
import soundfile as sf

from speaker_transcriber.audio import load_audio, save_wav


def test_load_wav_resamples_to_16k_mono(tmp_path):
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 220 * t)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sr, subtype="PCM_16")

    wav = load_audio(path, sample_rate=16000)

    assert wav.dtype == np.float32
    assert wav.ndim == 1
    assert abs(len(wav) - 16000 * 1.5) < 160  # within 10 ms
    assert np.abs(wav).max() <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    sr = 16000
    wav = 0.5 * np.sin(2 * np.pi * 330 * np.arange(sr) / sr).astype(np.float32)
    path = save_wav(tmp_path / "tone.wav", wav, sr)
    back = load_audio(path, sample_rate=sr)
    assert len(back) == len(wav)
    assert np.corrcoef(wav, back)[0, 1] > 0.999


def test_load_rejects_file_without_audio(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(str(path), np.zeros(0, dtype=np.float32), 16000)
    try:
        load_audio(path)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty audio")
