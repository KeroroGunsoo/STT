import numpy as np

from speaker_transcriber.diarization import build_windows, cluster_embeddings, merge_turns, relabel


def _blobs():
    rng = np.random.default_rng(0)
    a = np.tile([1.0, 0.0], (8, 1)) + rng.normal(0, 0.02, (8, 2))
    b = np.tile([0.0, 1.0], (8, 1)) + rng.normal(0, 0.02, (8, 2))
    emb = np.vstack([a, b])
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def test_cluster_embeddings_threshold_finds_two_speakers():
    labels = cluster_embeddings(_blobs(), num_speakers=None, threshold=0.30)
    assert len(set(labels)) == 2
    assert labels[0] == labels[7]
    assert labels[8] == labels[15]
    assert labels[0] != labels[8]


def test_cluster_embeddings_fixed_count():
    labels = cluster_embeddings(_blobs(), num_speakers=2)
    assert len(set(labels)) == 2


def test_cluster_embeddings_single_window():
    labels = cluster_embeddings(np.array([[1.0, 0.0]]))
    assert list(labels) == [0]


def test_build_windows_covers_tail():
    wins = build_windows([(0.0, 10.0)], window_s=3.0, hop_s=1.5)
    assert wins[0] == (0.0, 3.0)
    assert wins[-1][1] == 10.0
    assert all(e - s <= 3.0 + 1e-9 for s, e in wins)
    # coverage: no gap larger than hop between consecutive windows
    for (s1, e1), (s2, e2) in zip(wins, wins[1:]):
        assert s2 - e1 <= 1.5 + 1e-9 or s2 <= e1 + 1e-9


def test_merge_and_relabel():
    windows = [(0.0, 2.0), (1.5, 3.5), (4.0, 6.0), (6.5, 8.0)]
    labels = np.array([0, 0, 1, 1])
    turns = merge_turns(windows, labels, merge_gap_s=0.75)
    named = relabel(turns)
    assert named == [(0.0, 3.5, "SPEAKER_00"), (4.0, 8.0, "SPEAKER_01")]
