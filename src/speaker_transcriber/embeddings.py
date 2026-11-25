from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def prepare_ecapa_dir(models_dir: str | Path) -> Path:
    """Make the ECAPA checkpoint available in our own cache dir.

    Downloads via huggingface_hub and copies with plain file copies (no
    symlinks/hard links): speechbrain's default LocalStrategy.SYMLINK fails
    on Windows with WinError 1314 (privilege not held), so we never let it
    link from the HF cache — we load from this local directory directly.
    """
    from huggingface_hub import snapshot_download

    dest = Path(models_dir) / "ecapa-voxceleb"
    if not (dest / "hyperparams.yaml").exists():
        snapshot = Path(snapshot_download(ECAPA_SOURCE))
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".no_exist"))
    return dest


class SpeakerEmbedder:
    """ECAPA-TDNN speaker embeddings (speechbrain, VoxCeleb-pretrained), L2-normalized."""

    def __init__(self, models_dir: str | Path, device: str = "cpu"):
        import torch
        from speechbrain.inference import EncoderClassifier

        self.torch = torch
        self.device = device
        local = prepare_ecapa_dir(models_dir)
        self.clf = EncoderClassifier.from_hparams(
            source=str(local),
            savedir=str(local),
            run_opts={"device": device},
        )

    def embed(self, wavs: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Embed a list of 16 kHz mono waveforms; returns an (n, dim) L2-normalized array."""
        torch = self.torch
        embs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(wavs), batch_size):
                batch = [np.asarray(w, dtype=np.float32) for w in wavs[i : i + batch_size]]
                maxlen = max(1, max(len(w) for w in batch))
                x = torch.zeros(len(batch), maxlen, dtype=torch.float32)
                lens = torch.zeros(len(batch), dtype=torch.float32)
                for b, w in enumerate(batch):
                    x[b, : len(w)] = torch.from_numpy(w)
                    lens[b] = len(w) / maxlen
                x, lens = x.to(self.device), lens.to(self.device)
                e = self.clf.encode_batch(x, wav_lens=lens)
                e = e.reshape(e.shape[0], -1).cpu().numpy()
                embs.append(e)
        emb = np.concatenate(embs, axis=0).astype(np.float32) if embs else np.zeros((0, 192), np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms
