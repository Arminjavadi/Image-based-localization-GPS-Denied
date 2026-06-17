from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from avl.config import AVLConfig
from avl.models.cosplace import COSPLACE_TRANSFORM, load_cosplace
from avl.models.mixvpr import MIXVPR_TRANSFORM, load_mixvpr


class _ImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], transform) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        return self.transform(image), index


class VPREncoder:
    """State-of-the-art visual descriptor encoder (MixVPR or CosPlace)."""

    def __init__(self, config: AVLConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        if config.model == "mixvpr":
            self.model = load_mixvpr(config.descriptor_dim, config.cache_dir, self.device)
            self.transform = MIXVPR_TRANSFORM
        elif config.model == "cosplace":
            self.model = load_cosplace(config.cosplace_backbone, config.descriptor_dim, self.device)
            self.transform = COSPLACE_TRANSFORM
        else:
            raise ValueError(f"Unsupported model: {config.model}")

    @torch.inference_mode()
    def encode_paths(self, image_paths: Sequence[str | Path], show_progress: bool = True) -> np.ndarray:
        paths = [str(p) for p in image_paths]
        dataset = _ImageDataset(paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=min(4, max(0, len(paths) // 4)),
            pin_memory=self.device.type == "cuda",
        )

        descriptors = np.zeros((len(paths), self.config.descriptor_dim), dtype=np.float32)
        iterator: Iterable = loader
        if show_progress:
            iterator = tqdm(loader, desc="Encoding images", unit="batch")

        for batch, indices in iterator:
            batch = batch.to(self.device, non_blocking=True)
            outputs = self.model(batch)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            descriptors[indices.numpy()] = outputs.cpu().numpy()

        return descriptors

    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        return self.encode_paths([image_path], show_progress=False)[0]
