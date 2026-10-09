from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from avl.config import AVLConfig
from avl.models.boq import BOQ_TRANSFORM, DESCRIPTOR_DIM as BOQ_DIM, load_boq
from avl.models.cosplace import COSPLACE_TRANSFORM, load_cosplace
from avl.models.denseuav_vit import DENSEUAV_TRANSFORM, load_denseuav_vit
from avl.models.anyloc import (
    ANYLOC_SAT_TRANSFORM,
    ANYLOC_TRANSFORM,
    load_anyloc,
    load_anyloc_l,
    load_anyloc_g,
    load_anyloc_sat,
)
from avl.models.dinov2 import DINOV2_TRANSFORM, DINOV2_DIM, load_dinov2
from avl.models.dinov3 import BACKBONES as DINOV3_BACKBONES, DINOV3_TRANSFORM, load_dinov3
from avl.models.dinov2_ft import DINOV2_FT_TRANSFORM, load_dinov2_ft
from avl.models.eigenplaces import (
    EIGENPLACES_TRANSFORM,
    DESCRIPTOR_DIM as EIGENPLACES_DIM,
    load_eigenplaces,
)
from avl.models.game4loc import GAME4LOC_TRANSFORM, DESCRIPTOR_DIM as GAME4LOC_DIM, load_game4loc
from avl.models.megaloc import MEGALOC_TRANSFORM, DESCRIPTOR_DIM as MEGALOC_DIM, load_megaloc
from avl.models.infogeo import CHECKPOINTS as INFOGEO_CHECKPOINTS, DESCRIPTOR_DIM as INFOGEO_DIM, INFOGEO_TRANSFORM, load_infogeo
from avl.models.mixvpr import MIXVPR_TRANSFORM, load_mixvpr
from avl.models.sample4geo import (
    SAMPLE4GEO_TRANSFORM,
    DESCRIPTOR_DIM as SAMPLE4GEO_DIM,
    load_sample4geo,
)


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


class _QueryRotationDataset(Dataset):
    def __init__(self, image_path: str | Path, transform, rotations: int) -> None:
        with Image.open(image_path) as img:
            self.image = img.convert("RGB")
        self.transform = transform
        self.rotations = rotations

    def __len__(self) -> int:
        return self.rotations

    def __getitem__(self, index: int):
        if self.rotations == 1:
            image = self.image
        else:
            image = self.image.rotate(index * 90, expand=False)
        return self.transform(image), index


class VPREncoder:
    """One visual place-recognition encoder (see avl.ensemble for several at once)."""

    def __init__(self, config: AVLConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        if config.model == "denseuav-vit":
            if config.descriptor_dim != 512:
                raise ValueError("DenseUAV ViT descriptor dimension must be 512")
            self.model = load_denseuav_vit(self.device)
            self.transform = DENSEUAV_TRANSFORM
        elif config.model == "game4loc":
            if config.descriptor_dim != GAME4LOC_DIM:
                raise ValueError(f"Game4Loc descriptor dimension must be {GAME4LOC_DIM}")
            self.model = load_game4loc(self.device)
            self.transform = GAME4LOC_TRANSFORM
        elif config.model == "sample4geo":
            if config.descriptor_dim != SAMPLE4GEO_DIM:
                raise ValueError(f"Sample4Geo descriptor dimension must be {SAMPLE4GEO_DIM}")
            self.model = load_sample4geo(self.device)
            self.transform = SAMPLE4GEO_TRANSFORM
        elif config.model == "eigenplaces":
            if config.descriptor_dim != EIGENPLACES_DIM:
                raise ValueError(f"EigenPlaces descriptor dimension must be {EIGENPLACES_DIM}")
            self.model = load_eigenplaces(self.device)
            self.transform = EIGENPLACES_TRANSFORM
        elif config.model == "anyloc-lite":
            # Descriptor dimension is decided by the PCA fit on the target map.
            self.model = load_anyloc(self.device)
            self.transform = ANYLOC_TRANSFORM
        elif config.model == "anyloc-l":
            self.model = load_anyloc_l(self.device)
            self.transform = ANYLOC_TRANSFORM
        elif config.model == "anyloc-g":
            self.model = load_anyloc_g(self.device)
            self.transform = ANYLOC_TRANSFORM
        elif config.model == "anyloc-sat":
            self.model = load_anyloc_sat(self.device)
            self.transform = ANYLOC_SAT_TRANSFORM
        elif config.model == "megaloc":
            if config.descriptor_dim != MEGALOC_DIM:
                raise ValueError(f"MegaLoc descriptor dimension must be {MEGALOC_DIM}")
            self.model = load_megaloc(self.device)
            self.transform = MEGALOC_TRANSFORM
        elif config.model == "boq":
            if config.descriptor_dim != BOQ_DIM:
                raise ValueError(f"BoQ descriptor dimension must be {BOQ_DIM}")
            self.model = load_boq(self.device)
            self.transform = BOQ_TRANSFORM
        elif config.model in DINOV3_BACKBONES:
            dim = DINOV3_BACKBONES[config.model][1]
            if config.descriptor_dim != dim:
                raise ValueError(f"{config.model} descriptor dimension must be {dim}")
            self.model = load_dinov3(config.model, self.device)
            self.transform = DINOV3_TRANSFORM
        elif config.model in INFOGEO_CHECKPOINTS:
            if config.descriptor_dim != INFOGEO_DIM:
                raise ValueError(f"InfoGeo descriptor dimension must be {INFOGEO_DIM}")
            self.model = load_infogeo(config.model, self.device)
            self.transform = INFOGEO_TRANSFORM
        elif config.model == "dinov2":
            if config.descriptor_dim != DINOV2_DIM:
                raise ValueError(f"DINOv2 descriptor dimension must be {DINOV2_DIM}")
            self.model = load_dinov2(self.device)
            self.transform = DINOV2_TRANSFORM
        elif config.model == "dinov2-ft":
            self.model = load_dinov2_ft(config.head_weights, self.device)
            if config.descriptor_dim != self.model.output_dim:
                raise ValueError(f"dinov2-ft descriptor dimension must be {self.model.output_dim}")
            self.transform = DINOV2_FT_TRANSFORM
        elif config.model == "mixvpr":
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
    def encode_images(self, images: Sequence[Image.Image], batch_size: int | None = None) -> np.ndarray:
        """L2-normalised descriptors of in-memory images (query views after rotation / crop)."""
        size = batch_size or self.config.batch_size
        out = []
        for start in range(0, len(images), size):
            batch = torch.stack([self.transform(im) for im in images[start : start + size]])
            feats = self.model(batch.to(self.device))
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    @torch.inference_mode()
    def encode_image(self, image_path: str | Path) -> np.ndarray:
        return self.encode_paths([image_path], show_progress=False)[0]

    @torch.inference_mode()
    def encode_query(self, image_path: str | Path) -> np.ndarray:
        dataset = _QueryRotationDataset(
            image_path,
            self.transform,
            rotations=self.config.query_rotations,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.query_rotations,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        descriptors = np.zeros(
            (self.config.query_rotations, self.config.descriptor_dim),
            dtype=np.float32,
        )
        for batch, indices in loader:
            batch = batch.to(self.device, non_blocking=True)
            outputs = self.model(batch)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            descriptors[indices.numpy()] = outputs.cpu().numpy()
        return descriptors
