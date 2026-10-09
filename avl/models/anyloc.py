"""AnyLoc — DINOv2 dense features aggregated with VLAD.

Follows the AnyLoc recipe (Keetha et al., RA-L 2023): dense descriptors from a
self-supervised foundation model, aggregated with VLAD over a vocabulary of cluster
centres fitted *without labels* on imagery from the target domain. Nothing is
trained for place recognition.

Three variants are exposed through the model registry:

  anyloc-lite  DINOv2 ViT-B/14, final-layer patch tokens    (fast control)
  anyloc-l     DINOv2 ViT-L/14, intermediate "value" facet   (the paper's method)
  anyloc-g     DINOv2 ViT-G/14, layer-31 "value" facet       (paper backbone)
  anyloc-sat   DINOv3 ViT-L/16 pre-trained on SAT-493M (Maxar 0.6 m ortho imagery),
               intermediate "value" facet — same recipe, a backbone that has seen
               the reference domain (overhead imagery) during self-supervision

The paper uses ViT-G/14 with the value facet of an intermediate layer; "lite" keeps
ViT-B and the final tokens so it fits a laptop. The vocabulary is fitted on the
reference map itself, which suits a UAV workflow: the map exists offline before the
flight, so the domain-specific vocabulary AnyLoc depends on is free to build.

k-means and PCA are implemented here in torch rather than pulled from scikit-learn,
so the deployment image keeps its current dependency set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T


LITE_BACKBONE = "vit_base_patch14_dinov2.lvd142m"
LARGE_BACKBONE = "vit_large_patch14_dinov2.lvd142m"
GIANT_BACKBONE = "vit_giant_patch14_dinov2.lvd142m"
SAT_BACKBONE = "hf_hub:timm/vit_large_patch16_dinov3.sat493m"
IMG_SIZE = 224
SAT_IMG_SIZE = 256  # native DINOv3 size; 16x16 patches, same token count as DINOv2 @224
NUM_CLUSTERS = 32
PCA_DIM = 512


def _kmeans(features: torch.Tensor, k: int, iterations: int = 25, seed: int = 0) -> torch.Tensor:
    """Lloyd's algorithm with k-means++ style seeding on L2-normalised features."""
    generator = torch.Generator().manual_seed(seed)
    n = features.shape[0]
    centres = features[torch.randperm(n, generator=generator)[:k]].clone()
    for _ in range(iterations):
        assignments = torch.cdist(features, centres).argmin(dim=1)
        for cluster in range(k):
            members = features[assignments == cluster]
            if members.numel():
                centres[cluster] = members.mean(dim=0)
    return centres


class AnyLocModel(nn.Module):
    """DINOv2 backbone + VLAD + PCA. `desc_facet` picks the dense descriptor:

    - "token": the final-layer patch tokens (the `anyloc-lite` behaviour).
    - "value": the per-token *value* vectors of the attention at `desc_layer`,
      concatenated across heads — the facet the AnyLoc paper uses.
    """

    def __init__(
        self,
        backbone_name: str = LITE_BACKBONE,
        desc_facet: str = "token",
        desc_layer: int | None = None,
        num_clusters: int = NUM_CLUSTERS,
        pca_dim: int = PCA_DIM,
        img_size: int = IMG_SIZE,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, img_size=img_size
        )
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.embed_dim = int(self.backbone.embed_dim)
        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self.desc_facet = desc_facet

        n_blocks = len(self.backbone.blocks)
        layer = (n_blocks - 9) if desc_layer is None else desc_layer
        self.desc_layer = max(0, min(layer, n_blocks - 1))

        self.register_buffer("vocabulary", torch.zeros(num_clusters, self.embed_dim))
        self.register_buffer("pca_mean", torch.zeros(num_clusters * self.embed_dim))
        self.register_buffer("pca_components", torch.zeros(0, num_clusters * self.embed_dim))
        self.fitted = False

        self._captured: torch.Tensor | None = None
        if desc_facet == "value":
            self.backbone.blocks[self.desc_layer].attn.qkv.register_forward_hook(
                self._grab_qkv
            )

    def _grab_qkv(self, _module, _inputs, output: torch.Tensor) -> None:
        self._captured = output

    @property
    def output_dim(self) -> int:
        if self.pca_components.shape[0]:
            return int(self.pca_components.shape[0])
        return self.num_clusters * self.embed_dim

    @torch.inference_mode()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if self.desc_facet == "value":
            self._captured = None
            self.backbone.forward_features(images)
            qkv = self._captured
            if qkv is None:
                raise RuntimeError("AnyLoc value-facet hook did not fire")
            batch, tokens_n, _ = qkv.shape
            heads = self.backbone.blocks[self.desc_layer].attn.num_heads
            head_dim = self.embed_dim // heads
            # qkv: (B, N, 3*embed_dim) -> take the value block, re-join heads
            value = qkv.reshape(batch, tokens_n, 3, heads, head_dim)[:, :, 2]
            tokens = value.reshape(batch, tokens_n, self.embed_dim)[:, self.num_prefix_tokens:]
        else:
            tokens = self.backbone.forward_features(images)[:, self.num_prefix_tokens:]
        return torch.nn.functional.normalize(tokens, p=2, dim=-1)

    def vlad(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, P, D) -> (B, K*D) intra-normalised VLAD."""
        batch, _, dim = tokens.shape
        assignments = torch.cdist(tokens, self.vocabulary.expand(batch, -1, -1)).argmin(dim=-1)
        descriptors = tokens.new_zeros(batch, self.num_clusters, dim)
        for cluster in range(self.num_clusters):
            mask = (assignments == cluster).unsqueeze(-1)
            residual = (tokens - self.vocabulary[cluster]) * mask
            descriptors[:, cluster] = residual.sum(dim=1)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=-1)  # intra-normalise
        return torch.nn.functional.normalize(descriptors.flatten(1), p=2, dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError(
                "AnyLoc vocabulary is not fitted. Call fit_vocabulary() on the reference map first."
            )
        descriptors = self.vlad(self.patch_tokens(images))
        if self.pca_components.shape[0]:
            descriptors = (descriptors - self.pca_mean) @ self.pca_components.T
            descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=-1)
        return descriptors

    # ------------------------------------------------------------- fitting --
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vocabulary=self.vocabulary.cpu().numpy(),
            pca_mean=self.pca_mean.cpu().numpy(),
            pca_components=self.pca_components.cpu().numpy(),
        )

    def load(self, path: Path) -> None:
        data = np.load(path)
        device = self.vocabulary.device
        self.vocabulary = torch.from_numpy(data["vocabulary"]).to(device)
        self.pca_mean = torch.from_numpy(data["pca_mean"]).to(device)
        self.pca_components = torch.from_numpy(data["pca_components"]).to(device)
        self.fitted = True


# Backwards-compatible name — the class used to be AnyLoc-lite specific.
AnyLocLite = AnyLocModel


@torch.inference_mode()
def fit_vocabulary(
    model: AnyLocModel,
    image_paths: list[str],
    transform,
    device: torch.device,
    max_images: int = 200,
    pca_dim: int = PCA_DIM,
    batch_size: int = 8,
    verbose: bool = True,
) -> None:
    """Fit the VLAD vocabulary and PCA basis on images from the target map."""
    from PIL import Image

    embed_dim = model.embed_dim
    rng = np.random.default_rng(0)
    if len(image_paths) > max_images:
        sample = [image_paths[i] for i in rng.choice(len(image_paths), max_images, replace=False)]
    else:
        sample = list(image_paths)

    # --- pass 1: dense descriptors -> vocabulary ---
    token_bank = []
    for start in range(0, len(sample), batch_size):
        batch = torch.stack(
            [transform(Image.open(p).convert("RGB")) for p in sample[start : start + batch_size]]
        ).to(device)
        tokens = model.patch_tokens(batch).reshape(-1, embed_dim)
        # subsample patches to keep k-means tractable
        keep = rng.choice(tokens.shape[0], min(512, tokens.shape[0]), replace=False)
        token_bank.append(tokens[keep].cpu())
    tokens = torch.cat(token_bank)
    if verbose:
        print(f"[anyloc] fitting {model.num_clusters} clusters on {tokens.shape[0]} descriptors")
    model.vocabulary = _kmeans(tokens, model.num_clusters).to(device)
    model.fitted = True
    model.pca_components = model.pca_components.new_zeros(0, model.num_clusters * embed_dim)

    # --- pass 2: VLAD of the sample (with rotations, for more PCA samples) -> PCA ---
    vlads = []
    for start in range(0, len(sample), batch_size):
        images = [Image.open(p).convert("RGB") for p in sample[start : start + batch_size]]
        for rotation in (0, 90, 180, 270):
            batch = torch.stack([transform(im.rotate(rotation)) for im in images]).to(device)
            vlads.append(model.vlad(model.patch_tokens(batch)).cpu())
    vlads = torch.cat(vlads)

    components = min(pca_dim, vlads.shape[0], vlads.shape[1])
    mean = vlads.mean(dim=0)
    centred = vlads - mean
    # economy SVD on (N, D); N is small, so this is cheap
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    model.pca_mean = mean.to(device)
    model.pca_components = vh[:components].to(device)
    if verbose:
        print(f"[anyloc] PCA {vlads.shape[1]} -> {components} dims from {vlads.shape[0]} samples")


def load_anyloc(device: torch.device) -> AnyLocModel:
    """anyloc-lite — DINOv2 ViT-B/14, final patch tokens."""
    return AnyLocModel(LITE_BACKBONE, desc_facet="token").to(device).eval()


def load_anyloc_l(device: torch.device) -> AnyLocModel:
    """anyloc-l — DINOv2 ViT-L/14, intermediate value facet (the AnyLoc method)."""
    return AnyLocModel(LARGE_BACKBONE, desc_facet="value").to(device).eval()


def load_anyloc_g(device: torch.device) -> AnyLocModel:
    """anyloc-g — DINOv2 ViT-G/14, layer-31 value facet (the paper's backbone)."""
    return AnyLocModel(GIANT_BACKBONE, desc_facet="value", desc_layer=31).to(device).eval()


def load_anyloc_sat(device: torch.device) -> AnyLocModel:
    """anyloc-sat — DINOv3 ViT-L/16 (SAT-493M), intermediate value facet.

    Layer 15 of 24, as for anyloc-l. DINOv3 carries 4 register tokens after the
    CLS token; `num_prefix_tokens` (5) strips all of them before VLAD.
    """
    return AnyLocModel(SAT_BACKBONE, desc_facet="value", img_size=SAT_IMG_SIZE).to(device).eval()


_cfg = timm.get_pretrained_cfg(LITE_BACKBONE)  # DINOv2 lvd142m variants share the normalisation

ANYLOC_TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=_cfg.mean, std=_cfg.std),
    ]
)

# SAT-493M statistics (from the model's pretrained cfg), not ImageNet's
ANYLOC_SAT_TRANSFORM = T.Compose(
    [
        T.Resize((SAT_IMG_SIZE, SAT_IMG_SIZE), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
    ]
)
