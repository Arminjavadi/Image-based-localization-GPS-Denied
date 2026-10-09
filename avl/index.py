from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from avl.config import AVLConfig
from avl.metadata import ReferenceRecord


@dataclass
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray


#: Config fields written next to the index, so a loaded index knows how it was built
#: and which recipe (avl.pipeline) its queries must follow.
SAVED_FIELDS = (
    "model",
    "descriptor_dim",
    "cosplace_backbone",
    "query_rotations",
    "index_type",
    "hnsw_m",
    "hnsw_ef_construction",
    "hnsw_ef_search",
    "ivfpq_nlist",
    "ivfpq_m",
    "ivfpq_nbits",
    "score_threshold",
    "fusion_method",
    "fusion_softmax_temp",
    "fusion_cluster_radius_m",
    "heading_mode",
    "center_mode",
    "center_warmup",
    "scale_from_agl",
    "camera_k",
    "tile_m",
    "agl_gate",
    "min_agl_m",
    "dem_dir",
    "prior_k",
    "query_crop",
)


class HashIndex:
    """
    FAISS-backed approximate nearest-neighbor index.

    - HNSW: best recall/latency for medium-scale databases.
    - IVF-PQ: product-quantized hashing for large-scale compressed search.
    - FLAT: exact brute-force baseline.
    """

    def __init__(self, config: AVLConfig) -> None:
        self.config = config
        self.index: faiss.Index | None = None
        self.records: list[ReferenceRecord] = []
        #: member descriptor widths of an ensemble (avl.ensemble); None for one encoder
        self.blocks: list[int] | None = None

    @property
    def size(self) -> int:
        return len(self.records)

    def build(self, descriptors: np.ndarray, records: list[ReferenceRecord]) -> None:
        if len(descriptors) != len(records):
            raise ValueError("Descriptor count must match metadata record count")
        if len(descriptors) == 0:
            raise ValueError("Cannot build an empty index")

        descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
        faiss.normalize_L2(descriptors)
        dim = descriptors.shape[1]
        n = descriptors.shape[0]

        if self.config.index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif self.config.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
            self.index.hnsw.efSearch = self.config.hnsw_ef_search
        elif self.config.index_type == "ivfpq":
            nlist = min(self.config.ivfpq_nlist, max(1, n // 39))
            m = self.config.ivfpq_m
            if dim % m != 0:
                raise ValueError(f"Descriptor dim {dim} must be divisible by ivfpq_m={m}")
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, self.config.ivfpq_nbits, faiss.METRIC_INNER_PRODUCT)
            train_count = min(n, self.config.ivfpq_train_samples)
            train_vectors = descriptors if train_count == n else descriptors[
                np.random.default_rng(42).choice(n, size=train_count, replace=False)
            ]
            self.index.train(train_vectors)
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")

        self.index.add(descriptors)
        self.records = records

    def search(
        self, query: np.ndarray, top_k: int, allowed: np.ndarray | None = None
    ) -> SearchResult:
        """Nearest tiles to ``query``. ``allowed`` (tile indices) restricts the search
        to a window around the navigation prior (avl.geo.search_window)."""
        if self.index is None:
            raise RuntimeError("Index is not built or loaded")

        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)

        if self.config.index_type == "hnsw":
            self.index.hnsw.efSearch = max(self.config.hnsw_ef_search, top_k * 4)

        k = min(top_k, self.size)
        if allowed is None:
            scores, indices = self.index.search(query, k)
        else:
            allowed = np.ascontiguousarray(allowed, dtype=np.int64)
            k = min(k, len(allowed))
            if k == 0:
                return SearchResult(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
            selector = faiss.IDSelectorBatch(allowed)
            if self.config.index_type == "hnsw":
                params = faiss.SearchParametersHNSW(sel=selector, efSearch=self.index.hnsw.efSearch)
            elif self.config.index_type == "ivfpq":
                params = faiss.SearchParametersIVF(sel=selector, nprobe=self.index.nprobe)
            else:
                params = faiss.SearchParameters(sel=selector)
            scores, indices = self.index.search(query, k, params=params)
        valid = indices[0] >= 0
        return SearchResult(indices=indices[0][valid], scores=scores[0][valid])

    def search_orientations(
        self, queries: np.ndarray, top_k: int, allowed: np.ndarray | None = None
    ) -> SearchResult:
        queries = np.asarray(queries)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        best_by_location: dict[str, tuple[int, float]] = {}
        candidate_k = min(max(top_k * 8, top_k), self.size)
        for query in queries:
            result = self.search(query, candidate_k, allowed)
            for idx, score in zip(result.indices, result.scores):
                index = int(idx)
                record = self.records[index]
                location_id = record.image_id or f"index:{index}"
                previous = best_by_location.get(location_id)
                if previous is None or float(score) > previous[1]:
                    best_by_location[location_id] = (index, float(score))

        ranked = sorted(
            best_by_location.values(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return SearchResult(
            indices=np.asarray([index for index, _ in ranked], dtype=np.int64),
            scores=np.asarray([score for _, score in ranked], dtype=np.float32),
        )

    def save(self, directory: Path) -> None:
        if self.index is None:
            raise RuntimeError("Index is not built")

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))

        config = {name: getattr(self.config, name) for name in SAVED_FIELDS}
        config["dem_dir"] = str(config["dem_dir"])
        metadata = {
            "config": config,
            "blocks": self.blocks,
            "records": [asdict(record) for record in self.records],
        }
        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: Path, config: AVLConfig | None = None) -> "HashIndex":
        with open(directory / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        saved_config = metadata["config"]
        if config is None:
            # Indexes written before a field existed fall back to its default,
            # which is the behaviour they were built with.
            config = AVLConfig(
                **{name: saved_config[name] for name in SAVED_FIELDS if name in saved_config}
            )

        instance = cls(config)
        instance.index = faiss.read_index(str(directory / "index.faiss"))
        instance.records = [ReferenceRecord(**record) for record in metadata["records"]]
        instance.blocks = metadata.get("blocks")
        return instance
