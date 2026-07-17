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

    def search(self, query: np.ndarray, top_k: int) -> SearchResult:
        if self.index is None:
            raise RuntimeError("Index is not built or loaded")

        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)

        if self.config.index_type == "hnsw":
            self.index.hnsw.efSearch = max(self.config.hnsw_ef_search, top_k * 4)

        scores, indices = self.index.search(query, min(top_k, self.size))
        valid = indices[0] >= 0
        return SearchResult(indices=indices[0][valid], scores=scores[0][valid])

    def search_orientations(self, queries: np.ndarray, top_k: int) -> SearchResult:
        queries = np.asarray(queries)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        best_by_location: dict[str, tuple[int, float]] = {}
        candidate_k = min(max(top_k * 8, top_k), self.size)
        for query in queries:
            result = self.search(query, candidate_k)
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

        metadata = {
            "config": {
                "model": self.config.model,
                "descriptor_dim": self.config.descriptor_dim,
                "cosplace_backbone": self.config.cosplace_backbone,
                "query_rotations": self.config.query_rotations,
                "index_type": self.config.index_type,
                "hnsw_m": self.config.hnsw_m,
                "hnsw_ef_construction": self.config.hnsw_ef_construction,
                "hnsw_ef_search": self.config.hnsw_ef_search,
                "ivfpq_nlist": self.config.ivfpq_nlist,
                "ivfpq_m": self.config.ivfpq_m,
                "ivfpq_nbits": self.config.ivfpq_nbits,
            },
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
            config = AVLConfig(
                model=saved_config["model"],
                descriptor_dim=saved_config["descriptor_dim"],
                cosplace_backbone=saved_config.get("cosplace_backbone", "ResNet101"),
                query_rotations=saved_config.get("query_rotations", 4),
                index_type=saved_config["index_type"],
                hnsw_m=saved_config.get("hnsw_m", 32),
                hnsw_ef_construction=saved_config.get("hnsw_ef_construction", 200),
                hnsw_ef_search=saved_config.get("hnsw_ef_search", 128),
                ivfpq_nlist=saved_config.get("ivfpq_nlist", 4096),
                ivfpq_m=saved_config.get("ivfpq_m", 64),
                ivfpq_nbits=saved_config.get("ivfpq_nbits", 8),
            )

        instance = cls(config)
        instance.index = faiss.read_index(str(directory / "index.faiss"))
        instance.records = [ReferenceRecord(**record) for record in metadata["records"]]
        return instance
