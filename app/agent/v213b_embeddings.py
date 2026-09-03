"""V2.13B configurable embedding providers (framework-neutral)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from typing import Any

from app.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stable_hash_index(value: str, dimension: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % dimension


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0:
        return vector
    return [v / norm for v in vector]


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def identity(self) -> dict[str, Any]:
        return {
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension,
        }


class FeatureHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding for reproducible experiments (no network)."""

    def __init__(self, *, dimension: int = 128, model_name: str = "feature-hash-v1") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            for n in (1, 2, 3):
                if len(token) < n:
                    continue
                for i in range(len(token) - n + 1):
                    gram = token[i : i + n]
                    idx = _stable_hash_index(gram, self._dimension)
                    vector[idx] += 1.0
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        for gram in bigrams:
            idx = _stable_hash_index(gram, self._dimension)
            vector[idx] += 0.5
        return _normalize(vector)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Optional HTTP embedding provider (OpenAI-compatible /embeddings endpoint)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        timeout: float = 30.0,
    ) -> None:
        import httpx

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        response = self._client.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        return list(data["data"][0]["embedding"])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        rows = sorted(data["data"], key=lambda row: row["index"])
        return [list(row["embedding"]) for row in rows]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or Settings()
    provider = getattr(settings, "v213b_embedding_provider", "feature_hash")
    if provider == "openai" and settings.llm_api_key:
        return OpenAIEmbeddingProvider(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=getattr(settings, "v213b_embedding_model", "text-embedding-3-small"),
            dimension=int(getattr(settings, "v213b_embedding_dimension", 1536)),
            timeout=settings.llm_timeout_seconds,
        )
    return FeatureHashEmbeddingProvider(
        dimension=int(getattr(settings, "v213b_embedding_dimension", 128)),
        model_name=getattr(settings, "v213b_embedding_model", "feature-hash-v1"),
    )


def embedding_identity(provider: EmbeddingProvider) -> str:
    return json.dumps(provider.identity(), sort_keys=True)


__all__ = [
    "EmbeddingProvider",
    "FeatureHashEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "build_embedding_provider",
    "cosine_similarity",
    "embedding_identity",
]
