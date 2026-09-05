"""検索器の統一インターフェース。

すべての検索器は search(query, top_k) -> [(chunk_index, score)] を実装する。
インターフェースを揃えることで、ハイブリッドもリランクも
「検索器を包む検索器」として合成できる。

    Hybrid( BM25, Dense )          → 2つの結果を RRF で統合
    MMRRerank( Hybrid(...) )       → 統合結果を多様性で並べ替え

RAG のパイプラインは本質的にこの合成である。
"""
from __future__ import annotations

import numpy as np

from .bm25 import BM25
from .chunker import Chunk
from .fusion import rrf, weighted_sum
from .store import VectorStore


class BM25Retriever:
    """語彙一致検索。固有名詞・型番・略語に強い。"""

    def __init__(self, chunks: list[Chunk], function_weight: float = 0.2):
        self.chunks = chunks
        self.function_weight = function_weight
        self.bm25 = BM25().fit([c.embed_text for c in chunks])
        self.name = f"bm25(w={function_weight})"

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        return self.bm25.search(query, top_k, function_weight=self.function_weight)


class DenseRetriever:
    """密ベクトル検索。言い換え・口語表現に強い。"""

    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.store = VectorStore.build(embedder, chunks)
        self.name = f"dense({embedder.name})"

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        return self.store.search(self.embedder.encode_query(query), top_k)

    def query_vec(self, query: str) -> np.ndarray:
        return self.embedder.encode_query(query)


class HybridRetriever:
    """複数の検索器の結果を統合する。

    candidate_k が要点。統合前に各検索器から多めに候補を取らないと、
    片方の5位以下にしか出ない正解が統合の土俵に上がらない。
    ハイブリッドの効果は「1段目の再現率」に上限を縛られる。
    """

    def __init__(
        self,
        retrievers: list,
        method: str = "rrf",
        candidate_k: int = 30,
        rrf_k: int = 60,
        weights: list[float] | None = None,
    ):
        self.retrievers = retrievers
        self.method = method
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.weights = weights
        self.chunks = retrievers[0].chunks
        names = "+".join(r.name.split("(")[0] for r in retrievers)
        self.name = f"hybrid[{names}/{method}]"

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        rankings = [r.search(query, self.candidate_k) for r in self.retrievers]
        if self.method == "rrf":
            fused = rrf(rankings, k=self.rrf_k, weights=self.weights)
        elif self.method == "weighted":
            fused = weighted_sum(rankings, weights=self.weights)
        else:
            raise ValueError(f"未知の統合方式: {self.method}")
        return fused[:top_k]


class RerankRetriever:
    """1段目の候補を2段目で並べ替える。

    candidate_k を top_k より十分大きく取ることが前提。
    candidate_k == top_k ではリランクする対象が無く、順番が入れ替わるだけになる。
    """

    def __init__(self, base, reranker, candidate_k: int = 20, dense: DenseRetriever | None = None):
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.dense = dense
        self.chunks = base.chunks
        self.name = f"{base.name}+{reranker.name}"

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        candidates = self.base.search(query, self.candidate_k)
        if hasattr(self.reranker, "rerank_texts"):  # クロスエンコーダ
            texts = [c.embed_text for c in self.chunks]
            return self.reranker.rerank_texts(query, candidates, texts, top_k)
        # MMR はベクトルを必要とするので dense 検索器から借りる
        if self.dense is None:
            raise ValueError("MMR には dense 検索器の指定が必要です")
        return self.reranker.rerank(
            self.dense.query_vec(query), self.dense.store.vectors, candidates, top_k
        )


def to_chunks(retriever, results: list[tuple[int, float]]) -> list[Chunk]:
    return [retriever.chunks[i] for i, _ in results]
