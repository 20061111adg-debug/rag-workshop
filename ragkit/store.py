"""ベクトルストア。

Chroma / FAISS / pgvector が何をしているかを 30 行で示す。
本質は「正規化済みベクトルの行列積を取り、上位 k 件を返す」だけ。

    similarity = Q · Dᵀ        (正規化済みなら内積 = cosine 類似度)

商用ベクトルDBが提供している差分は、この計算そのものではなく:
  - 近似最近傍探索 (HNSW/IVF)  : 全件走査を避け、100万件規模で ms 応答
  - 永続化とメタデータフィルタ  : 属性での絞り込み、更新・削除
  - 分散・レプリケーション

**数万チャンク程度なら numpy の全件走査で十分に速い。**
「まず全件走査、遅くなったらANN」が正しい順序で、
最初からベクトルDBを立てる必要はない、というのがこの実装の主張。
"""
from __future__ import annotations

import numpy as np

from ._numpy_compat import matmul


class VectorStore:
    def __init__(self, vectors: np.ndarray, payloads: list):
        assert len(vectors) == len(payloads), "ベクトル数とチャンク数が一致しません"
        self.vectors = vectors.astype(np.float32)
        self.payloads = payloads

    @classmethod
    def build(cls, embedder, chunks: list) -> "VectorStore":
        texts = [c.embed_text for c in chunks]
        embedder.fit(texts)
        return cls(embedder.encode_documents(texts), chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        scores = matmul(self.vectors, query_vec)  # (n,) 全件との内積
        # argpartition で上位k件だけ部分選択する（全件ソートより速い）
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def __len__(self) -> int:
        return len(self.payloads)

    @property
    def memory_mb(self) -> float:
        return self.vectors.nbytes / 1024 / 1024
