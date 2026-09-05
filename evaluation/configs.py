"""比較対象となる RAG 構成の定義。

Step1 の素朴な実装から段階的に部品を足していき、
「どの一手が何ポイント効いたか」を分解できるように並べている。
埋め込みモデルとチャンク集合はコストが高いので使い回す。
"""
from __future__ import annotations

from ragkit.chunker import STRATEGIES
from ragkit.embedder import E5Embedder, LsaEmbedder, TfidfEmbedder
from ragkit.rerank import MMRReranker
from ragkit.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
)


def build(docs, include_cross_encoder: bool = False) -> dict:
    """{構成名: 検索器} を返す。"""
    chunks = {name: fn(docs) for name, fn in STRATEGIES.items()}
    cfg: dict = {}

    # ── Step 1: 素朴な RAG ────────────────────────────────
    # 固定長分割 + TF-IDF。多くの入門記事の実装がこれに相当する
    cfg["S1 固定長 + TF-IDF"] = DenseRetriever(chunks["fixed"], TfidfEmbedder())

    # ── Step 2: チャンク戦略だけを変える（埋め込みは E5 で固定）─────
    e5 = E5Embedder()  # モデルは全構成で共有する
    for name in ("fixed", "fixed_overlap", "heading", "heading_context"):
        cfg[f"S2 {name} + E5"] = DenseRetriever(chunks[name], e5)

    # ── Step 3: 検索方式の比較（チャンクは heading_context で固定）──
    best = chunks["heading_context"]
    cfg["S3 BM25 単独(素朴 w=1.0)"] = BM25Retriever(best, function_weight=1.0)
    cfg["S3 BM25 単独(機能語減衰)"] = BM25Retriever(best, function_weight=0.2)
    cfg["S3 Dense 単独(LSA)"] = DenseRetriever(best, LsaEmbedder(128))
    cfg["S3 Dense 単独(E5)"] = DenseRetriever(best, e5)

    bm = cfg["S3 BM25 単独(機能語減衰)"]
    dn = cfg["S3 Dense 単独(E5)"]
    cfg["S3 Hybrid RRF"] = HybridRetriever([bm, dn], method="rrf")
    cfg["S3 Hybrid 加重和"] = HybridRetriever([bm, dn], method="weighted")
    cfg["S3 Hybrid RRF(dense重視2:1)"] = HybridRetriever(
        [bm, dn], method="rrf", weights=[1.0, 2.0]
    )

    # ── Step 4: リランク ─────────────────────────────────
    cfg["S4 Hybrid + MMR(λ=0.7)"] = RerankRetriever(
        cfg["S3 Hybrid RRF"], MMRReranker(0.7), candidate_k=20, dense=dn
    )
    cfg["S4 Hybrid + MMR(λ=0.5)"] = RerankRetriever(
        cfg["S3 Hybrid RRF"], MMRReranker(0.5), candidate_k=20, dense=dn
    )
    if include_cross_encoder:
        from ragkit.rerank import CrossEncoderReranker

        ce = CrossEncoderReranker()
        cfg["S4 Hybrid + CrossEncoder"] = RerankRetriever(
            cfg["S3 Hybrid RRF"], ce, candidate_k=20
        )
        cfg["S4 Dense + CrossEncoder"] = RerankRetriever(dn, ce, candidate_k=20)
    return cfg
