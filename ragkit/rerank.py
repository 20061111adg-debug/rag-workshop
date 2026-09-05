"""リランキング（再順位付け）。

二段構えにするのが定石。
  1段目 (検索)  : 数万件から高速に候補50件を絞る。精度より再現率を優先
  2段目 (リランク): 50件をじっくり評価して上位5件に絞る。精度を優先

なぜ分けるのか。1段目に使うバイエンコーダはクエリと文書を独立にベクトル化するため、
文書側を事前計算でき高速だが、両者の相互作用を見られない。
クロスエンコーダは [クエリ; 文書] を連結して1回推論するため精度は高いが、
検索のたびに候補数だけ推論が要る。全件には使えないので候補絞り込みの後に置く。

ここでは対照的な2種を実装する。
  MMRReranker          : モデル不要。冗長性を除き多様性を確保する
  CrossEncoderReranker : モデル必要。関連性そのものを測り直す
"""
from __future__ import annotations

import numpy as np

from ._numpy_compat import matmul


class MMRReranker:
    """Maximal Marginal Relevance。関連性と多様性のトレードオフを取る。

        MMR = argmax_d [ λ · sim(q, d) − (1−λ) · max_{s∈選択済} sim(d, s) ]

    素朴な top-k は「ほぼ同じ内容のチャンク」で埋まりやすい。
    特に見出し分割 + オーバーラップを併用すると近傍チャンクが並んで入り、
    LLM に渡すコンテキストの実質的な情報量が落ちる。

    複数文書にまたがる質問（例:「セキュリティと就業ルールの両面から」）では、
    1位の文書に似たものを並べるより、別の観点を混ぜたほうが答えられる。
    λ=1.0 で通常の関連度順、λ を下げるほど多様性を優先する。
    """

    name = "mmr"

    def __init__(self, lambda_: float = 0.7):
        self.lambda_ = lambda_

    def rerank(
        self, query_vec: np.ndarray, doc_vecs: np.ndarray, candidates: list[tuple[int, float]], top_k: int
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        idxs = [i for i, _ in candidates]
        sub = doc_vecs[idxs]  # (m, dim)
        rel = matmul(sub, query_vec)  # クエリとの類似度
        sim = matmul(sub, sub.T)  # 候補同士の類似度

        selected: list[int] = []  # sub 内のローカル添字
        remaining = list(range(len(idxs)))
        while remaining and len(selected) < top_k:
            if not selected:
                # 1件目は純粋に関連度が最大のものを選ぶ
                best = max(remaining, key=lambda j: rel[j])
            else:
                def mmr_score(j: int) -> float:
                    redundancy = max(float(sim[j][s]) for s in selected)
                    return self.lambda_ * float(rel[j]) - (1 - self.lambda_) * redundancy

                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)

        # 返すスコアは MMR 値ではなくクエリとの関連度。
        # MMR 値は選択順序を決めるための内部量で、選択済み集合に依存するため
        # 閾値判定や構成間の比較には使えない。
        return [(idxs[j], float(rel[j])) for j in selected]


class CrossEncoderReranker:
    """クロスエンコーダによる再順位付け。

    [CLS] query [SEP] passage [SEP] を1本のモデルに通し、関連度スコアを直接出す。
    バイエンコーダのように文書を事前ベクトル化できないため、
    候補数 × 推論回数のコストがかかる。候補50件で CPU 数百 ms 程度。
    """

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        from sentence_transformers import CrossEncoder  # 遅延 import

        self.model = CrossEncoder(model_name, device="cpu", max_length=512)
        self.name = "cross-encoder"

    def rerank_texts(
        self, query: str, candidates: list[tuple[int, float]], texts: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        pairs = [(query, texts[i]) for i, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip([i for i, _ in candidates], scores), key=lambda x: -x[1]
        )
        return [(int(i), float(s)) for i, s in ranked[:top_k]]
