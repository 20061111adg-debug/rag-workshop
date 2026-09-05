"""BM25 (Okapi) のスクラッチ実装。

「ベクトル検索があれば全文検索は要らない」は誤り。
固有名詞・型番・略語のような低頻度語の完全一致では、いまだに BM25 が強い。
埋め込みモデルは学習データに無い語をベクトル空間で正しく配置できないため。

スコア式:
    score(q, d) = Σ_{t∈q} IDF(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

    IDF(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )

3つの要素の合成として読むと理解しやすい。
  1. IDF        : 珍しい語ほど強く効かせる
  2. 出現頻度の飽和 : k1 により、同じ語が10回出ても2回の5倍にはならない
  3. 長さ正規化  : b により、長い文書が有利になりすぎるのを抑える
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .tokenize_ja import tokenize, weighted_query_tokens


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[Counter] = []
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # 転置索引
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def fit(self, texts: list[str]) -> "BM25":
        self.doc_tokens = [Counter(tokenize(t)) for t in texts]
        self.doc_len = [sum(c.values()) for c in self.doc_tokens]
        self.n_docs = len(texts)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        # 転置索引: token -> [(doc_index, 出現回数), ...]
        for i, counter in enumerate(self.doc_tokens):
            for token, freq in counter.items():
                self.postings[token].append((i, freq))

        # IDF は索引構築時に一度だけ計算しておく
        for token, plist in self.postings.items():
            df = len(plist)
            self.idf[token] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
        return self

    def search(
        self, query: str, top_k: int = 5, function_weight: float = 1.0
    ) -> list[tuple[int, float]]:
        """クエリに含まれるトークンの postings だけを走査する。

        全文書とスコア計算をせず、転置索引で候補を絞るのが全文検索の要。
        ベクトル検索が全件との内積を必要とするのと対照的。

        function_weight は機能語らしいトークンの減衰率。
        小規模コーパスでは IDF だけでは「ですか」「とは」を無力化できないため、
        日本語ではこの後処理が実効的に効く（1.0 にすれば素朴な BM25 に戻る）。
        """
        weights: dict[str, float] = {}
        for token, w in weighted_query_tokens(query, function_weight):
            weights[token] = max(weights.get(token, 0.0), w)

        scores: dict[int, float] = defaultdict(float)

        for token, w in weights.items():
            if token not in self.postings:
                continue
            idf = self.idf[token] * w
            for doc_i, freq in self.postings[token]:
                dl = self.doc_len[doc_i]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_i] += idf * (freq * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]
