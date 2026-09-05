"""埋め込み（ベクトル化）のバックエンド。

意図的に3段階を用意している。「密ベクトル検索は魔法ではない」ことを
段階的に確認するため。

  TfidfEmbedder : 語彙一致のみ。次元 = 語彙数。意味は一切理解しない
  LsaEmbedder   : TF-IDF を SVD で低次元化。共起から擬似的な意味を獲得（numpy のみ）
  E5Embedder    : ニューラル埋め込み。事前学習された意味空間（要モデルDL）

前2つは numpy だけで動くため、モデルをダウンロードできない環境でも
RAG のアーキテクチャ全体を最後まで動かせる。
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ._numpy_compat import matmul
from .tokenize_ja import tokenize


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化する。

    正規化しておくと cosine 類似度が単なる内積になり、
    検索が1回の行列積で済む（store.py の高速化はこれに依存）。
    """
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


class TfidfEmbedder:
    """TF-IDF ベクトル。密ベクトルとの対照用ベースライン。

    「ベクトル検索」と呼ばれていても、これは語彙一致でしかない。
    クエリに含まれない語で書かれた文書は、原理的に絶対にヒットしない。
    """

    name = "tfidf"

    def fit(self, docs: list[str]) -> "TfidfEmbedder":
        counters = [Counter(tokenize(d)) for d in docs]
        vocab = sorted({t for c in counters for t in c})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        n = len(docs)
        df = Counter(t for c in counters for t in c)
        self.idf = np.array(
            [math.log((n + 1) / (df[t] + 1)) + 1.0 for t in vocab], dtype=np.float32
        )
        self.dim = len(vocab)
        self._doc_matrix = l2_normalize(self._to_matrix(counters))
        return self

    def _to_matrix(self, counters: list[Counter]) -> np.ndarray:
        mat = np.zeros((len(counters), self.dim), dtype=np.float32)
        for i, c in enumerate(counters):
            for t, f in c.items():
                j = self.vocab.get(t)
                if j is not None:
                    mat[i, j] = (1.0 + math.log(f)) * self.idf[j]
        return mat

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self._to_matrix([Counter(tokenize(t)) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]


class LsaEmbedder:
    """潜在意味解析（LSA）。TF-IDF 行列を SVD で低次元に圧縮する。

    共起パターンから「一緒に出てくる語は近い」を学習するため、
    完全一致しない語でもある程度ヒットするようになる。
    これが「密ベクトル埋め込み」の最も素朴な形であり、
    ニューラル埋め込みとの違いは "何から共起を学んだか" の規模でしかない。

    - LSA : 目の前のコーパス（数十チャンク）から学習 → 語彙外に無力
    - E5  : Web規模の多言語コーパスから事前学習 → 未知の言い換えにも対応
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.name = f"lsa{dim}"

    def fit(self, docs: list[str]) -> "LsaEmbedder":
        self._tfidf = TfidfEmbedder().fit(docs)
        x = self._tfidf._doc_matrix  # (n, vocab)
        k = min(self.dim, min(x.shape) - 1)
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        self._vt = vt[:k]  # (k, vocab) 射影行列
        self.dim = k
        self.name = f"lsa{k}"
        return self

    def _project(self, texts: list[str]) -> np.ndarray:
        sparse = self._tfidf.encode_documents(texts)  # (n, vocab)
        return l2_normalize(matmul(sparse, self._vt.T))  # (n, k)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._project(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._project([text])[0]


class E5Embedder:
    """multilingual-e5-small によるニューラル埋め込み（118M パラメータ / CPU 可）。

    実装上の要点は「クエリと文書で接頭辞が異なる」こと。
      文書: "passage: 在宅勤務手当として..."
      質問: "query: 家で働くとお金はもらえますか"

    E5 系はこの非対称な形式で学習されているため、接頭辞を付け忘れると
    モデルの性能を数ポイント落とす。ライブラリが自動で付けてくれるとは限らず、
    埋め込みモデルごとに作法が違う点は実務で確実に踏む落とし穴。
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer  # 遅延 import

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = model_name.split("/")[-1]

    def fit(self, docs: list[str]) -> "E5Embedder":
        return self  # 事前学習済みなのでコーパスへの適合は不要

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            f"query: {text}",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
