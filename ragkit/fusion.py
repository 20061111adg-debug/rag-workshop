"""複数の検索結果の統合。

BM25 と密ベクトルは得意分野が違う（固有名詞 vs 言い換え）ので、
両方を走らせて統合すると単独より安定する。これがハイブリッド検索。

問題は「スコアの単位が違う」こと。
  BM25   : 0〜十数の非有界スコア。コーパスや語彙で分布が変わる
  cosine : -1〜1 に収まるが、実際の分布はモデル依存で偏る

そのまま足すと片方に支配される。解法は2つ。
  1. スコア正規化して加重和 → 重みとスケーリングの調整が必要
  2. スコアを捨てて「順位」だけ使う (RRF) → 調整不要で頑健

実務では RRF が既定値として選ばれることが多い。理由は下記 rrf() を参照。
"""
from __future__ import annotations

from collections import defaultdict


def rrf(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion。

        RRF(d) = Σ_r  w_r / (k + rank_r(d))       rank は 1 始まり

    スコアの絶対値を一切使わず順位のみを見るため、
    正規化もスケール合わせも不要で、検索器を増減しても壊れない。

    定数 k=60 は元論文 (Cormack et al., 2009) の推奨値。
    k が大きいほど上位と下位の差が縮まり、順位差に鈍感になる。
      k=0  → 1位が 1.0、2位が 0.5 と、1位が極端に強い
      k=60 → 1位が 0.0164、2位が 0.0161 とほぼ横並び（多数決に近い）
    「複数の検索器が揃って上位に挙げた文書」を拾いたいので、
    ある程度大きい k のほうが合議として機能する。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, (idx, _score) in enumerate(ranking, start=1):
            scores[idx] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def weighted_sum(
    rankings: list[list[tuple[int, float]]], weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """スコアを min-max 正規化してから加重和を取る。

    RRF と違いスコアの大小（1位と2位がどれだけ離れているか）を保持できるが、
    正規化が「その検索での最大・最小」に依存するため、
    クエリごとにスケールが変わり不安定になりやすい。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        if not ranking:
            continue
        norm = _minmax([s for _, s in ranking])
        for (idx, _), ns in zip(ranking, norm):
            scores[idx] += w * ns
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
