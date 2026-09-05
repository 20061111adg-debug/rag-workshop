"""検索精度の評価指標。

RAG の改善は「体感で良くなった」では前に進まない。
生成の良し悪しを測るには LLM 評価が要るが、
**検索の良し悪しは正解データさえあれば LLM なしで測れる**。
RAG の失敗の大半は検索側で起きるため、まずここを数値化するのが費用対効果が高い。

用語:
  gold      : その質問に答えるために必要な (doc_id, 見出し) の集合
  retrieved : 検索が返したチャンクの順位付きリスト
  hit       : チャンクが跨ぐ見出しのいずれかが gold に含まれること
"""
from __future__ import annotations

import math


def _is_hit(chunk, gold: set[tuple[str, str]]) -> bool:
    return any((chunk.doc_id, h) in gold for h in chunk.headings)


def recall_at_k(chunks: list, gold: set[tuple[str, str]], k: int) -> float:
    """上位k件で gold の何割を回収できたか。

    RAG で最も重要な指標。ここで拾えなかった情報は、
    後段の LLM がどれだけ賢くても絶対に答えに現れない（検索が天井を決める）。
    """
    if not gold:
        return float("nan")
    found = {
        (c.doc_id, h) for c in chunks[:k] for h in c.headings if (c.doc_id, h) in gold
    }
    return len(found) / len(gold)


def precision_at_k(chunks: list, gold: set[tuple[str, str]], k: int) -> float:
    """上位k件のうち何割が当たりか。ノイズの少なさ = LLM のトークン効率。"""
    if not gold or k == 0:
        return float("nan")
    top = chunks[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if _is_hit(c, gold)) / len(top)


def mrr(chunks: list, gold: set[tuple[str, str]]) -> float:
    """最初の正解が何位に出たかの逆数。1位=1.0, 2位=0.5, 3位=0.33。

    「1位に出るか」を重視する指標。リランクの効果が最も鋭く出る。
    """
    if not gold:
        return float("nan")
    for i, c in enumerate(chunks, 1):
        if _is_hit(c, gold):
            return 1.0 / i
    return 0.0


def ndcg_at_k(chunks: list, gold: set[tuple[str, str]], k: int) -> float:
    """順位を対数で割り引いた正規化累積利得。

    Recall は「拾えたか」しか見ないが、nDCG は「上位に置けたか」まで見る。
    LLM のコンテキストは前方ほど効きやすいため、順位の質は実効精度に直結する。
    """
    if not gold:
        return float("nan")

    # 同じ gold セクションを覆うチャンクが複数返ることがある
    # （特にオーバーラップ分割）。2回目以降は新しい情報を含まないので加点しない。
    # これを怠ると DCG が理想DCGを超え、nDCG が 1.0 を上回るという
    # 指標として破綻した値になる。
    seen: set[tuple[str, str]] = set()
    dcg = 0.0
    for i, c in enumerate(chunks[:k], 1):
        new_hits = {(c.doc_id, h) for h in c.headings if (c.doc_id, h) in gold} - seen
        if new_hits:
            dcg += 1.0 / math.log2(i + 1)
            seen |= new_hits

    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(chunks: list, gold: set[tuple[str, str]], ks=(1, 3, 5, 10)) -> dict:
    out = {"mrr": mrr(chunks, gold)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(chunks, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(chunks, gold, k)
    out["precision@5"] = precision_at_k(chunks, gold, 5)
    return out


def aggregate(per_query: list[dict]) -> dict:
    """質問ごとのスコアを平均する（macro average）。

    NaN（回答不能問題）は除外して平均する。
    """
    if not per_query:
        return {}
    keys = per_query[0].keys()
    out = {}
    for key in keys:
        vals = [q[key] for q in per_query if not math.isnan(q[key])]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out
