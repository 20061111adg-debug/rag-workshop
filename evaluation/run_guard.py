"""回答不能質問の検出（ハルシネーション対策の評価）。

    ./.venv/bin/python -m evaluation.run_guard

RAG の事故で最も厄介なのは「答えが無いのに、それらしい答えを作ってしまう」こと。
社内文書検索では特に致命的で、存在しない規程を根拠に業務判断されうる。

防衛線は2つある。
  1. 検索スコアの閾値      : そもそも LLM に渡さない（本スクリプトが評価する層）
  2. プロンプトによる指示  : 「無ければ無いと言え」（generator.py の SYSTEM_PROMPT）

ここでは1つ目を数値化する。「答えられる質問の1位スコア」と
「答えられない質問の1位スコア」が分離できていなければ、閾値では守れない。
"""
from __future__ import annotations

from evaluation.configs import build
from evaluation.dataset import load_queries
from ragkit.loader import load_corpus


def _stats(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    s = sorted(vals)
    return f"min={s[0]:+.3f} 中央={s[len(s)//2]:+.3f} max={s[-1]:+.3f}"


def auc(pos: list[float], neg: list[float]) -> float:
    """順位ベースの AUC（回答可能の方が高スコアになる確率）。

    閾値の選び方に依存しないため、「そもそも分離できているのか」を見るのに適する。
    0.5 = 完全にランダム（分離できていない）、1.0 = 完全に分離。
    """
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """回答可否を最もよく分ける閾値と、その正解率を返す。"""
    best = (0.0, 0.0)
    for t in sorted(set(pos + neg)):
        acc = (sum(1 for p in pos if p >= t) + sum(1 for n in neg if n < t)) / (
            len(pos) + len(neg)
        )
        if acc > best[1]:
            best = (t, acc)
    return best


def main() -> None:
    docs = load_corpus("data/corpus")
    queries = load_queries()
    answerable = [q for q in queries if q.answerable]
    unanswerable = [q for q in queries if not q.answerable]

    configs = build(docs, include_cross_encoder=True)
    targets = {
        k: v
        for k, v in configs.items()
        if k
        in (
            "S3 BM25 単独(機能語減衰)",
            "S3 Dense 単独(E5)",
            "S3 Hybrid RRF",
            "S4 Dense + CrossEncoder",
        )
    }

    baseline = len(answerable) / (len(answerable) + len(unanswerable))
    print(f"回答可能 {len(answerable)}件 / 回答不能 {len(unanswerable)}件")
    print(f"基準線: 何も判定せず常に回答した場合の正解率 = {baseline:.1%}")
    print("　この値を上回らない閾値は、門番として機能していない。\n")
    print(
        f"{'構成':<28}{'回答可能の1位スコア':<34}{'回答不能の1位スコア':<34}"
        f"{'最良閾値':>10}{'正解率':>8}{'AUC':>7}{'判定':>10}"
    )
    print("-" * 133)

    for name, r in targets.items():
        pos = [r.search(q.question, 1)[0][1] for q in answerable]
        neg = [r.search(q.question, 1)[0][1] for q in unanswerable]
        t, acc = best_threshold(pos, neg)
        a = auc(pos, neg)
        if a >= 0.90:
            verdict = "実用可"
        elif a >= 0.78:
            verdict = "弱い"
        else:
            verdict = "不可"
        print(
            f"{name:<28}{_stats(pos):<34}{_stats(neg):<34}"
            f"{t:>10.3f}{acc:>8.1%}{a:>7.2f}{verdict:>10}"
        )

    print(
        "\n読み取り方:\n"
        "  ・どの構成も AUC 0.8 前後にとどまり、分離は完全ではない。\n"
        "    スコア閾値だけでハルシネーションは防げず、\n"
        "    プロンプト側の指示（「無ければ無いと言え」）との二段構えが必要。\n"
        "  ・E5 の cosine は 0.816〜0.914 と極端に狭い範囲に固まる。\n"
        "    順位付けには十分機能するが、絶対値には意味が無く、\n"
        "    「0.8 以上なら関連あり」といった直感的な閾値は成立しない。\n"
        "    閾値はモデルとコーパスごとに実測して決めるしかない。\n"
        "  ・RRF のスコアは 0.031〜0.033 のごく狭い離散値しか取らない。\n"
        "    値の意味は「何個の検索器が上位に挙げたか」という合議の度合いであって、\n"
        "    関連度の強さではない。閾値としては極めて壊れやすい。"
    )


if __name__ == "__main__":
    main()
