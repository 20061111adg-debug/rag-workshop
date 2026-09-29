"""Step 4: リランクは何をしているのか、候補は何件取るべきか

    ./.venv/bin/python steps/step4_rerank.py

検索を2段に分けるのが定石。
  1段目（検索）  : 数万件から候補を絞る。速さと「取りこぼさないこと」が仕事
  2段目（リランク）: 候補だけを精査して並べ替える。精度が仕事

なぜ分けるのか。
  バイエンコーダ（1段目）はクエリと文書を別々にベクトル化する。
  文書側を事前計算できるので速いが、両者の相互作用は見られない。
  クロスエンコーダ（2段目）は [クエリ; 文書] を連結して1回推論する。
  精度は高いが、検索のたびに候補数だけ推論が要る。全件には使えない。

この構造から重要な帰結が出る:
  **1段目に求められるのは順位の精度ではなく、候補に正解を含めること。**
  候補に入っていない正解は、2段目がどれだけ賢くても絶対に救えない。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import load_queries
from evaluation.metrics import aggregate, evaluate
from ragkit.chunker import STRATEGIES
from ragkit.embedder import E5Embedder
from ragkit.loader import load_corpus
from ragkit.rerank import CrossEncoderReranker
from ragkit.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
)


def hit(chunk, gold) -> bool:
    return any((chunk.doc_id, h) in gold for h in chunk.headings)


def candidate_recall(retriever, queries, chunks, k: int) -> float:
    """1段目が候補k件の中に正解をどれだけ含められたか（2段目の性能上限）。"""
    total = 0.0
    for q in queries:
        found = {
            (chunks[i].doc_id, h)
            for i, _ in retriever.search(q.question, top_k=k)
            for h in chunks[i].headings
            if (chunks[i].doc_id, h) in q.gold
        }
        total += len(found) / len(q.gold)
    return total / len(queries)


def score(retriever, queries, chunks) -> dict:
    per = []
    for q in queries:
        hits = [chunks[i] for i, _ in retriever.search(q.question, top_k=10)]
        per.append(evaluate(hits, q.gold))
    return aggregate(per)


def main() -> None:
    import goal

    print(goal.describe())
    docs = load_corpus("data/corpus")
    chunks = STRATEGIES["heading_context"](docs)
    queries = [q for q in load_queries() if q.answerable]

    dense = DenseRetriever(chunks, E5Embedder())
    hybrid = HybridRetriever([BM25Retriever(chunks), dense], method="rrf")
    ce = CrossEncoderReranker()

    # ── ① 実際の並べ替えを1問見る ──────────────────────
    q = next(x for x in queries if x.qid == "q11")  # パソコン紛失（1段目が苦手な質問）
    print("=" * 74)
    print(f"① クロスエンコーダは順位をどう変えるか : 「{q.question}」")
    print("=" * 74)
    candidates = hybrid.search(q.question, top_k=20)  # 候補は20件取る
    after = ce.rerank_texts(q.question, candidates, [c.embed_text for c in chunks], 8)
    before = candidates[:8]  # 1段目の上位8件だけ表示

    gold_pos = next(
        (i for i, (idx, _) in enumerate(candidates, 1) if hit(chunks[idx], q.gold)), None
    )
    print(f"  1段目での正解の位置: {gold_pos}位（候補20件中）← 上位8件には入っていない\n")
    print(f"  {'':2} {'1段目(Hybrid RRF)':<34} {'2段目(CrossEncoder)':<34}")
    for r, ((i_b, _), (i_a, s_a)) in enumerate(zip(before, after), 1):
        mark_b = "◎" if hit(chunks[i_b], q.gold) else "  "
        mark_a = "◎" if hit(chunks[i_a], q.gold) else "  "
        lb = f"{mark_b}{chunks[i_b].doc_id[:12]}/{chunks[i_b].primary_heading[:14]}"
        la = f"{mark_a}{chunks[i_a].doc_id[:12]}/{chunks[i_a].primary_heading[:14]} {s_a:+.1f}"
        print(f"  {r}. {lb:<34} {la:<34}")
    print("""  ◎ = 正解セクション

  1段目は「紛失」という語や周辺の意味でFAQを上位に並べたが、正解の
  「インシデント発生時の対応」は候補の下の方に埋もれていた。
  2段目は質問と各チャンクを1本につないで読み直すため、
  「パソコンを失くした = 報告すべきインシデント」という対応関係を捉えて引き上げる。
  これがバイエンコーダには原理的にできない仕事。""")

    # ── ② 候補数をいくつにすべきか ──────────────────────
    print()
    print("=" * 74)
    print("② 候補数(candidate_k)を変えるとどうなるか")
    print("=" * 74)
    print(f"  {'候補数':>6} {'1段目の候補到達率':>18} {'最終MRR':>10} {'最終Recall@5':>14} {'1問あたり':>10}")
    print("  " + "-" * 66)
    for k in (3, 5, 10, 20, 40):
        ceiling = candidate_recall(hybrid, queries, chunks, k)
        rr = RerankRetriever(hybrid, ce, candidate_k=k)
        t0 = time.perf_counter()
        sc = score(rr, queries, chunks)
        ms = (time.perf_counter() - t0) / len(queries) * 1000
        print(f"  {k:>6} {ceiling:>17.1%} {sc['mrr']:>10.3f} {sc['recall@5']:>14.3f} {ms:>9.0f}ms")
    print("""
  候補到達率が2段目の天井。候補に入っていない正解は絶対に救えない。
  候補を増やすほど天井は上がるが、推論回数に比例して遅くなる。""")

    # ── ③ 1段目の違いは吸収されるのか ────────────────────
    print()
    print("=" * 74)
    print("③ 1段目を変えても結果は変わるのか（候補20件で固定）")
    print("=" * 74)
    print(f"  {'構成':<34}{'1段目のみ':>12}{'+CrossEncoder':>16}")
    print("  " + "-" * 62)
    for name, base in (("Dense 単独(E5)", dense), ("Hybrid RRF", hybrid)):
        s1 = score(base, queries, chunks)["mrr"]
        s2 = score(RerankRetriever(base, ce, candidate_k=20), queries, chunks)["mrr"]
        print(f"  {name:<34}{s1:>12.3f}{s2:>16.3f}")
    goal.judge("Step 4（候補20件をクロスエンコーダで並べ直す）",
               RerankRetriever(hybrid, ce, candidate_k=20))

    print("""
  1段目では 0.927 と 0.935 で差があるのに、リランク後は同じ値になる。
  候補20件の中に正解が入ってさえいれば、1段目の細かい順位差は
  2段目が吸収してしまうということ。

  → 1段目のチューニングに時間をかけるより、2段目を入れるほうが効く。
  → ただし「候補に入れる」ことだけは1段目にしかできない。

考えてほしいこと
  1. ②で候補数を増やし続けると、どこで頭打ちになったか。それはなぜか？
  2. 1問300msは、社内チャットボットとして許容できるか？（ユーザーは何秒待てるか）
  3. 遅さを緩和する方法を3つ挙げるとしたら何か？
""")


if __name__ == "__main__":
    main()
