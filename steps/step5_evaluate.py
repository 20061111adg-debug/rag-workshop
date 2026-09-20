"""Step 5: 評価セットは何をしていて、どこまで信じてよいのか

    ./.venv/bin/python steps/step5_evaluate.py

Step 1〜4 のすべての判断は、41問の評価セットの上に乗っている。
土台が傾いていれば、その上の結論も全部傾く。ここではその土台を検分する。

  ① 評価セットの中身
  ② 指標を1問で手計算する
  ③ 正解ラベルが1つ間違っていると何が起きるか（実際に起きた事故の再現）
  ④ 41問で検出できる差の下限はどこか
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import load_queries
from evaluation.metrics import aggregate, evaluate, mrr, ndcg_at_k, recall_at_k
from ragkit.chunker import STRATEGIES
from ragkit.embedder import E5Embedder
from ragkit.loader import load_corpus
from ragkit.rerank import CrossEncoderReranker
from ragkit.retriever import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever


def per_query_mrr(retriever, queries, chunks) -> dict[str, float]:
    out = {}
    for q in queries:
        hits = [chunks[i] for i, _ in retriever.search(q.question, top_k=10)]
        out[q.qid] = mrr(hits, q.gold)
    return out


def main() -> None:
    docs = load_corpus("data/corpus")
    chunks = STRATEGIES["heading_context"](docs)
    all_queries = load_queries()
    queries = [q for q in all_queries if q.answerable]

    dense = DenseRetriever(chunks, E5Embedder())
    hybrid = HybridRetriever([BM25Retriever(chunks), dense], method="rrf")
    reranked = RerankRetriever(hybrid, CrossEncoderReranker(), candidate_k=20)

    # ── ① 評価セットの中身 ──────────────────────────────
    print("=" * 74)
    print("① 評価セットとは何か")
    print("=" * 74)
    q = next(x for x in queries if x.qid == "q13")
    print(f"  質問     : {q.question}")
    print(f"  種別     : {q.type}")
    print(f"  正解     : {sorted(q.gold)}")
    print(f"  期待語句 : {q.answer_keywords}")
    print(f"""
  「正解」は文章ではなく、(文書, 見出し) の集合として持つ。
  正解の文章を書いてしまうと、表現が少し違うだけで採点できなくなる。
  **どこを見れば答えが書いてあるか**だけを持てば、LLM なしで採点できる。

  内訳: 全{len(all_queries)}問 = 回答可能{len(queries)}問 + 回答不能{len(all_queries)-len(queries)}問""")

    # ── ② 指標を手計算する ──────────────────────────────
    print()
    print("=" * 74)
    print(f"② 指標を1問で手計算する : 「{q.question[:30]}…」")
    print("=" * 74)
    hits = [chunks[i] for i, _ in hybrid.search(q.question, top_k=5)]
    print(f"  正解は {len(q.gold)} セクション。上位5件の内訳:")
    for i, c in enumerate(hits, 1):
        ok = any((c.doc_id, h) in q.gold for h in c.headings)
        print(f"    {i}位 {'◎正解' if ok else '  外れ'}  {c.doc_id} / {c.primary_heading[:24]}")
    r5, m, n5 = recall_at_k(hits, q.gold, 5), mrr(hits, q.gold), ndcg_at_k(hits, q.gold, 5)
    print(f"""
  Recall@5 = 拾えた正解 / 全正解         = {r5:.3f}   「取りこぼしていないか」
  MRR      = 1 / 最初に正解が出た順位     = {m:.3f}   「1位に置けたか」
  nDCG@5   = 順位を log で割り引いた利得   = {n5:.3f}   「上位に寄せられたか」

  この3つは別のことを測っている。片方だけ見ると判断を誤る。""")

    # ── ③ ラベルが1つ狂うと何が起きるか ─────────────────
    print()
    print("=" * 74)
    print("③ 正解ラベルが間違っていると何が起きるか（実際に起きた事故の再現）")
    print("=" * 74)
    broken = []
    for x in all_queries:
        if x.qid in ("q06", "q17"):
            g = {("04_onboarding_faq", "経費について") if s[0] == "04_onboarding_faq" else s for s in x.gold}
            x = type(x)(x.qid, x.type, x.question, g, x.answer_keywords, x.note)
        broken.append(x)
    broken = [x for x in broken if x.answerable]

    for label, qs in (("誤ラベル（修正前）", broken), ("正しいラベル（修正後）", queries)):
        rows = []
        for x in qs:
            hits = [chunks[i] for i, _ in hybrid.search(x.question, top_k=10)]
            rows.append(evaluate(hits, x.gold))
        agg = aggregate(rows)
        dis = aggregate([r for x, r in zip(qs, rows) if x.type == "distractor"])
        print(f"  {label:<24} 全体MRR {agg['mrr']:.3f}   distractor の Recall@5 {dis['recall@5']:.2f}")
    print("""
  41問中たった2問のラベルが違うだけで、distractor の評価が 0.88 と 1.00 に割れる。
  0.88 を見て「これは検索では解けない問題だ」と結論づけていたが、それは幻だった。
  **測る道具が壊れていると、間違った結論に確信を持ってたどり着く。**""")

    # ── ④ 41問で検出できる差の下限 ──────────────────────
    print()
    print("=" * 74)
    print("④ 何問あれば、その差を『差』と言ってよいのか")
    print("=" * 74)
    a = per_query_mrr(dense, queries, chunks)
    b = per_query_mrr(hybrid, queries, chunks)
    c = per_query_mrr(reranked, queries, chunks)
    qids = [x.qid for x in queries]
    full = lambda d: sum(d.values()) / len(d)
    print(f"  全31問での MRR : 密ベクトル {full(a):.3f} / ハイブリッド {full(b):.3f} / +リランク {full(c):.3f}")
    print(f"""
  差は ハイブリッド-密ベクトル = {full(b)-full(a):+.3f} 、 リランク-密ベクトル = {full(c)-full(a):+.3f} 。
  この差が「本物」か「たまたまこの31問でそうなっただけ」かを確かめたい。

  やり方はブートストラップ法。31問から**重複を許して**31問を引き直し、
  そのたびに差を計算する。これを2000回繰り返すと差の散らばりが分かる。
  散らばりの範囲（95%信頼区間）が 0 をまたぐなら、差があるとは言えない。
""")
    rng = random.Random(0)

    def boot(x: dict, y: dict, n: int, trials: int = 2000):
        diffs = []
        for _ in range(trials):
            sample = [rng.choice(qids) for _ in range(n)]  # 復元抽出
            diffs.append(sum(y[i] - x[i] for i in sample) / n)
        diffs.sort()
        return diffs[int(trials * 0.025)], diffs[int(trials * 0.975)]

    print(f"  {'比較':<26}{'質問数':>6}{'差の95%信頼区間':>22}{'判定':>18}")
    print("  " + "-" * 72)
    for label, x, y in (("ハイブリッド vs 密ベクトル", a, b), ("リランク vs 密ベクトル", a, c)):
        for n in (10, 31, 100):
            lo, hi = boot(x, y, n)
            verdict = "差があると言える" if lo > 0 else "差があるとは言えない"
            print(f"  {label:<26}{n:>6}{f'[{lo:+.3f}, {hi:+.3f}]':>22}{verdict:>18}")
        print()
    print("""  読み取り方 — 結果は予想より厳しい

  **31問では、どちらの差も統計的に確かめられなかった。**

  ・ハイブリッド vs 密ベクトル（差 +0.008）
    31問の区間 [-0.024, +0.048] は 0 をまたぐ。差があるとは言えない。
    100問に増やしても [-0.010, +0.030] でまだまたぐ。
    この程度の差を確かめるには、さらに多くの質問が要る。

  ・リランク vs 密ベクトル（差 +0.051）
    31問の区間 [-0.027, +0.137] も 0 をまたぐ。**これも言えない。**
    100問なら [+0.006, +0.098] となり、ようやく差だと言える。

  MRR は1問あたり 1.0 / 0.5 / 0.33 … と飛び飛びの値しか取らないため、
  平均のばらつきが大きい。31問では分解能が足りない。

  ここから導かれる、この学習プロジェクト全体への但し書き:

    Step 1→2 の改善（MRR 0.605→0.927、差 +0.32）は十分に大きく、疑いない。
    しかし Step 3・Step 4 の差（+0.008、+0.05）は、
    **この評価セットでは「そう見えた」以上のことを言えない。**

  これは手法の優劣を否定するものではない。
  クロスエンコーダが効くこと自体は広く知られている。
  言えないのは「**この41問で、それを証明した**」という部分。

  社内共有では、この区別をつけて話すと信頼される。
  「効きました」ではなく「この規模では大きな差だけが確認できました」が正確。

考えてほしいこと
  1. 自社文書で評価セットを作るとき、何問くらい用意すべきか？
  2. その41問は誰が正解を決めるのか？（あなた？業務の担当者？）
  3. 評価セットが間違っていないことは、どうやって確かめるのか？
""")


if __name__ == "__main__":
    main()
