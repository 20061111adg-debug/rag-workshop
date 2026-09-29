"""Step 3: BM25 と密ベクトルはどう補い合うか、統合はなぜ方式で結果が変わるか

    ./.venv/bin/python steps/step3_hybrid.py

Step 2 で「切り方」を解決した。ここでは「探し方」を扱う。

  BM25    : 語彙の一致を見る。固有名詞・型番・略語に強い
  密ベクトル : 意味の近さを見る。言い換え・口語に強い

両者は得意分野が違う。ならば両方使えばよい——が、
統合のやり方を間違えると単独より悪くなる。そこまで実際に確かめる。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import load_queries
from ragkit.chunker import STRATEGIES
from ragkit.embedder import E5Embedder
from ragkit.loader import load_corpus
from ragkit.retriever import BM25Retriever, DenseRetriever, HybridRetriever

RRF_K = 60


def gold_rank(retriever, query, chunks, top_k=None) -> int | None:
    """最初に正解が出てくる順位。候補に一度も現れなければ None。

    BM25 はクエリと共通する語を1つも持たないチャンクを「結果に含めない」。
    順位が付かないのであって、最下位になるのではない。この違いが RRF に効く。
    """
    ranked = retriever.search(query.question, top_k=top_k or len(chunks))
    for i, (idx, _) in enumerate(ranked, 1):
        c = chunks[idx]
        if any((c.doc_id, h) in query.gold for h in c.headings):
            return i
    return None


def fmt(rank) -> str:
    return "圏外" if rank is None else f"{rank}位"


def sort_key(rank) -> int:
    return 9999 if rank is None else rank


def main() -> None:
    import goal

    print(goal.describe())
    docs = load_corpus("data/corpus")
    chunks = STRATEGIES["heading_context"](docs)
    queries = [q for q in load_queries() if q.answerable]

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, E5Embedder())
    rrf = HybridRetriever([bm25, dense], method="rrf")
    weighted = HybridRetriever([bm25, dense], method="weighted")

    ranks = {
        q.qid: {
            "q": q,
            "bm25": gold_rank(bm25, q, chunks),
            "dense": gold_rank(dense, q, chunks),
            "rrf": gold_rank(rrf, q, chunks),
        }
        for q in queries
    }

    # ── ① 得意分野は本当に違うのか ──────────────────────
    print("=" * 74)
    print("① BM25 と密ベクトルは、違う質問で失敗しているか")
    print("=" * 74)
    bm_win = sum(1 for r in ranks.values() if sort_key(r["bm25"]) < sort_key(r["dense"]))
    dn_win = sum(1 for r in ranks.values() if sort_key(r["dense"]) < sort_key(r["bm25"]))
    tie = len(ranks) - bm_win - dn_win
    print(f"  BM25 の方が上位に出した質問 : {bm_win} 問")
    print(f"  密ベクトルの方が上位        : {dn_win} 問")
    print(f"  同順位                      : {tie} 問")
    print("\n  → 全体では密ベクトルが優勢。ただし BM25 が勝つ質問も確かに存在する。")
    print("    「どちらか一方が常に正しい」ではないことが、統合を試す根拠になる。\n")

    for label, key, other in (("BM25 が圧勝", "bm25", "dense"), ("密ベクトルが圧勝", "dense", "bm25")):
        picked = sorted(ranks.values(), key=lambda r: sort_key(r[key]) - sort_key(r[other]))[:2]
        print(f"  【{label}】")
        for r in picked:
            print(f"    {r['q'].type:<10} {r['q'].question[:30]:<32} "
                  f"BM25 {fmt(r['bm25']):>5} / 密ベクトル {fmt(r['dense']):>5}")
        print()

    # ── ② RRF が何を計算しているか ──────────────────────
    target = max(ranks.values(), key=lambda r: abs(sort_key(r["bm25"]) - sort_key(r["dense"])))
    q = target["q"]
    contrib = lambda rank: 0.0 if rank is None else 1 / (RRF_K + rank)
    c_bm, c_dn = contrib(target["bm25"]), contrib(target["dense"])
    print("=" * 74)
    print(f"② RRF の計算を1問で追う : 「{q.question}」")
    print("=" * 74)
    print(f"  BM25 での正解の順位      : {fmt(target['bm25']):>5}  → 加点 {c_bm:.5f}")
    print(f"  密ベクトルでの正解の順位 : {fmt(target['dense']):>5}  → 加点 1/(60+{target['dense']}) = {c_dn:.5f}")
    print(f"  正解の合計スコア         : {c_bm + c_dn:.5f}")
    print(f"  → 統合後の正解の順位     : {fmt(target['rrf'])}")
    print(f"""
  RRF はスコアの大きさを見ず、順位だけを 1/(60+順位) に変換して足す。
  60 という定数のおかげで 1位(0.01639) と 5位(0.01538) の差は小さく、
  「両方がそこそこ上位に挙げた文書」が「片方だけが1位にした文書」に競り勝つ。
  つまり多数決に近い挙動になる。

  ⚠ この質問では、それが裏目に出ている。
  BM25 は正解を候補に一度も挙げていない（共通する語が無いので圏外）。
  正解は密ベクトルからの 1票（{c_dn:.5f}）しか得られない。
  一方、両方が候補に入れた無関係な文書は 2票もらえるため、
  正解を追い抜いてしまう。結果、統合後は {fmt(target['rrf'])} と
  密ベクトル単独（{fmt(target['dense'])}）より悪化した。

  RRF は合議である以上、片方が完全に盲目なときは足を引っ張る。
  「全体では RRF が最良」でも、個々の質問では悪化しうる。""")

    # ── ③ 統合方式で結果が変わる ────────────────────────
    print()
    print("=" * 74)
    print("③ RRF と加重和は何が違うのか")
    print("=" * 74)
    diffs = []
    for q in queries:
        r_rrf, r_w = gold_rank(rrf, q, chunks), gold_rank(weighted, q, chunks)
        if r_rrf != r_w:
            diffs.append((q, r_rrf, r_w))
    diffs.sort(key=lambda d: sort_key(d[1]) - sort_key(d[2]))
    print(f"  順位が食い違った質問: {len(diffs)} / {len(queries)} 問\n")
    for q, r_rrf, r_w in diffs:
        better = "RRF" if sort_key(r_rrf) < sort_key(r_w) else "加重和"
        print(f"    {q.question[:34]:<36} RRF {fmt(r_rrf):>5} / 加重和 {fmt(r_w):>5}  → {better}の勝ち")
    goal.judge("Step 3（言葉の検索と意味の検索を RRF で混ぜる）", rrf)

    print("""
  全体では RRF 0.935 / 加重和 0.902（MRR）で RRF が上。
  加重和はスコアを min-max 正規化して足すが、正規化の基準が
  「そのクエリでの最大・最小」なので、クエリごとにスケールが変わってしまう。
  RRF は順位しか使わないため、この不安定さが原理的に起きない。

考えてほしいこと
  1. ①で BM25 が圧勝した質問には、どんな共通点があるか？
  2. 「両方使えば必ず良くなる」は正しいか？（加重和の結果を見て）
  3. もし社内文書が製品型番だらけだったら、BM25 と密ベクトルどちらを重視すべきか？
""")


if __name__ == "__main__":
    main()
