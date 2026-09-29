"""この検証のゴールと合格条件。全 Step 共通。

**先に「何ができれば成功か」を決めてから測る。**
これを決めずに数字だけ眺めても、良くなったのか分からない。

  やること   : 社内文書について質問したら、答えが書いてある箇所を見つけてくる
  テスト質問 : 下の QUESTIONS（同じことを、違う言い方で聞いている）
  正解       : GOLD のセクション
  合格条件   : すべての質問で、正解が上位 PASS_K 件に入ること

上位 PASS_K 件としたのは、AI に渡すのがその数件だから。
そこに入っていなければ、AI は原理的に答えられない。

なお、ここでの合否は**質問2問だけの判定**である。
本当に良くなったかは 41 問で測る必要がある（step5_evaluate.py）。
2問で判断すると誤るということ自体が、Step 5 の主題になっている。
"""
from __future__ import annotations

QUESTIONS = [
    "在宅勤務の手当はいくらですか",   # 文書と同じ言葉を使った質問
    "家で働くとお金はもらえますか",   # 同じ意味を、別の言葉で聞いた質問
]

GOLD = ("01_remote_work", "6. 通信費および設備の補助")

PASS_K = 3


def gold_rank(retriever, question: str) -> int | None:
    """正解が何位に出たか。候補に一度も現れなければ None。"""
    hits = retriever.search(question, top_k=len(retriever.chunks))
    for i, (idx, _) in enumerate(hits, 1):
        c = retriever.chunks[idx]
        if c.doc_id == GOLD[0] and GOLD[1] in c.headings:
            return i
    return None


def describe() -> str:
    return (
        "【この検証のゴール】\n"
        f"  質問に対して、答えが書いてある箇所（{GOLD[0]} / {GOLD[1]}）を\n"
        f"  上位 {PASS_K} 件以内に出せれば合格。\n"
        f"  テスト質問は {len(QUESTIONS)} 問。同じことを違う言い方で聞いている。\n"
    )


def judge(name: str, retriever) -> bool:
    """合否を表示して返す。各 Step の末尾で呼ぶ。"""
    ranks = [gold_rank(retriever, q) for q in QUESTIONS]
    ok = all(r is not None and r <= PASS_K for r in ranks)
    fmt = lambda r: "圏外" if r is None else f"{r}位"

    print("\n" + "=" * 62)
    print(f"判定 : {name}")
    print("=" * 62)
    for q, r in zip(QUESTIONS, ranks):
        mark = "○" if (r is not None and r <= PASS_K) else "×"
        print(f"  {mark} {q:<22} 正解は {fmt(r):>5}")
    print(f"\n  → {'合格' if ok else '不合格'}（合格条件: 全問で上位 {PASS_K} 件以内）")
    return ok
