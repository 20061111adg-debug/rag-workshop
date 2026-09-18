"""Step 2: チャンク分割で検索結果がどう変わるか

    ./.venv/bin/python steps/step2_chunking.py

Step 1 の失敗には2つの原因があった。
  A. 語彙が一致しないと見つからない（「家で働く」≠「在宅勤務」）
  B. 固定長で切るので、チャンクが意味のまとまりになっていない

このステップでは A を先に解消する（意味を捉える埋め込みモデル E5 に固定）。
そのうえで「切り方」だけを変え、B の影響を単独で観察する。
"""
from __future__ import annotations

import sys
from pathlib import Path

# steps/ から直接実行しても ragkit を import できるよう、リポジトリ直下をパスに加える
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragkit.chunker import STRATEGIES
from ragkit.embedder import E5Embedder
from ragkit.loader import load_corpus
from ragkit.retriever import DenseRetriever

QUESTION = "家で働くとお金はもらえますか"
TARGET = ("01_remote_work", "6. 通信費および設備の補助")  # 正解のセクション


def show_chunks(name: str, chunks) -> None:
    """正解セクションを含むチャンクが、どう切られているかを表示する。"""
    hits = [c for c in chunks if c.doc_id == TARGET[0] and TARGET[1] in c.headings]
    print(f"\n■ {name}: 正解セクションに触れるチャンクは {len(hits)} 個")
    for c in hits:
        print(f"  ┌ 埋め込まれる文章（{len(c.embed_text)}文字 / 跨ぐ見出し {len(c.headings)}個）")
        for line in c.embed_text.strip().splitlines():
            print(f"  │ {line}")
        print("  └")


def main() -> None:
    docs = load_corpus("data/corpus")
    embedder = E5Embedder()  # 全戦略で同じモデルを使う＝違いは切り方だけ

    print("=" * 72)
    print("① 同じ規程が、切り方によってどう見えるか")
    print("=" * 72)
    for name in ("fixed", "heading_context"):
        show_chunks(name, STRATEGIES[name](docs))

    print()
    print("=" * 72)
    print(f"② 質問「{QUESTION}」で、正解が何位に来るか")
    print("=" * 72)
    for name, fn in STRATEGIES.items():
        chunks = fn(docs)
        r = DenseRetriever(chunks, embedder)
        ranked = r.search(QUESTION, top_k=len(chunks))
        rank = next(
            i for i, (idx, _) in enumerate(ranked, 1)
            if chunks[idx].doc_id == TARGET[0] and TARGET[1] in chunks[idx].headings
        )
        top = chunks[ranked[0][0]]
        print(f"  {name:<16} 正解の順位: {rank:>2}位   1位: {top.doc_id} / {top.primary_heading}")

    print("""
考えてほしいこと
  1. fixed のチャンクは「何についての文章か」が文章だけで分かるか？
  2. heading_context の1行目「リモートワーク規程 > 6. 通信費…」は何の役に立っているか？
  3. 全体評価（41問）では heading_context が最も良かった（MRR 0.927）。
     ②で1問だけ見た結果と、全体の結果は一致しているか？しないなら、なぜ評価セットが必要なのか？
""")


if __name__ == "__main__":
    main()
