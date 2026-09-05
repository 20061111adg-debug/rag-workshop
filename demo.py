"""対話デモ。任意の質問を投げて検索と回答を確認する。

    ./.venv/bin/python demo.py                          # 既定構成で対話
    ./.venv/bin/python demo.py -q "コアタイムは何時？"      # 単発質問
    ./.venv/bin/python demo.py --config naive           # 素朴な構成と比較
    ./.venv/bin/python demo.py --llm                    # Claude API で回答生成

--llm を付けなければ API キーは不要（検索結果を抽出表示する）。
"""
from __future__ import annotations

import argparse
import sys

from ragkit.chunker import STRATEGIES
from ragkit.embedder import E5Embedder, TfidfEmbedder
from ragkit.generator import ExtractiveGenerator
from ragkit.loader import load_corpus
from ragkit.pipeline import RagPipeline
from ragkit.rerank import CrossEncoderReranker
from ragkit.retriever import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever

CONFIGS = ("naive", "bm25", "dense", "hybrid", "best")


def build_retriever(name: str, docs):
    if name == "naive":
        return DenseRetriever(STRATEGIES["fixed"](docs), TfidfEmbedder())
    chunks = STRATEGIES["heading_context"](docs)
    if name == "bm25":
        return BM25Retriever(chunks)
    dense = DenseRetriever(chunks, E5Embedder())
    if name == "dense":
        return dense
    hybrid = HybridRetriever([BM25Retriever(chunks), dense])
    if name == "hybrid":
        return hybrid
    return RerankRetriever(hybrid, CrossEncoderReranker(), candidate_k=20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("--config", choices=CONFIGS, default="best")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--llm", action="store_true", help="Claude API で回答を生成する")
    args = ap.parse_args()

    docs = load_corpus("data/corpus")
    print(f"構成 '{args.config}' を準備中…", file=sys.stderr)
    retriever = build_retriever(args.config, docs)

    generator = ExtractiveGenerator()
    if args.llm:
        from ragkit.generator import ClaudeGenerator

        generator = ClaudeGenerator()

    pipe = RagPipeline(retriever, generator, top_k=args.top_k)
    print(f"検索器: {retriever.name} / 生成器: {generator.name}", file=sys.stderr)

    if args.question:
        print(pipe.run(args.question).show())
        return

    print("質問を入力してください（空行または Ctrl-D で終了）", file=sys.stderr)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print()
        print(pipe.run(q).show())


if __name__ == "__main__":
    main()
