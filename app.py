"""ブラウザで動く社内文書チャットのデモサーバ。

    ./.venv/bin/python app.py            # http://localhost:8000
    ./.venv/bin/python app.py --port 9000
    ./.venv/bin/python app.py --llm      # 回答生成に Claude API を使う

Web フレームワークは使わず Python 標準の http.server だけで書いている。
RAG のデモに必要なのは「質問を受けて、検索して、返す」だけで、
それは標準ライブラリで十分に足りる、ということを示すため。

デモの肝は構成を切り替えられること。
同じ質問に対して「素朴な構成」と「最良の構成」がどう違うかを
その場で見せられると、数値の表より遥かに伝わる。
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ragkit.chunker import STRATEGIES
from ragkit.embedder import E5Embedder, TfidfEmbedder
from ragkit.generator import ExtractiveGenerator
from ragkit.loader import load_corpus
from ragkit.pipeline import RagPipeline
from ragkit.rerank import CrossEncoderReranker
from ragkit.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankRetriever,
)

WEB_DIR = Path(__file__).parent / "web"

# 画面に出す構成の一覧。key は API でやり取りする識別子
CONFIGS = {
    "naive": {"label": "① 素朴（固定長 + TF-IDF）", "note": "入門記事によくある実装"},
    "bm25": {"label": "② BM25（語彙一致）", "note": "固有名詞に強い"},
    "dense": {"label": "③ 密ベクトル（E5）", "note": "言い換えに強い"},
    "hybrid": {"label": "④ ハイブリッド（RRF）", "note": "両者を順位で統合"},
    "best": {"label": "⑤ ＋リランク（最良）", "note": "候補20件を精査。遅い"},
}

_state: dict = {}


@lru_cache(maxsize=None)
def get_retriever(name: str):
    """構成ごとの検索器を作る。モデルとチャンクは使い回す。"""
    docs = _state["docs"]
    if name == "naive":
        return DenseRetriever(STRATEGIES["fixed"](docs), TfidfEmbedder())
    chunks = _state["chunks"]
    if name == "bm25":
        return BM25Retriever(chunks)
    dense = _state["dense"]
    if name == "dense":
        return dense
    hybrid = HybridRetriever([BM25Retriever(chunks), dense], method="rrf")
    if name == "hybrid":
        return hybrid
    return RerankRetriever(hybrid, _state["ce"], candidate_k=20)


def answer(question: str, config: str, top_k: int) -> dict:
    retriever = get_retriever(config)
    pipe = RagPipeline(retriever, _state["generator"], top_k=top_k)
    t0 = time.perf_counter()
    result = pipe.run(question)
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "answer": result.answer.text,
        "generator": result.answer.generator,
        "elapsed_ms": round(elapsed, 1),
        "timings": {k: round(v, 1) for k, v in result.timings_ms.items()},
        "sources": [
            {
                "rank": i,
                "score": round(s, 4),
                "doc_id": c.doc_id,
                "title": c.meta.get("title", c.title),
                "heading": c.primary_heading,
                "updated": c.meta.get("updated", ""),
                "owner": c.meta.get("owner", ""),
                # 文脈ヘッダ行を除いた本文を返す
                "text": c.text.split("\n", 1)[-1].strip(),
            }
            for i, (c, s) in enumerate(zip(result.retrieved, result.scores), 1)
        ],
    }


def compare(question: str) -> list[dict]:
    """全構成で同じ質問を引き、1位だけを並べる。構成の違いを見せるため。"""
    out = []
    for name, meta in CONFIGS.items():
        retriever = get_retriever(name)
        t0 = time.perf_counter()
        hits = retriever.search(question, top_k=1)
        ms = (time.perf_counter() - t0) * 1000
        top = retriever.chunks[hits[0][0]] if hits else None
        out.append({
            "config": name,
            "label": meta["label"],
            "note": meta["note"],
            "elapsed_ms": round(ms, 1),
            "top": None if top is None else {
                "title": top.meta.get("title", top.title),
                "heading": top.primary_heading,
                "score": round(hits[0][1], 4),
            },
        })
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # アクセスログを静かにする
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/configs":
            self._json({
                "configs": [{"key": k, **v} for k, v in CONFIGS.items()],
                "generator": _state["generator"].name,
                "chunks": len(_state["chunks"]),
                "docs": len(_state["docs"]),
            })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "不正なJSONです"}, 400)

        question = (payload.get("question") or "").strip()
        if not question:
            return self._json({"error": "質問が空です"}, 400)

        try:
            if self.path == "/api/ask":
                cfg = payload.get("config", "best")
                if cfg not in CONFIGS:
                    return self._json({"error": f"未知の構成: {cfg}"}, 400)
                return self._json(answer(question, cfg, int(payload.get("top_k", 3))))
            if self.path == "/api/compare":
                return self._json({"results": compare(question)})
        except Exception as exc:  # デモなので原因を画面に返す
            return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--llm", action="store_true", help="回答生成に Claude API を使う")
    args = ap.parse_args()

    print("文書と埋め込みモデルを読み込んでいます…")
    docs = load_corpus("data/corpus")
    _state["docs"] = docs
    _state["chunks"] = STRATEGIES["heading_context"](docs)
    _state["dense"] = DenseRetriever(_state["chunks"], E5Embedder())
    _state["ce"] = CrossEncoderReranker()

    if args.llm:
        from ragkit.generator import ClaudeGenerator

        _state["generator"] = ClaudeGenerator()
    else:
        _state["generator"] = ExtractiveGenerator()

    for name in CONFIGS:  # 起動時に全構成を用意しておく（初回質問を待たせない）
        get_retriever(name)

    print(f"  文書 {len(docs)}件 / チャンク {len(_state['chunks'])}件")
    print(f"  回答生成: {_state['generator'].name}")
    print(f"\n  http://localhost:{args.port}  を開いてください（Ctrl-C で終了）\n")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
