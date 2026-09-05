"""RAG パイプラインの組み立て。

    質問 → [検索] → [順位付け] → [プロンプト構築] → [生成] → 引用付き回答

この薄さが要点で、RAG 固有の難しさは
「検索が正解を拾えるか」と「拾ったものをどう提示するか」に集中している。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .generator import Answer, ExtractiveGenerator


@dataclass
class RagResult:
    question: str
    answer: Answer
    retrieved: list
    scores: list
    timings_ms: dict = field(default_factory=dict)

    def show(self) -> str:
        lines = [f"Q. {self.question}", "", self.answer.text, "", "── 参照した文書 ──"]
        for i, (c, s) in enumerate(zip(self.retrieved, self.scores), 1):
            lines.append(f"  [{i}] score={s:.4f}  {c.doc_id} / {c.primary_heading}")
        t = " / ".join(f"{k} {v:.0f}ms" for k, v in self.timings_ms.items())
        lines.append(f"── {t} ──")
        return "\n".join(lines)


class RagPipeline:
    def __init__(self, retriever, generator=None, top_k: int = 5, min_score: float | None = None):
        self.retriever = retriever
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k
        # 閾値を下回る結果を捨てる。「答えが無い質問」に何かを答えてしまうのを防ぐ
        # 第一の防衛線。第二の防衛線はシステムプロンプトの指示（generator.py）。
        self.min_score = min_score

    def run(self, question: str) -> RagResult:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, top_k=self.top_k)
        if self.min_score is not None:
            hits = [(i, s) for i, s in hits if s >= self.min_score]
        t1 = time.perf_counter()

        chunks = [self.retriever.chunks[i] for i, _ in hits]
        answer = self.generator.generate(question, chunks)
        t2 = time.perf_counter()

        return RagResult(
            question=question,
            answer=answer,
            retrieved=chunks,
            scores=[s for _, s in hits],
            timings_ms={"検索": (t1 - t0) * 1000, "生成": (t2 - t1) * 1000},
        )
