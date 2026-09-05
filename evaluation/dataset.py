"""評価データセットの読み込み。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalQuery:
    qid: str
    type: str
    question: str
    gold: set[tuple[str, str]]
    answer_keywords: list[str]
    note: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def load_queries(path: str | Path = "data/eval/qa.jsonl") -> list[EvalQuery]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append(
            EvalQuery(
                qid=d["qid"],
                type=d["type"],
                question=d["question"],
                gold={(doc, sec) for doc, sec in d["gold"]},
                answer_keywords=d.get("answer_keywords", []),
                note=d.get("note", ""),
            )
        )
    return rows


def validate(queries: list[EvalQuery], chunks: list) -> list[str]:
    """gold に指定した (doc_id, 見出し) が実在するかを検査する。

    評価セットの typo は「アルゴリズムが悪い」と誤診させる最大の原因。
    必ず起動時に検証する。
    """
    existing = {(c.doc_id, h) for c in chunks for h in c.headings}
    problems = []
    for q in queries:
        for g in q.gold:
            if g not in existing:
                problems.append(f"{q.qid}: gold {g} がコーパスに存在しません")
    return problems
