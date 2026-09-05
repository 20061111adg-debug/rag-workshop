"""検索精度の一括評価。

    ./.venv/bin/python -m evaluation.run_eval [--cross-encoder] [--k 5]

LLM を一切呼ばずに、構成間の優劣を数値で比較する。
RAG の失敗の大半は検索段で起きるため、まずここを固めるのが定石。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from evaluation.configs import build
from evaluation.dataset import load_queries, validate
from evaluation.metrics import aggregate, evaluate
from ragkit.loader import load_corpus

REPORT_DIR = Path("reports")


def run(include_cross_encoder: bool = False, top_k: int = 5) -> dict:
    docs = load_corpus("data/corpus")
    queries = load_queries()
    answerable = [q for q in queries if q.answerable]

    configs = build(docs, include_cross_encoder=include_cross_encoder)

    problems = validate(queries, next(iter(configs.values())).chunks)
    if problems:
        raise SystemExit("評価セットの不整合:\n" + "\n".join(problems))

    results: dict = {}
    for name, retriever in configs.items():
        per_query, by_type, latencies = [], defaultdict(list), []
        for q in answerable:
            t0 = time.perf_counter()
            hits = retriever.search(q.question, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)
            chunks = [retriever.chunks[i] for i, _ in hits]
            scores = evaluate(chunks, q.gold)
            per_query.append(scores)
            by_type[q.type].append(scores)

        results[name] = {
            "overall": aggregate(per_query),
            "by_type": {t: aggregate(v) for t, v in by_type.items()},
            "n_chunks": len(retriever.chunks),
            "latency_ms": sum(latencies) / len(latencies),
        }
    return results


def _fmt_table(results: dict, top_k: int) -> str:
    header = f"| 構成 | チャンク数 | MRR | Recall@1 | Recall@{top_k} | nDCG@{top_k} | 遅延 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for name, r in results.items():
        o = r["overall"]
        rows.append(
            f"| {name} | {r['n_chunks']} | {o['mrr']:.3f} | {o['recall@1']:.3f} | "
            f"{o[f'recall@{top_k}']:.3f} | {o[f'ndcg@{top_k}']:.3f} | {r['latency_ms']:.1f}ms |"
        )
    return "\n".join(rows)


def _fmt_by_type(results: dict, top_k: int) -> str:
    types = sorted({t for r in results.values() for t in r["by_type"]})
    rows = ["| 構成 | " + " | ".join(types) + " |", "|---|" + "---:|" * len(types)]
    for name, r in results.items():
        cells = []
        for t in types:
            v = r["by_type"].get(t, {}).get(f"recall@{top_k}")
            cells.append(f"{v:.2f}" if v is not None else "-")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross-encoder", action="store_true", help="クロスエンコーダ構成も評価する")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    results = run(include_cross_encoder=args.cross_encoder, top_k=args.k)

    table = _fmt_table(results, args.k)
    by_type = _fmt_by_type(results, args.k)
    print("\n■ 全体スコア\n")
    print(table)
    print(f"\n■ 質問タイプ別 Recall@{args.k}\n")
    print(by_type)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "retrieval_scores.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / "retrieval_report.md").write_text(
        f"# 検索精度比較レポート\n\n## 全体スコア\n\n{table}\n\n"
        f"## 質問タイプ別 Recall@{args.k}\n\n{by_type}\n",
        encoding="utf-8",
    )
    print(f"\n→ {REPORT_DIR}/retrieval_report.md に保存しました")


if __name__ == "__main__":
    main()
