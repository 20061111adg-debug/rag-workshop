"""評価指標そのものの検証。

指標が壊れていると、改善しているのか悪化しているのか判断できなくなる。
「評価する側を評価する」テストは省略されがちだが、
nDCG > 1.0 のような破綻は実際に起きる（本プロジェクトでも起きた）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from evaluation.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


@dataclass
class FakeChunk:
    doc_id: str
    headings: tuple


G = {("d1", "s1"), ("d1", "s2")}


def _c(doc, *heads):
    return FakeChunk(doc, tuple(heads))


def test_recall():
    assert recall_at_k([_c("d1", "s1")], G, 5) == 0.5
    assert recall_at_k([_c("d1", "s1"), _c("d1", "s2")], G, 5) == 1.0
    assert recall_at_k([_c("d9", "x")], G, 5) == 0.0
    # 同じセクションを2回引いても再現率は増えない
    assert recall_at_k([_c("d1", "s1"), _c("d1", "s1")], G, 5) == 0.5


def test_mrr():
    assert mrr([_c("d9", "x"), _c("d1", "s1")], G) == 0.5
    assert mrr([_c("d1", "s2")], G) == 1.0
    assert mrr([_c("d9", "x")], G) == 0.0


def test_ndcg_never_exceeds_one():
    """重複ヒットで nDCG が 1.0 を超えないこと（実際に踏んだバグの回帰テスト）。"""
    dup = [_c("d1", "s1")] * 5
    assert ndcg_at_k(dup, G, 5) <= 1.0
    many = [_c("d1", "s1"), _c("d1", "s1"), _c("d1", "s2"), _c("d1", "s2")]
    assert ndcg_at_k(many, G, 5) <= 1.0


def test_ndcg_perfect_and_order():
    perfect = [_c("d1", "s1"), _c("d1", "s2")]
    assert math.isclose(ndcg_at_k(perfect, G, 5), 1.0)
    worse = [_c("d9", "x"), _c("d1", "s1"), _c("d1", "s2")]
    assert ndcg_at_k(worse, G, 5) < 1.0


def test_precision():
    assert precision_at_k([_c("d1", "s1"), _c("d9", "x")], G, 5) == 0.5


def test_unanswerable_is_nan():
    assert math.isnan(recall_at_k([_c("d1", "s1")], set(), 5))
    assert math.isnan(mrr([_c("d1", "s1")], set()))


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"{len(fns)} 件すべて成功")
