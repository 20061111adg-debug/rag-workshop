"""numpy の環境依存な挙動を吸収する層。

macOS の Accelerate BLAS + numpy 2.0.x の組み合わせでは、
matmul が浮動小数点例外フラグを誤って立てるため
"divide by zero / overflow / invalid encountered in matmul" が出る。

ゼロも無限大も含まない乱数入力でも発生し、計算結果自体は正しい
（float64 での手計算と最大誤差 2e-5、非有限値なし）ことを検証済み。
numpy 2.1 以降で修正されているが、2.1 は Python 3.10 以上を要求するため、
Python 3.9 環境では抑制する。

抑制範囲をこの関数内に限定しているのが要点。
グローバルに np.seterr(all="ignore") してしまうと、
本物の数値異常（正規化前のゼロ除算など）まで見逃す。
"""
from __future__ import annotations

import numpy as np


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """偽陽性の FP 警告を抑制した行列積。"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b
