"""Step 1: 依存ライブラリ最小構成の RAG（全体像を1ファイルで掴む）

    ./.venv/bin/python steps/step1_naive_rag.py

ragkit を一切 import せず、標準ライブラリ + numpy だけで
RAG の5工程を上から下まで書き下している。
フレームワークを使うと隠れてしまう「実際には何が起きているのか」を確認するのが目的。

    1. 読み込み   : 文書をテキストとして取得
    2. 分割       : 固定長で切る（最も素朴なやり方）
    3. ベクトル化 : 語彙を次元とする TF ベクトル
    4. 検索       : cosine 類似度で上位k件
    5. 生成       : 検索結果をプロンプトに詰める

この実装の限界は、そのまま Step 2 以降の改善テーマになる。
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

CORPUS_DIR = Path("data/corpus")
CHUNK_SIZE = 300
TOP_K = 3


# ── 1. 読み込み ──────────────────────────────────────────
def load_texts() -> list[tuple[str, str]]:
    out = []
    for p in sorted(CORPUS_DIR.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        body = re.sub(r"\A---.*?---\s*", "", raw, flags=re.DOTALL)  # frontmatter除去
        body = re.sub(r"^> .*$\n?", "", body, flags=re.MULTILINE)  # ダミー文書の注意書きを除去
        out.append((p.stem, body))
    return out


# ── 2. 分割 ─────────────────────────────────────────────
def split(texts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """固定長で機械的に切る。文の途中でも見出しの途中でも構わず切断する。"""
    chunks = []
    for doc_id, body in texts:
        for i in range(0, len(body), CHUNK_SIZE):
            piece = body[i : i + CHUNK_SIZE].strip()
            if piece:
                chunks.append((doc_id, piece))
    return chunks


# ── 3. ベクトル化 ────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """日本語には語の区切りが無いので、文字bi-gramで代用する。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return [t[i : i + 2] for i in range(len(t) - 1)]


def vectorize(chunks: list[tuple[str, str]]):
    counters = [Counter(tokenize(c)) for _, c in chunks]
    vocab = {t: i for i, t in enumerate(sorted({t for c in counters for t in c}))}

    def to_vec(counter: Counter) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float32)
        for t, f in counter.items():
            if t in vocab:
                v[vocab[t]] = f
        n = np.linalg.norm(v)
        return v / n if n else v  # L2正規化 → 内積がそのまま cosine 類似度になる

    matrix = np.vstack([to_vec(c) for c in counters])
    return matrix, (lambda q: to_vec(Counter(tokenize(q))))


# ── 4. 検索 ─────────────────────────────────────────────
def search(matrix: np.ndarray, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = matrix @ qvec  # 全チャンクとの内積を一度に計算
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


# ── 5. 生成（ここでは LLM を呼ばずプロンプトを組み立てるだけ）──
def build_prompt(question: str, hits: list[str]) -> str:
    ctx = "\n\n---\n\n".join(f"[{i}] {h}" for i, h in enumerate(hits, 1))
    return (
        "以下の参考文書のみを根拠に、引用番号を付けて回答してください。\n"
        "文書に無い場合は「記載がありません」と答えてください。\n\n"
        f"# 参考文書\n{ctx}\n\n# 質問\n{question}"
    )


def main() -> None:
    chunks = split(load_texts())
    matrix, embed_query = vectorize(chunks)
    print(f"{len(chunks)} チャンク / 語彙 {matrix.shape[1]} 次元\n")

    for q in [
        "在宅勤務の手当はいくらですか",
        "家で働くとお金はもらえますか",  # 上と同じ意味。語彙が違うだけ
    ]:
        print(f"Q: {q}")
        hits = search(matrix, embed_query(q), TOP_K)
        for rank, (i, s) in enumerate(hits, 1):
            doc_id, text = chunks[i]
            print(f"  {rank}. {s:.3f} [{doc_id}] {text[:56].replace(chr(10), ' ')}…")
        print()

    print("=" * 70)
    print("この素朴な実装の問題点（Step 2 以降で1つずつ潰していく）")
    print("=" * 70)
    print("""
1. 固定長分割が意味のまとまりを壊す
   → 表の途中や文の途中で切れ、チャンク単体では意味が通らなくなる
   → Step 2: 見出し構造に沿った分割へ

2. 語彙一致でしかない
   → 「在宅勤務手当」と「家で働くとお金」が全く別物として扱われる
   → 上の2つの質問で結果が変わることを確認できる
   → Step 3: 意味を捉えるニューラル埋め込みへ

3. 高頻度な文字bi-gram（「して」「こと」）がスコアを支配する
   → IDF による重み付けが無いため、内容語が埋もれる
   → Step 3: TF-IDF / BM25 へ

4. 正しく動いているか誰にも分からない
   → 「なんとなく良さそう」以外の判断材料が無い
   → Step 5: 評価セットと指標で数値化する（最重要）
""")


if __name__ == "__main__":
    main()
