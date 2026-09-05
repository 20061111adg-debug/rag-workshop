"""形態素解析器を使わない日本語トークナイズ。

英語の BM25 は空白で単語分割できるが、日本語には語の境界が無い。
MeCab / Sudachi は精度が高い一方、辞書の導入と環境差が再現性の壁になる。

ここでは「文字 N-gram」方式を採る。
  - ラテン文字・数字の連なり : そのまま1トークン（VPN, 5,000, KintaiOne）
  - カタカナの連なり         : 全体を1トークン + 文字bi-gram（リモートワーク）
  - 漢字・ひらがな           : 文字bi-gram（在宅勤 → 在宅, 宅勤）

利点: 辞書不要・未知語に強い・実装が短い
欠点: 索引が膨らむ・「東京都」と「京都」のような偶発一致が起きる

なお「する」「こと」等の機能語は除去していない。
高頻度語は BM25 の IDF が自動的に無力化するため、
ストップワード辞書を手で持つ必要はない（これも IDF の役割を理解する良い題材）。
"""
from __future__ import annotations

import re
import unicodedata

# 文字種ごとの連なりを取り出す
LATIN_NUM = re.compile(r"[a-z0-9][a-z0-9_\-.,/#@+]*[a-z0-9]|[a-z0-9]")
KATAKANA = re.compile(r"[゠-ヿㇰ-ㇿ]{2,}")
CJK_KANA = re.compile(r"[぀-ゟ一-鿿々-〆]+")


def normalize(text: str) -> str:
    """NFKC 正規化 + 小文字化。

    全角英数→半角、半角カナ→全角カナ が揃うため、表記ゆれの大半が吸収される。
    「ＶＰＮ」「VPN」「vpn」が同一トークンになるのはこの処理のおかげ。
    """
    return unicodedata.normalize("NFKC", text).lower()


def _bigrams(s: str) -> list[str]:
    if len(s) == 1:
        return [s]
    return [s[i : i + 2] for i in range(len(s) - 1)]


def tokenize(text: str) -> list[str]:
    """テキストをトークン列に変換する。"""
    text = normalize(text)
    tokens: list[str] = []

    for m in LATIN_NUM.finditer(text):
        tokens.append(m.group())

    for m in KATAKANA.finditer(text):
        run = m.group()
        tokens.append(run)  # 語全体（「セキュリティ」）
        tokens.extend(_bigrams(run))  # 部分一致用

    for m in CJK_KANA.finditer(text):
        tokens.extend(_bigrams(m.group()))

    return tokens


HIRAGANA_RE = re.compile(r"\A[぀-ゟー]+\Z")


def is_function_like(token: str) -> bool:
    """ひらがなのみで構成されるトークンか。

    日本語の内容語は漢字・カタカナ・ラテン文字を含むことが多く、
    ひらがなだけの bi-gram（「です」「とは」「して」）は機能語である確率が高い。
    完璧な判定ではないが、辞書を持たずに済む近似として実用的。
    """
    return bool(HIRAGANA_RE.match(token))


def weighted_query_tokens(
    text: str, function_weight: float = 1.0
) -> list[tuple[str, float]]:
    """クエリを (トークン, 重み) の列に変換する。

    function_weight=1.0 : 素朴な実装（全トークン等価）
    function_weight<1.0 : 機能語らしいトークンを減衰させる

    なぜクエリ側だけを減衰させるのか:
    索引側で機能語を削ると文書長 |d| が変わり、BM25 の長さ正規化が歪む。
    クエリ側の重み付けなら、索引を作り直さずに効果を検証できる。
    """
    return [
        (t, function_weight if is_function_like(t) else 1.0) for t in tokenize(text)
    ]
