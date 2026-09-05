"""チャンク分割戦略。

RAG の精度に最も効く割に、最も雑に扱われがちな工程。
「何文字で切るか」ではなく「意味のまとまりをどう保つか」が本質。

本モジュールは4戦略を用意し、同一の評価セットで比較できるようにしている。
  fixed            : 固定長。文書構造を完全に無視する素朴な実装
  fixed_overlap    : 固定長 + オーバーラップ。境界の分断を緩和
  heading          : 見出し単位。長いセクションのみ再分割
  heading_context  : 見出し単位 + 文脈ヘッダ付与（Contextual Chunking）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import Document

# 日本語の文末（。！？）と改行を文の境界とみなす
SENT_SPLIT_RE = re.compile(r"(?<=[。！？])|(?<=\n)")

# チャンクが「そのセクションを含む」とみなす最小の重なり文字数。
# 評価時の正解判定に用いる。
MIN_SECTION_OVERLAP = 50


@dataclass
class Chunk:
    """検索とLLM入力の最小単位。

    text と embed_text を分けているのが要点。
    ベクトル化するテキストと、LLMに渡すテキストは同じである必要はない。
    """

    chunk_id: str
    doc_id: str
    title: str
    text: str  # LLM に渡す本文
    embed_text: str  # ベクトル化・BM25索引の対象テキスト
    headings: tuple[str, ...]  # このチャンクが跨ぐセクション見出し（評価の正解判定に使用）
    meta: dict = field(default_factory=dict)

    @property
    def primary_heading(self) -> str:
        return self.headings[0] if self.headings else "(unknown)"

    def display(self, width: int = 80) -> str:
        head = f"[{self.doc_id} / {self.primary_heading}]"
        body = self.text.replace("\n", " ")
        return f"{head} {body[:width]}{'…' if len(body) > width else ''}"


def _flatten(doc: Document) -> tuple[str, list[tuple[str, int, int]]]:
    """文書を1本の文字列に平坦化し、各セクションの文字範囲を返す。

    固定長分割したチャンクが「どの見出しに属するか」を後から復元するために必要。
    """
    parts: list[str] = []
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for sec in doc.sections:
        block = f"{sec.heading}\n{sec.text}\n\n"
        parts.append(block)
        spans.append((sec.heading, cursor, cursor + len(block)))
        cursor += len(block)
    return "".join(parts), spans


def _headings_for_span(
    spans: list[tuple[str, int, int]], start: int, end: int
) -> tuple[str, ...]:
    """文字範囲 [start, end) と十分に重なるセクション見出しを返す。"""
    hits: list[tuple[int, str]] = []
    for heading, s, e in spans:
        overlap = min(end, e) - max(start, s)
        if overlap >= min(MIN_SECTION_OVERLAP, e - s):
            hits.append((overlap, heading))
    hits.sort(reverse=True)
    return tuple(h for _, h in hits)


def _split_sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT_RE.split(text) if s and s.strip()]


def chunk_fixed(
    docs: list[Document], size: int = 300, overlap: int = 0
) -> list[Chunk]:
    """固定長スライディングウィンドウ。

    実装は最も簡単だが、表の途中や文の途中で平気で切断する。
    overlap=0 が「素朴なRAG」、overlap>0 が定番の緩和策。
    """
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    for doc in docs:
        flat, spans = _flatten(doc)
        for i, start in enumerate(range(0, max(1, len(flat)), step)):
            text = flat[start : start + size]
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#f{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text,
                    embed_text=text,
                    headings=_headings_for_span(spans, start, start + len(text)),
                    meta=doc.meta,
                )
            )
    return chunks


def chunk_by_heading(
    docs: list[Document],
    max_chars: int = 600,
    overlap_sentences: int = 1,
    add_context_header: bool = False,
) -> list[Chunk]:
    """見出し単位で分割し、長すぎるセクションのみ文境界で再分割する。

    add_context_header=True のとき、埋め込み対象テキストの先頭に
    「文書タイトル > 見出し」を付与する（Contextual Chunking）。
    セクション後半のチャンクは主語や文脈を失いがちなので、
    このヘッダが検索時の意味的な手掛かりになる。
    """
    chunks: list[Chunk] = []
    for doc in docs:
        for sec in doc.sections:
            pieces: list[str]
            if len(sec.text) <= max_chars:
                pieces = [sec.text]
            else:
                pieces, buf = [], ""
                sents = _split_sentences(sec.text)
                idx = 0
                while idx < len(sents):
                    if buf and len(buf) + len(sents[idx]) > max_chars:
                        pieces.append(buf)
                        # 直前の n 文を次チャンクの先頭に引き継ぐ
                        back = _split_sentences(buf)[-overlap_sentences:] if overlap_sentences else []
                        buf = "".join(back)
                    buf += sents[idx]
                    idx += 1
                if buf.strip():
                    pieces.append(buf)

            for j, piece in enumerate(pieces):
                header = f"{doc.title} > {sec.heading}\n"
                body = piece.strip()
                text = header + body  # LLM には常に文脈付きで渡す
                embed_text = text if add_context_header else body
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{sec.order}-{j}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=text,
                        embed_text=embed_text,
                        headings=(sec.heading,),
                        meta=doc.meta,
                    )
                )
    return chunks


STRATEGIES = {
    "fixed":           lambda d: chunk_fixed(d, size=300, overlap=0),
    "fixed_overlap":   lambda d: chunk_fixed(d, size=300, overlap=100),
    "heading":         lambda d: chunk_by_heading(d, add_context_header=False),
    "heading_context": lambda d: chunk_by_heading(d, add_context_header=True),
}
