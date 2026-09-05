"""文書の読み込みと構造化。

RAG の最初の工程。ここで「見出し構造」を保持できるかどうかが、
後段のチャンク品質を大きく左右する。
プレーンテキストとして読むとこの情報は失われる。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Section:
    """`## ` 見出し単位の論理ブロック。評価の gold 単位でもある。"""

    doc_id: str
    heading: str  # 例: "6. 通信費および設備の補助"
    text: str  # 見出し行を含まない本文
    order: int


@dataclass
class Document:
    doc_id: str
    title: str
    category: str
    updated: str
    owner: str
    body: str
    sections: list[Section] = field(default_factory=list)

    @property
    def meta(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "category": self.category,
            "updated": self.updated,
            "owner": self.owner,
        }


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """YAML ライブラリを使わずに `key: value` 形式の frontmatter を読む。"""
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, raw[m.end() :]


def _split_sections(doc_id: str, body: str) -> list[Section]:
    """`## ` 見出しで本文を分割する。

    `# `（H1）はタイトル行なので本文からは除外する。
    見出しが1つも無い文書は、全体を1セクションとして扱う。
    """
    lines = body.splitlines()
    sections: list[Section] = []
    cur_heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if cur_heading is None:
            return
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(doc_id, cur_heading, text, len(sections)))

    for line in lines:
        if line.startswith("## "):
            flush()
            cur_heading = line[3:].strip()
            buf = []
        elif line.startswith("# "):
            continue  # H1 はタイトル
        else:
            buf.append(line)
    flush()

    if not sections:
        text = "\n".join(l for l in lines if not l.startswith("# ")).strip()
        sections = [Section(doc_id, "(no heading)", text, 0)]
    return sections


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    doc_id = meta.get("doc_id") or path.stem
    doc = Document(
        doc_id=doc_id,
        title=meta.get("title", path.stem),
        category=meta.get("category", ""),
        updated=meta.get("updated", ""),
        owner=meta.get("owner", ""),
        body=body.strip(),
    )
    doc.sections = _split_sections(doc_id, body)
    return doc


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    corpus_dir = Path(corpus_dir)
    return [load_document(p) for p in sorted(corpus_dir.glob("*.md"))]
