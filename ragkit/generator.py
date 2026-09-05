"""回答生成レイヤ。

RAG の "G"。ここは意図的に差し替え可能にしてある。

  ExtractiveGenerator : LLM を一切呼ばない。検索結果を整形して返すだけ
  ClaudeGenerator     : Claude API を呼ぶ

なぜ分けるか。
1. APIキーが無くても、パイプライン全体を動かして検索精度を評価できる
2. 会社ごとに使えるモデルが違う。差し替え点を1箇所に閉じ込めておく
3. **RAG の品質の大部分は検索側で決まる**ことを、実装の形として示すため

生成器を差し替えても検索精度は1ミリも変わらない。
逆に検索が正解を拾えていなければ、どんなモデルでも正answerは出せない。
"""
from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PROMPT = """あなたは社内文書に基づいて質問に答えるアシスタントです。

厳守事項:
1. 提供された「参考文書」に書かれている内容のみを根拠に回答すること。
2. 回答の各主張の末尾に、根拠とした文書の番号を [1] の形式で必ず付けること。
3. 参考文書に答えが含まれていない場合は、推測で補わず
   「提供された文書には記載がありません」と明確に述べること。
4. 文書間で内容が食い違う場合は、両方を提示し、更新日が新しい方を優先すると述べること。
5. 数値・期限・金額は文書の表記をそのまま引用すること。言い換えて丸めないこと。
"""


def build_prompt(question: str, chunks: list) -> str:
    """検索結果を LLM 入力に整形する。

    設計上の要点:
    - 各チャンクに通し番号と出典メタデータを付ける（引用を可能にするため）
    - 更新日を含める（規程類は「どちらが新しいか」が答えの一部になる）
    - 質問を末尾に置く。長い文脈では末尾が最も参照されやすい
    """
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(
            f"[{i}] 出典: {c.meta.get('title', c.title)} / {c.primary_heading}"
            f"（更新日: {c.meta.get('updated', '不明')}, 管掌: {c.meta.get('owner', '不明')}）\n"
            f"{c.text}"
        )
    context = "\n\n---\n\n".join(blocks) if blocks else "(該当する文書が見つかりませんでした)"
    return f"# 参考文書\n\n{context}\n\n# 質問\n\n{question}"


@dataclass
class Answer:
    text: str
    citations: list
    generator: str


class ExtractiveGenerator:
    """LLM を使わない抽出型の回答。

    検索結果をそのまま提示する。流暢さは無いが、
    - ハルシネーションが構造的に起こり得ない
    - APIコストがゼロ
    - 検索結果の良し悪しが生で見える（デバッグに最適）
    という利点がある。RAG の検証段階ではむしろこちらが有用。
    """

    name = "extractive"

    def generate(self, question: str, chunks: list) -> Answer:
        if not chunks:
            return Answer("該当する文書が見つかりませんでした。", [], self.name)
        lines = [f"「{question}」に関連する記述は以下のとおりです。\n"]
        for i, c in enumerate(chunks, 1):
            body = c.text.split("\n", 1)[-1].strip()  # 文脈ヘッダ行を除く
            lines.append(
                f"[{i}] {c.meta.get('title', c.title)} / {c.primary_heading}\n{body}\n"
            )
        return Answer("\n".join(lines), list(chunks), self.name)


class ClaudeGenerator:
    """Claude API による回答生成。

    認証は環境変数 ANTHROPIC_API_KEY、または `ant auth login` のプロファイルから
    自動解決される（引数でキーを渡す必要はない）。
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "low",
        max_tokens: int = 4096,
        use_refusal_fallback: bool = True,
    ):
        import anthropic  # 遅延 import: APIを使わない構成では未インストールでよい

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        # 社内文書の抽出的Q&Aは難問ではないため effort は low で十分。
        # 複数文書の突き合わせや矛盾検出をさせるなら high 以上に上げる。
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_refusal_fallback = use_refusal_fallback
        self.name = f"claude({model},effort={effort})"

    def _params(self, question: str, chunks: list) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": build_prompt(question, chunks)}],
        }

    def generate(self, question: str, chunks: list) -> Answer:
        params = self._params(question, chunks)

        if self.use_refusal_fallback:
            # ポリシー判断で拒否された場合に、同一リクエスト内で別モデルに引き継がせる。
            # ベータ未有効の組織では 400 になるため、その場合は通常呼び出しに落とす。
            try:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **params,
                )
            except self.anthropic.BadRequestError:
                self.use_refusal_fallback = False
                resp = self.client.messages.create(**params)
        else:
            resp = self.client.messages.create(**params)

        if resp.stop_reason == "refusal":
            return Answer("(モデルが回答を拒否しました)", list(chunks), self.name)

        # content は TextBlock / ThinkingBlock などの混在リスト。type を見て取り出す
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Answer(text, list(chunks), self.name)
