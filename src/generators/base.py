from __future__ import annotations

from typing import Protocol


class Generator(Protocol):
    def generate(self, question: str, context: str) -> str: ...


def build_prompt(question: str, context: str) -> str:
    return f"""You are a senior AI model advisor. Answer only from the supplied context.
Give a direct recommendation first, then concise technical reasons and trade-offs.
If the evidence is insufficient, explicitly say what information is missing.

CONTEXT:
{context}

QUESTION:
{question}
"""
