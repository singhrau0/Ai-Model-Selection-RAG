import os

from openai import OpenAI

from .base import build_prompt


class OpenAIGenerator:
    """OpenAI Responses API adapter."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, question: str, context: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=build_prompt(question, context),
        )
        return response.output_text.strip()
