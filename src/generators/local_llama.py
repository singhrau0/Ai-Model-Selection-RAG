import os

import ollama

from .base import build_prompt


class LocalLlamaGenerator:
    """Local Llama through Ollama (no cloud key required)."""

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST"))

    def generate(self, question: str, context: str) -> str:
        response = self.client.generate(
            model=self.model, prompt=build_prompt(question, context)
        )
        return response["response"].strip()
