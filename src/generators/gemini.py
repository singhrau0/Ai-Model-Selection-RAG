import os

from google import genai

from .base import build_prompt


class GeminiGenerator:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def generate(self, question: str, context: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=build_prompt(question, context)
        )
        return (response.text or "The model returned an empty answer.").strip()
