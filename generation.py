"""Interactive RAG entry point.

Examples:
    python generation.py --provider local --retrieval graph_hybrid
    python generation.py --provider gemini --retrieval hybrid
"""

import argparse

from dotenv import load_dotenv

from src.generators import GeminiGenerator, LocalLlamaGenerator, OpenAIGenerator
from src.retriever import RetrieverSuite
from src.vectorization import Vectorizer


def make_generator(provider: str):
    return {
        "gemini": GeminiGenerator,
        "openai": OpenAIGenerator,
        "local": LocalLlamaGenerator,
    }[provider]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the AI model knowledge base")
    parser.add_argument("--provider", choices=["gemini", "openai", "local"], default="local")
    parser.add_argument(
        "--retrieval",
        choices=["dense", "sparse", "mmr", "hybrid", "graph", "graph_hybrid"],
        default="graph_hybrid",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    vector_db = Vectorizer().hfembedder()
    retriever = RetrieverSuite(vector_db)
    generator = make_generator(args.provider)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        results = retriever.retrieve(question, k=args.top_k, method=args.retrieval)
        context = "\n\n".join(document.page_content for document in results)
        print(f"\n{args.provider.title()} response:\n{generator.generate(question, context)}")
        print("\nSources:")
        for document in results:
            metadata = document.metadata
            print(f"- {metadata.get('source', 'unknown')} page {metadata.get('page', '-')}")


if __name__ == "__main__":
    main()
