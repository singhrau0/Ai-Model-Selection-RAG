import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "gemini-3.1-flash-lite"

TOP_K = 3


# Load Gemini API key
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from the .env file."
    )


# Load documents
documents = []

documents += DirectoryLoader(
    DATA_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
).load()

documents += DirectoryLoader(
    DATA_DIR,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader,
).load()

documents += DirectoryLoader(
    DATA_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
).load()

print(f"Loaded documents/pages: {len(documents)}")


# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = text_splitter.split_documents(documents)

print(f"Created chunks: {len(chunks)}")


# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# Create or load Chroma DB
if Path(CHROMA_DIR).exists():

    print("Loading existing Chroma DB...")

    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
    )

else:

    print("Creating new Chroma DB...")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )


# Retriever
retriever = vector_db.as_retriever(
    search_kwargs={"k": TOP_K}
)


# Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


# Generate answer using Gemini
def generate_answer(question: str, context: str) -> str:

    prompt = f"""
You are a question-answering assistant for a RAG application.

Answer the question using only the provided context.

Rules:
1. Give a direct answer first.
2. Explain the answer using details from the context.
3. Write around 4 to 8 sentences.
4. Do not use outside knowledge.
5. Do not invent facts.
6. If the answer is unavailable, say:
"The provided context does not contain enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    try:

        response = client.models.generate_content(
            model=QA_MODEL,
            contents=prompt,
        )

        if not response.text:
            return "The model returned an empty answer."

        return response.text.strip()

    except Exception as error:

        return (
            f"Gemini inference failed: "
            f"{type(error).__name__}: {error}"
        )


# Question-answer loop
while True:

    question = input("\nQuestion: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Stopped.")
        break

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    print("\nRetrieved context:")
    print(context)

    answer = generate_answer(
        question=question,
        context=context,
    )

    print("\nAnswer:")
    print(answer)

    print("\nSources used:")

    for i, doc in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get(
            "source",
            "unknown source",
        )

        page = doc.metadata.get(
            "page",
            "",
        )

        print(f"{i}. {source} page {page}")

    print("-" * 60)