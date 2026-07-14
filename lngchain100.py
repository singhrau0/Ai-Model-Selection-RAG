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


# Load API key
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing.")


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

print("Documents loaded:", len(documents))


# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# Create or load vector database
if Path(CHROMA_DIR).exists():
    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
else:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


# Create retriever
retriever = vector_db.as_retriever(
    search_kwargs={"k": TOP_K}
)


# Create Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY
)


def generate_answer(question, context):

    prompt = f"""
Answer the question using only the provided context.

If the answer is not available, say:
"The provided context does not contain enough information."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=QA_MODEL,
        contents=prompt,
    )

    return response.text


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

    answer = generate_answer(question, context)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

   