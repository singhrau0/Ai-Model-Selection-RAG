from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Vectorizer:
    def __init__(self, model_name=EMBEDDING_MODEL, persist_directory=CHROMA_DIR):
        self.model_name = model_name
        self.persist_directory = persist_directory

    def dense_vectors(self, texts):
        """Return dense vectors directly for experiments or inspection."""
        return self.embedding_model().embed_documents(list(texts))

    def sparse_vectors(self, texts):
        """Return a fitted TF-IDF vectorizer and its efficient CSR sparse matrix."""
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20_000)
        return vectorizer, vectorizer.fit_transform(list(texts))

    def embedding_model(self):
        return HuggingFaceEmbeddings(model_name=self.model_name)

    def hfembedder(self, chunks=None, rebuild=False):
        embedding_model = self.embedding_model()
        path = Path(self.persist_directory)
        if path.exists() and not rebuild:
            print("Loading existing Chroma DB...")

            vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding_model,
            )
        else:
            print("Creating new Chroma DB...")
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=self.persist_directory,
            )

        print("Chrom DB is Created")
        return vector_db
