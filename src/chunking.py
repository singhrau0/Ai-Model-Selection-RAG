"""Common chunking methods using LangChain splitters."""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class Chunker:
    def _add_ids(self, chunks):
        for number, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = str(number)
        print(f"Created chunks: {len(chunks)}")
        return chunks

    def fixed_chunking(self, documents, chunk_size=500):
        """Split text into fixed character-sized chunks."""
        splitter = CharacterTextSplitter(
            separator="", chunk_size=chunk_size, chunk_overlap=0
        )
        return self._add_ids(splitter.split_documents(documents))

    def paragraph_chunking(self, documents, chunk_size=1000):
        """Prefer paragraph boundaries, then line and sentence boundaries."""
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=chunk_size,
            chunk_overlap=0,
        )
        return self._add_ids(splitter.split_documents(documents))

    def overlap_chunking(self, documents, chunk_size=500, chunk_overlap=100):
        """Recursive chunks with overlap for continuity between chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return self._add_ids(splitter.split_documents(documents))

    def semantic_chunking(self, documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Split where embedding similarity indicates a topic change."""
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        return self._add_ids(splitter.split_documents(documents))

    def recursiveoverlap(self, documents):
        """Backward-compatible name used by scripts/testretrieval.py."""
        return self.overlap_chunking(documents)
