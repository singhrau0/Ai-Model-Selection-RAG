"""Small library-based retriever collection."""

import networkx as nx
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class RetrieverSuite:
    def __init__(self, vector_db, documents=None):
        self.vector_db = vector_db
        self.documents = documents or self.load_documents()
        self.dense_retriever = vector_db.as_retriever()
        self.sparse_retriever = BM25Retriever.from_documents(self.documents)
        self.graph = nx.path_graph(len(self.documents))

    def load_documents(self):
        data = self.vector_db.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(data["documents"], data["metadatas"])
        ]

    def dense(self, query, k=5):
        self.dense_retriever.search_kwargs = {"k": k}
        return self.dense_retriever.invoke(query)

    def sparse(self, query, k=5):
        self.sparse_retriever.k = k
        return self.sparse_retriever.invoke(query)

    def mmr(self, query, k=5):
        retriever = self.vector_db.as_retriever(
            search_type="mmr", search_kwargs={"k": k}
        )
        return retriever.invoke(query)

    def hybrid(self, query, k=5):
        self.dense_retriever.search_kwargs = {"k": k}
        self.sparse_retriever.k = k
        retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.5, 0.5],
        )
        return retriever.invoke(query)[:k]

    def graph_search(self, query, k=5, hops=1):
        seeds = self.dense(query, k)
        indexes = {doc.page_content: index for index, doc in enumerate(self.documents)}
        nodes = []

        for document in seeds:
            index = indexes.get(document.page_content)
            if index is not None:
                nodes.extend(nx.ego_graph(self.graph, index, radius=hops).nodes)

        return [self.documents[index] for index in dict.fromkeys(nodes)][:k]

    def graph_hybrid(self, query, k=5):
        documents = self.hybrid(query, k) + self.graph_search(query, k)
        unique = {document.page_content: document for document in documents}
        return list(unique.values())[:k]

    def retrieve(self, query, k=5, method="hybrid"):
        methods = {
            "dense": self.dense,
            "sparse": self.sparse,
            "mmr": self.mmr,
            "hybrid": self.hybrid,
            "graph": self.graph_search,
            "graph_hybrid": self.graph_hybrid,
        }
        return methods[method](query, k)


class mainretriever:
    def Topkretriever(self, question, vector_db, TOP_K):
        documents = vector_db.as_retriever(search_kwargs={"k": TOP_K}).invoke(question)
        return "\n\n".join(document.page_content for document in documents)
