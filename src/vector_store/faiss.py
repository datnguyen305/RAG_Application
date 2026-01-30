import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


class FaissVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embedding_dim = len(self.embeddings.embed_query("hello world"))
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
    def add_texts(self, texts, metadatas=None, ids=None):
        return self.vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)

    def save(self, path):
        self.vector_store.save_local(path)

    def load(self, path):
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )