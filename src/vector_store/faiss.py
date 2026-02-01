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
    
    def add_documents(self, all_splits):
        ids = self.vector_store.add_documents(documents=all_splits)
        return ids
    
