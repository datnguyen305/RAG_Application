import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


class FaissVectorStore:
    def __init__(self, embeddings, index = "Euclidean"):
        self.embeddings = embeddings
        self.embedding_dim = len(self.embeddings.embed_query("hello world"))
        
        if index == "Euclidean":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif index == "Cosine" or index == "InnerProduct": 
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index == "HNSW":
            self.index = faiss.IndexHNSWFlat(
                self.embedding_dim, 
                32, 
                faiss.METRIC_INNER_PRODUCT
            )
            
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
    
    def add_documents(self, all_splits):
        ids = self.vector_store.add_documents(documents=all_splits)
        return ids
    
    def similarity_search(self, query: str, k: int = 20):
        return self.vector_store.similarity_search(query, k=k)

    def save_local(self, path: str):
        self.vector_store.save_local(path)

    def load_local(path: str, embeddings):
        vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        instance = FaissVectorStore(embeddings)
        instance.vector_store = vector_store
        instance.index = vector_store.index
        return instance