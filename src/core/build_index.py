from ..loading import Loader
from ..splitting import Splitter
from ..embeddings import Embedded
from ..vector_store import FaissVectorStore

URLS = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

def build_index(URLS: list[str]):
    print("Building FAISS index...")
    loader = Loader(source=URLS)
    docs = loader.load()

    splits = Splitter().split_documents(docs)

    embedder = Embedded()
    vs = FaissVectorStore(embeddings=embedder.model)
    vs.add_documents(splits)

    vs.save_local("data/faiss_index")

if __name__ == "__main__":
    build_index(URLS)