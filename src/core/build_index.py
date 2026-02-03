import os

from ..loading import Loader
from ..splitting import Splitter
from ..embeddings import Embedded
from ..vector_store import FaissVectorStore
DATA_DIR = "data/source"

URLS = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if os.path.isfile(os.path.join(DATA_DIR, f))
]

additional_url = []

if additional_url:
    URLS.extend(additional_url)

def build_index(URLS: list[str]):
    print("Building FAISS index...")
    loader = Loader(source=URLS)
    docs = loader.load()

    splits = Splitter().split_documents(docs)

    embedder = Embedded()
    vs = FaissVectorStore(embeddings=embedder.model, index="HNSW")
    vs.add_documents(splits)

    vs.save_local("data/faiss_index")

if __name__ == "__main__":
    build_index(URLS)