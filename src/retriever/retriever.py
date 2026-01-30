from src.vectorstore.db import get_vectorstore

def get_retriever():
    db = get_vectorstore()
    return db.as_retriever(search_kwargs={"k": 3})