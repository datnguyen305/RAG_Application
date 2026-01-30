from langchain.chains import RetrievalQA
from src.llm.model import get_llm
from src.retriever.retriever import get_retriever

def build_rag_chain():
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=get_retriever()
    )
