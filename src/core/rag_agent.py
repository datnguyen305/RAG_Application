import os
from ..embeddings import Embedded
from ..vector_store import FaissVectorStore
from ..tools.retrieve_context import make_retrieve_context_tool
from ..middleware.prompt_with_context import make_prompt_with_context
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

def create_rag_agent():
    embedder = Embedded()
    vector_store = FaissVectorStore.load_local(
        "data/faiss_index",
        embeddings=embedder.model
    )

    tools = [make_retrieve_context_tool(vector_store)]
    middleware = [make_prompt_with_context(vector_store)]

    model = ChatOpenAI(
        temperature=0,
        model=os.getenv("CHAT_MODEL")
    )

    return create_agent(
        model=model,
        tools=tools,
        middleware=middleware
    )
