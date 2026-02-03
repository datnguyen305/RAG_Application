import numpy as np
import config.setting
import os

from .loading import Loader
from .splitting import Splitter 
from .embeddings import Embedded
from .vector_store import FaissVectorStore
from .tools.retrieve_context import make_retrieve_context_tool
from .middleware.prompt_with_context import make_prompt_with_context
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


url = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]


if __name__ == "__main__":
    pdf_list = [url[0]]
    tools = []
    middleware = []

    # Create Loader
    loader_instance = Loader(source=pdf_list)
    documents = loader_instance.load()

    # Create Splitter
    splitter_instance = Splitter()
    all_splits = splitter_instance.split_documents(documents)

    # Embedding 
    embedder_instance = Embedded()
    print(f"Create Embed instance: {embedder_instance.model_name}")

    # Create Vector Store
    vector_store = FaissVectorStore(embeddings=embedder_instance.model)
    document_ids = vector_store.add_documents(all_splits)

    # Create LLM Model
    model = ChatOpenAI(temperature=0, model=os.getenv("CHAT_MODEL"))
    print(f"Create LLM Model: {model.model_name}")

    # Create Tool
    retrieve_context = make_retrieve_context_tool(vector_store)
    tools.append(retrieve_context)

    # Create Prompt Middleware
    middleware.append(make_prompt_with_context(vector_store))

    # Create 
    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware
    )

    query = "What is task decomposition?"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    
    



