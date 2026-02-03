from fastapi import FastAPI
from src.core.rag_agent import create_rag_agent

app = FastAPI()
agent = create_rag_agent()

@app.post("/chat")
def chat(query: str):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return {"answer": result["messages"][-1].content}

@app.get("/")
def health():
    return {"status": "RAG API is running"}