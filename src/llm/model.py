from langchain_openai import ChatOpenAI
from config.settings import MODEL_NAME

def get_llm():
    return ChatOpenAI(model=MODEL_NAME)