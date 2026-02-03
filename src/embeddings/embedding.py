import os
import config.setting
from langchain_openai import OpenAIEmbeddings


class Embedded:
    def __init__(self, model_env: str = "EMBEDDING_MODEL"):
        self.model_name = os.getenv(model_env)
        if not self.model_name:
            raise ValueError(f"Environment variable {model_env} is not set")
        
        if self.model_name: 
            self.model = OpenAIEmbeddings(model=self.model_name)
    def __repr__(self):
        return f"Embedded(model_name='{self.model_name}')"
    
    def embed(self, texts: list[str]):
        return self.model.embed_documents(texts)
