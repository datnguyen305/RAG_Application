from langchain_openai import OpenAIEmbeddings

def get_embedding():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )