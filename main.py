import config.setting
from src.chains.rag_chain import build_rag_chain
import parser

def main():
    rag = build_rag_chain()

    question = "LangChain là gì?"
    result = rag.invoke({"query": question})

    print(result)

if __name__ == "__main__":
    main()
    