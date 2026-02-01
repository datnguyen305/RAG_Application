import config.setting
from src.chains.rag_chain import build_rag_chain
import argparse

def main():
    rag = build_rag_chain()

    question = "LangChain là gì?"
    result = rag.invoke({"query": question})

    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--reward', action='store', default='rouge-l', help='reward function for RL')

    args = parser.parse_args()