from .document_loading import Loader
import config.setting
import os

url = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

if __name__ == "__main__":
    pdf_list = ["documents.pdf", url[0]]
    loader = Loader(source=pdf_list)
    documents = loader.load()
    print(documents)
    



