from .loading import Loader
from .splitting import Splitter 
import config.setting
import os

url = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

if __name__ == "__main__":
    pdf_list = [url[0]]

    # Create Loader
    loader = Loader(source=pdf_list)
    documents = loader.load()

    # Create Splitter
    splitter = Splitter()
    splits = splitter.split_documents(documents)
    print(f"Number of splits: {len(splits)}")
    print(f"Data type of splits: {type(splits)}")

    #
    
    
    



