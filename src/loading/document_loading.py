import bs4
import config.setting
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader
)


bs4_strainer_testing = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))


class Loader:
    def __init__(self, source: list[str] = None, bs4_strainer: bs4.SoupStrainer =None) -> None:
        """
        This initializes the loader 

        Args:
            source: list(str) | source of the document
            loader_type: (str) | type of the loader, can be "web", "pdf", "txt"
            bs4_strainer: (bs4.SoupStrainer) | strainer for web loader

        Returns:
            Loader datatype 
        """
        if source is None:
            self.source = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
        elif isinstance(source, str):
            self.source = [source]
        else:
            self.source = source
        
        if bs4_strainer:
            self.bs4_strainer = bs4_strainer
        else: 
            self.bs4_strainer = bs4_strainer_testing

    def _get_loader(self, source=None):
        if source.startswith("http"):
            if not isinstance(source, list):
                source = [source]
            return WebBaseLoader(
                web_paths=(source),
                bs_kwargs={"parse_only": self.bs4_strainer} if self.bs4_strainer else {})
            

        elif source.endswith(".pdf"):
            return PyPDFLoader(source)

        elif source.endswith(".txt"):
            return TextLoader(source, encoding="utf-8")

        else:
            raise ValueError("Unsupported loader type")

    def load(self):
        """
        Docstring for load
        
        Args: 
            None
        Returns: 
            documents: (list) | list of documents
        """
        documents = []

        for i, s in enumerate(self.source):
            loader = self._get_loader(s)
            documents.extend(loader.load())

        return documents
    