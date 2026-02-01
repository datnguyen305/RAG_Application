from langchain_text_splitters import RecursiveCharacterTextSplitter

class Splitter:
    def __init__(self, splitter_type="recursive"):
        """
        This initializes the splitter 

        Args:
            splitter_type: (str) | type of the splitter, can be "recursive"
        
        Returns:
            Splitter datatype 
        """
        self.splitter_type = splitter_type
        if self.splitter_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
        else:
            raise ValueError("Unsupported splitter type")
    def split_documents(self, documents):
        """
        Docstring for split_documents
        
        Args: 
            documents: (list) | list of documents
        Returns: 
            splits: (list) | list of splitted documents
        """
        if self.splitter_type == "recursive":
            return self.splitter.split_documents(documents)