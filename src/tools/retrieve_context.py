from langchain.tools import tool

def make_retrieve_context_tool(vector_store, reranker=None):

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""

        # Retrieve relevant documents from the vector store
        retrieved_docs = vector_store.similarity_search(query, k=20)

        # Reranker 
        if reranker:
            retrieved_docs = reranker.rerank(query, retrieved_docs)

        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context
