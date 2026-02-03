from langchain.agents.middleware import dynamic_prompt, ModelRequest


def make_prompt_with_context(vector_store):

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            "You are a helpful assistant. If you can't retrieve the info inside the database just say 'I don't know'. Use the following context in your response:"
            f"\n\n{docs_content}"
        )

        return system_message

    return prompt_with_context
