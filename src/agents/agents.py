from langchain.agents import create_agent

class RAGAgent:
    def __init__(self, model, tools, system_prompt: str | None = None):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt or (
            "You have access to tools that retrieve context. "
            "Use them to answer user queries accurately."
        )

        self.agent = self._build_agent()

    def _build_agent(self):
        return create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def invoke(self, query: str):
        return self.agent.invoke({"input": query})
    
    def stream(self, inputs: dict, stream_mode: str = "values"):
        return self.agent.stream(inputs, stream_mode=stream_mode)