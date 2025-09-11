import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama


# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=Ollama(
        model="llama3.1:latest",
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    ),
    system_prompt="You are a helpful assistant. Use the available tools if necessary. Converse normally otherwise.",
)
from llama_index.core.workflow import Context

# create context
ctx = Context(agent)



async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response), end=f"\n\n{'-'*20}\n\n")

    # run agent with context
    response = await agent.run("My name is firstname is Logan", ctx=ctx)
    print(str(response), end=f"\n\n{'-'*20}\n\n")
    response = await agent.run("My name is lastname is Paul", ctx=ctx)
    print(str(response), end=f"\n\n{'-'*20}\n\n")
    response = await agent.run("What is my full name?", ctx=ctx)
    print(str(response), end=f"\n\n{'-'*20}\n\n")

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())