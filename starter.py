import asyncio
from llama_index.core.agent.workflow import FunctionAgent, BaseWorkflowAgent
from llama_index.llms.ollama import Ollama
from enum import Enum
from pprint import pprint
from llama_index.core.workflow import Context
from typing import List, Sequence, Optional

# Enum for model names
class Model(Enum):
    qwen = "qwen3:4b"
    llama3 = "llama3.1:latest"
    llama3_2 = "llama3.2:1b"
    gemma3 = "gemma3:latest"    # No tool support yet


# TOOLS
# --------------------------------
# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


def converse(query: str) -> str:
    """Useful for general conversation."""
    return query

# AGENT SETUP
# --------------------------------
# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply, converse],
    llm=Ollama(
        model=Model.llama3_2.value,
        request_timeout=360.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    ),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)
ctx = Context(agent)

# HELPERS
# --------------------------------
async def stream_response(agent: FunctionAgent, query: str, ctx: Optional[Context] = None):
    """Stream the agent's response."""
    # print(f"User: {query}")
    workflow_handler = agent.run(query, ctx=ctx)
    print("Agent:", end=" ", flush=True)
    async for chunk in workflow_handler.stream_events():
        # Filter for LLM text generation events
        if hasattr(chunk, 'delta') and chunk.delta:
            print(chunk.delta, end="", flush=True)
        elif hasattr(chunk, 'text') and chunk.text:
            print(chunk.text, end="", flush=True)
    print(f"\n{'-'*20}", flush=True)  # For a newline after streaming

async def start_agent():
    print("Starting agent...")
    await stream_response(agent, "Hello, what can you do?")

async def main():
    # await start_agent()

    # while user_input := input("User: "):
    #     await stream_response(agent, user_input, ctx=ctx)
    await stream_response(agent, input("User: "), ctx=ctx)
    await stream_response(agent, input("User: "), ctx=ctx)
    await stream_response(agent, input("User: "), ctx=ctx)


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())