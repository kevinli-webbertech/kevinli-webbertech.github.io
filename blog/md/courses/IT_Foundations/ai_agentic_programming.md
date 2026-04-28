# AI Agentic Programming

## What is an AI Agent?

An **AI agent** is an autonomous software system that perceives its environment, makes decisions, and takes actions to achieve a specified goal — often without requiring step-by-step human instruction. Unlike a simple prompt-response model, an agent:

- Maintains **memory** of prior interactions
- Has access to **tools** (web search, code execution, APIs, file systems)
- Plans and **reasons** across multiple steps
- **Iterates** until a goal is met or a stopping condition is reached

---

## Core Components of an AI Agent

| Component | Description |
|-----------|-------------|
| **LLM (Brain)** | A large language model (e.g., GPT-4o, Claude, Gemini) that reasons and generates plans |
| **Tools** | External functions the agent can call: web search, calculator, code interpreter, database queries |
| **Memory** | Short-term (in-context) and long-term (vector DB, external storage) retention of facts |
| **Planner** | Decides what action to take next, often using a ReAct or Chain-of-Thought loop |
| **Executor** | Runs the selected action and returns results to the LLM |

---

## Agent Reasoning Patterns

### 1. ReAct (Reason + Act)

The agent alternates between **Thought**, **Action**, and **Observation** steps:

```
Thought: I need to find today's top AI news.
Action: web_search("top AI news today")
Observation: [search results]
Thought: I found relevant results, let me summarize.
Final Answer: Here is a summary of today's top AI news...
```

### 2. Chain-of-Thought (CoT)

The model is prompted to "think step by step" before producing an answer. Useful for math, logic, and planning.

### 3. Plan-and-Execute

The agent first creates a **plan** (list of sub-tasks), then executes each step independently. More robust for long-horizon tasks.

### 4. Reflexion

After each attempt the agent **evaluates** its own output and retries if the result was wrong or incomplete.

---

## Agentic Programming Frameworks

### LangChain / LangGraph

```python
import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
tools = [DuckDuckGoSearchRun()]

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "What is the latest news about AI agents?"})
print(result["output"])
```

**LangGraph** extends LangChain to build **stateful, multi-actor** agent graphs using nodes and edges.

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph.add_node("tools", call_tools)
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})
```

### AutoGen (Microsoft)

AutoGen enables **multi-agent conversations** where agents collaborate, debate, or cross-check each other.

```python
import autogen

import os
assistant = autogen.AssistantAgent("assistant", llm_config={"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")})
user_proxy = autogen.UserProxyAgent("user_proxy", human_input_mode="NEVER",
                                    code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="Write a Python script to scrape stock prices.")
```

### CrewAI

CrewAI organizes agents into **crews** with defined roles and goals, suitable for workflow automation.

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find recent AI breakthroughs",
                   backstory="Expert in ML literature", verbose=True)

task = Task(description="Search for the top 3 AI papers from 2025",
            expected_output="A bullet-point summary", agent=researcher)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
```

### OpenAI Agents SDK

OpenAI's native SDK (2025) provides lightweight primitives for building agentic loops:

```python
from agents import Agent, Runner, tool

@tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 72°F."

agent = Agent(name="WeatherBot", instructions="Answer weather questions.", tools=[get_weather])
result = Runner.run_sync(agent, "What's the weather in New York?")
print(result.final_output)
```

---

## Tool Use and Function Calling

Tools are the "hands" of an agent. They are defined as typed functions and passed to the LLM so it can decide when and how to call them.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run a Python code snippet and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    }
]
```

Common tool categories:

- **Search** – DuckDuckGo, Tavily, Bing Search API
- **Code execution** – Python REPL, Jupyter kernel
- **File I/O** – read/write local or cloud files
- **APIs** – weather, stocks, calendars, databases
- **Browser** – Playwright, Selenium for web automation

---

## Memory in Agents

| Memory Type | Storage | Example |
|-------------|---------|---------|
| In-context | LLM window | Last N messages in chat history |
| External short-term | Redis / SQLite | Session-level key-value store |
| Long-term semantic | Vector DB (FAISS, Chroma, Pinecone) | User facts retrieved by similarity search |
| Episodic | Structured logs | Past task outcomes for reflection |

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory, verbose=True)
chain.predict(input="My name is Alice.")
chain.predict(input="What is my name?")  # Agent remembers "Alice"
```

---

## Multi-Agent Systems

Multiple specialized agents collaborate to solve complex tasks:

```
User Request
     │
     ▼
 Orchestrator Agent
   ├──► Research Agent  ──► web_search, summarize
   ├──► Coder Agent     ──► write_code, run_tests
   └──► Reviewer Agent  ──► check_quality, approve
     │
     ▼
  Final Response
```

**Benefits:**
- Specialization — each agent excels at one thing
- Parallelism — agents can work concurrently
- Fault isolation — one agent failing doesn't crash the system

---

## Building a Simple Agent from Scratch (Python)

```python
import openai, json, os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def search_web(query: str) -> str:
    # Stub — replace with real search API
    return f"Results for '{query}': [Article 1], [Article 2]"

tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the internet for current information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}]

messages = [{"role": "user", "content": "What are the top 3 AI agent frameworks in 2025?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
    )
    msg = response.choices[0].message
    messages.append(msg)

    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = search_web(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
    else:
        print("Agent:", msg.content)
        break
```

---

## Evaluation and Safety

### Evaluation Metrics

- **Task completion rate** – Did the agent achieve the goal?
- **Tool call accuracy** – Were the right tools called with correct arguments?
- **Hallucination rate** – Did the agent fabricate facts instead of using tools?
- **Latency / cost** – How many LLM calls and tokens were consumed?

### Safety Guardrails

- **Input validation** – sanitize user requests before passing to the agent
- **Tool permissions** – restrict which tools an agent can call
- **Output filtering** – check final responses for harmful content
- **Human-in-the-loop (HITL)** – require human approval for high-risk actions (e.g., sending emails, making purchases)
- **Rate limiting** – prevent runaway agentic loops from consuming excessive API tokens

---

## Practical Use Cases

| Domain | Agent Application |
|--------|------------------|
| Software Engineering | Code generation, debugging, PR review |
| Customer Support | Autonomous ticket resolution, FAQ answering |
| Data Analysis | Query generation, chart creation, report writing |
| Research | Literature search, summarization, hypothesis generation |
| DevOps | Infrastructure provisioning, log analysis, alerting |
| Education | Personalized tutoring, assignment grading |

---

## Popular Platforms and Tools (2025)

| Tool / Platform | Description |
|----------------|-------------|
| **LangChain** | Modular framework for building LLM chains and agents |
| **LangGraph** | Graph-based stateful agent orchestration |
| **AutoGen** | Multi-agent conversation framework by Microsoft |
| **CrewAI** | Role-based agent crews for workflow automation |
| **OpenAI Agents SDK** | Lightweight first-party SDK for OpenAI models |
| **Semantic Kernel** | Microsoft's SDK for integrating AI into apps |
| **Haystack** | NLP/search pipeline with agent capabilities |
| **Agno (Phidata)** | Multi-modal agents with memory and storage |

---

## Deep Dive: Major Agentic Frameworks

### LangChain

**Website:** [https://www.langchain.com](https://www.langchain.com)  
**GitHub:** [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)  
**Docs:** [https://python.langchain.com](https://python.langchain.com)

#### What is LangChain?

LangChain is an open-source framework launched in 2022 that makes it easier to build applications powered by large language models. Its core idea is the **chain** — a sequence of components (prompts, LLMs, tools, parsers) that can be composed into complex pipelines.

LangChain provides:
- **Model integrations** — Works with OpenAI, Anthropic, Google, Mistral, Ollama (local), and dozens more
- **Prompt templates** — Reusable, parameterized prompts with variable injection
- **Output parsers** — Structured extraction (JSON, lists, Pydantic models) from LLM responses
- **Retrievers** — Interface to vector stores and document sources for RAG (Retrieval-Augmented Generation)
- **Agents and tools** — ReAct-style agents with built-in tool integrations

#### LangChain Expression Language (LCEL)

LCEL is LangChain's modern declarative syntax for composing chains using the `|` pipe operator:

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "quantum computing"})
print(result)
```

#### RAG Pipeline with LangChain

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load and embed documents
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_texts(
    ["LangChain is a framework for LLM apps.", "FAISS is a vector similarity search library."],
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# Build RAG chain
llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_template(
    "Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {question}"
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is LangChain?"))
```

#### When to Use LangChain

| Use Case | Good Fit? |
|----------|-----------|
| RAG / document Q&A | ✅ Excellent |
| Simple LLM chains | ✅ Excellent |
| Complex stateful agents | ⚠️ Use LangGraph instead |
| Multi-agent systems | ⚠️ Use LangGraph instead |
| Production APIs | ✅ With LangServe |

---

### LangGraph

**Website:** [https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)  
**GitHub:** [https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)  
**Docs:** [https://langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)

#### What is LangGraph?

LangGraph is an extension of LangChain (by the same team) that models agentic workflows as **directed graphs** — a set of nodes (functions) connected by edges (transitions). It was designed to overcome the limitations of linear chains when building:

- **Multi-step agents** that loop, branch, or retry
- **Human-in-the-loop** workflows requiring approval gates
- **Multi-agent systems** where agents hand off work to one another
- **Persistent, stateful conversations** that survive across sessions

The key insight: rather than running straight from A → B → C, real agents need cycles — the ability to observe a result, decide what to do next, and loop back.

#### Core Concepts

| Concept | Description |
|---------|-------------|
| **State** | A typed dictionary shared across all nodes; each node reads and writes state |
| **Node** | A Python function that takes state and returns updated state |
| **Edge** | A transition between nodes; can be unconditional or conditional |
| **Conditional Edge** | A function that decides which node to go to next based on state |
| **Checkpointer** | Persists state to disk/DB, enabling long-running and resumable workflows |

#### Full LangGraph Agent Example

```python
import os
import json
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# --- Define state ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- Define tools ---
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol."""
    prices = {"AAPL": "$189.50", "GOOGL": "$175.20", "MSFT": "$415.80"}
    return prices.get(ticker.upper(), f"No data for {ticker}")

tools = [get_stock_price]
tools_by_name = {t.name: t for t in tools}

# --- Set up LLM ---
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
).bind_tools(tools)

# --- Define nodes ---
def call_llm(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    results = []
    for tc in last_message.tool_calls:
        tool_fn = tools_by_name[tc["name"]]
        output = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

# --- Build graph ---
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", call_tools)
graph.set_entry_point("llm")
graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")

app = graph.compile()

# --- Run ---
result = app.invoke({"messages": [HumanMessage(content="What is the stock price of AAPL?")]})
print(result["messages"][-1].content)
```

#### LangGraph vs LangChain

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Execution model | Linear / sequential | Graph with cycles |
| Loops & branching | Limited | Native |
| State management | Manual | Built-in typed state |
| Human-in-the-loop | Workaround needed | First-class support |
| Multi-agent | Difficult | Native |
| Best for | RAG, simple chains | Complex agents, workflows |

---

### AutoGen (Microsoft)

**Website:** [https://microsoft.github.io/autogen](https://microsoft.github.io/autogen)  
**GitHub:** [https://github.com/microsoft/autogen](https://github.com/microsoft/autogen)  
**Docs:** [https://microsoft.github.io/autogen/docs/Getting-Started](https://microsoft.github.io/autogen/docs/Getting-Started)

#### What is AutoGen?

AutoGen is an open-source framework developed by **Microsoft Research** that enables building multi-agent applications through **conversational AI patterns**. Its core model: agents talk to each other in structured conversations to collaboratively solve tasks.

AutoGen's strengths:
- **Multi-agent dialogue** — agents send messages back and forth until a task is resolved
- **Code execution** — a built-in `UserProxyAgent` can execute code locally and feed results back
- **Flexible human involvement** — set `human_input_mode` to `ALWAYS`, `NEVER`, or `TERMINATE`
- **Group chat** — multiple agents in a single conversation, with a manager routing messages
- **Model agnostic** — supports OpenAI, Azure OpenAI, Gemini, Anthropic, and local models

#### AutoGen Agent Types

| Agent Type | Role |
|-----------|------|
| `AssistantAgent` | LLM-backed agent that generates responses and code |
| `UserProxyAgent` | Acts on behalf of the human; can execute code automatically |
| `GroupChatManager` | Routes messages to the right agent in a multi-agent group chat |
| `ConversableAgent` | Base class for custom agents with full control |

#### Two-Agent Code Generation Example

```python
import os
import autogen

config_list = [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]

assistant = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="You are an expert Python developer. Write clean, well-commented code."
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    code_execution_config={
        "work_dir": "coding_workspace",
        "use_docker": False
    }
)

user_proxy.initiate_chat(
    assistant,
    message="Write and run a Python script that fetches the current Bitcoin price from a public API and prints it."
)
```

#### Group Chat with Multiple Agents

```python
import os
import autogen

config_list = [{"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}]
llm_config = {"config_list": config_list}

planner = autogen.AssistantAgent("Planner", llm_config=llm_config,
    system_message="Break down tasks into clear steps.")
coder = autogen.AssistantAgent("Coder", llm_config=llm_config,
    system_message="Write Python code to implement the plan.")
reviewer = autogen.AssistantAgent("Reviewer", llm_config=llm_config,
    system_message="Review the code for bugs and improvements. Say TERMINATE when done.")
user_proxy = autogen.UserProxyAgent("User", human_input_mode="NEVER",
    code_execution_config={"work_dir": "workspace", "use_docker": False})

groupchat = autogen.GroupChat(
    agents=[user_proxy, planner, coder, reviewer],
    messages=[],
    max_round=12
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="Build a CSV parser that reads sales data and outputs a summary report.")
```

#### AutoGen v0.4 (AgentChat API)

AutoGen v0.4 introduced a redesigned API (`autogen_agentchat`):

```python
import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

agent1 = AssistantAgent("Writer", model_client=model_client,
    system_message="Write a short blog post on the given topic.")
agent2 = AssistantAgent("Editor", model_client=model_client,
    system_message="Edit the blog post for clarity and grammar.")

team = RoundRobinGroupChat([agent1, agent2], max_turns=4)

asyncio.run(Console(team.run_stream(task="The future of AI agents in education")))
```

---

### CrewAI

**Website:** [https://www.crewai.com](https://www.crewai.com)  
**GitHub:** [https://github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)  
**Docs:** [https://docs.crewai.com](https://docs.crewai.com)

#### What is CrewAI?

CrewAI is an open-source Python framework that structures AI agents as a **crew of collaborating workers** — each agent has a defined **role**, **goal**, and **backstory**, and is assigned specific **tasks** within a coordinated **workflow**.

Inspired by how human teams operate, CrewAI is built for:
- **Role-based agent design** — agents are specialists (Researcher, Writer, Analyst, etc.)
- **Sequential and parallel workflows** — tasks run in order or concurrently
- **Tool assignment** — each agent gets exactly the tools it needs
- **Delegation** — agents can hand off sub-tasks to teammates
- **Memory and caching** — agents remember context within and across runs

#### CrewAI Core Concepts

| Concept | Description |
|---------|-------------|
| `Agent` | An AI worker with a role, goal, backstory, and optional tools |
| `Task` | A specific piece of work assigned to an agent with expected output |
| `Crew` | A group of agents working together on a set of tasks |
| `Process` | Execution strategy: `sequential` (default) or `hierarchical` (manager delegates) |
| `Tool` | A function an agent can call — built-in tools or custom ones |

#### Research & Writing Crew Example

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()  # Requires SERPER_API_KEY env var

# --- Define Agents ---
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive and accurate information on the assigned topic",
    backstory="You are an expert researcher with 10 years of experience in technology journalism.",
    tools=[search_tool],
    verbose=True,
    llm="gpt-4o",        # Uses OPENAI_API_KEY from environment
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging, well-structured articles based on research findings",
    backstory="You are a skilled tech writer who translates complex topics into readable content.",
    verbose=True,
    llm="gpt-4o",
)

editor = Agent(
    role="Editor",
    goal="Polish content for grammar, clarity, and style",
    backstory="You are a detail-oriented editor who ensures publication-quality output.",
    verbose=True,
    llm="gpt-4o",
)

# --- Define Tasks ---
research_task = Task(
    description="Research the current state of AI agents in 2025. Cover key frameworks, use cases, and trends.",
    expected_output="A detailed research report with key facts, statistics, and insights.",
    agent=researcher
)

writing_task = Task(
    description="Write a 600-word blog post based on the research report.",
    expected_output="A well-structured blog post with introduction, body, and conclusion.",
    agent=writer,
    context=[research_task]   # Receives output from research_task
)

editing_task = Task(
    description="Edit the blog post for grammar, readability, and flow.",
    expected_output="A polished, publication-ready blog post.",
    agent=editor,
    context=[writing_task]
)

# --- Assemble and Run Crew ---
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
print(result.raw)
```

#### Hierarchical Process (Manager Pattern)

In hierarchical mode, a manager LLM automatically delegates tasks to the most appropriate agent:

```python
import os
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o",   # Manager uses OPENAI_API_KEY from environment
    verbose=True
)

result = crew.kickoff()
```

#### Custom Tools in CrewAI

```python
import os
from crewai.tools import BaseTool
from openai import OpenAI

class SummarizeTool(BaseTool):
    name: str = "summarize_text"
    description: str = "Summarizes a long piece of text into bullet points."

    def _run(self, text: str) -> str:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following text as bullet points."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content

summarize_tool = SummarizeTool()
```

#### Framework Comparison

| Feature | LangChain | LangGraph | AutoGen | CrewAI |
|---------|-----------|-----------|---------|--------|
| Primary Abstraction | Chains / Runnables | Graphs / Nodes | Conversational Agents | Role-based Crews |
| Multi-agent | Limited | ✅ Native | ✅ Native | ✅ Native |
| State management | Manual | ✅ Typed state | Message history | Task context |
| Human-in-the-loop | Workaround | ✅ First-class | ✅ `human_input_mode` | ✅ Human agent |
| Code execution | Via tools | Via tools | ✅ Built-in sandbox | Via tools |
| Learning curve | Medium | High | Medium | Low |
| Best for | RAG, pipelines | Complex stateful agents | Code generation, debate | Workflow automation |
| License | MIT | MIT | MIT | MIT |
| Website | [langchain.com](https://www.langchain.com) | [langchain.com/langgraph](https://www.langchain.com/langgraph) | [microsoft.github.io/autogen](https://microsoft.github.io/autogen) | [crewai.com](https://www.crewai.com) |

---

## Summary

AI agentic programming represents a paradigm shift from single-turn LLM calls to **autonomous, goal-directed systems** that can plan, use tools, remember context, and collaborate. Key takeaways:

1. Agents = LLM + Tools + Memory + Planning loop
2. Choose a framework (LangGraph, AutoGen, CrewAI) based on your complexity needs
3. Design tools carefully — they are the primary way agents interact with the real world
4. Always include safety guardrails and human-in-the-loop checkpoints for production systems
5. Evaluate agents rigorously: task success rate, tool accuracy, and cost efficiency

---

*Last updated: April 2026*
