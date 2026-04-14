ReAct AI Agent with LangGraph
This repository demonstrates a ReAct (Reasoning + Acting) agent built with LangGraph and LangChain. The agent combines step‑by‑step reasoning with the ability to use external tools (web search, domain‑specific logic) to answer complex, real‑world queries.

🧠 What is ReAct?
ReAct stands for Reasoning + Acting. It enables LLMs to:

Think – maintain an internal dialogue and break down a problem.

Act – call external tools (search engines, calculators, APIs) to gather information.

Observe – incorporate tool results into the reasoning loop.

This cycle continues until the agent has enough information to give a final answer.

✨ Features
Web search – real‑time information retrieval using Tavily Search API.

Clothing recommendation – domain‑specific tool that maps weather conditions to clothing advice.

Manual ReAct loop – educational step‑by‑step execution of the reasoning–acting flow.

Automated LangGraph – state machine that handles tool calling, message management, and routing.

Streaming output – pretty‑printed messages that show the agent’s thoughts and actions.

📦 Installation
1. Clone the repository
bash
git clone https://github.com/yourusername/react-langgraph-agent.git
cd react-langgraph-agent
2. Install dependencies
bash
pip install -U langgraph langchain-openai langchain-community tavily-python
Alternatively, use the exact versions from the notebook:
langgraph==0.3.34, langchain-openai==0.3.14, langchainhub==0.1.21, langchain==0.3.24, pygraphviz==1.14, langchain-community==0.3.23

3. Set up API keys
You need two API keys:

OpenAI API key – for the GPT‑4o-mini model.

Tavily API key – for web search. Get one here.

bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"
Or set them directly in Python (not recommended for production):

python
import os
os.environ["OPENAI_API_KEY"] = "..."
os.environ["TAVILY_API_KEY"] = "..."
🚀 Usage
You can run the agent either interactively (using the LangGraph compiled graph) or step‑by‑step to understand the internal flow.

Interactive execution (automated graph)
python
from langchain_core.messages import HumanMessage
from your_agent_script import graph, print_stream  # assuming the code is in a module

inputs = {"messages": [HumanMessage(content="What's the weather like in Zurich, and what should I wear based on the temperature?")]}
print_stream(graph.stream(inputs, stream_mode="values"))
The agent will:

Think about what information is needed.

Call the search tool to get Zurich’s weather.

Observe the result and decide to call the clothing recommendation tool.

Generate a final answer with both weather and clothing advice.

Manual loop (educational)
The notebook also contains a manual implementation that shows exactly how the model generates tool calls, how tools are executed, and how the conversation state evolves.

🛠️ How It Works
1. Tools
Two tools are defined:

search_tool – wraps the Tavily search API.

recommend_clothing – maps weather keywords (snow, rain, hot, cold) to clothing suggestions.

All tools are stored in a dictionary tools_by_name for easy lookup.

2. Agent State
The state is a TypedDict containing a messages key with a reducer (add_messages) that automatically appends new messages.

3. Graph Nodes
call_model – invokes the LLM with the system prompt and current conversation history. The model can return tool calls.

tool_node – executes all tool calls from the last AI message and returns ToolMessage objects.

4. Conditional Routing
A should_continue function checks the last message:

If it has tool_calls → go to the tools node.

Otherwise → end the conversation.

5. Graph Construction
python
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", should_continue, {...})
workflow.set_entry_point("agent")
graph = workflow.compile()
The compiled graph automatically runs the ReAct loop until no more tool calls are required.

📝 Example Output
For the query:
“What's the weather like in Zurich, and what should I wear based on the temperature?”

The agent might output:

text
================================ AI Message ================================

I'll search for the current weather in Zurich first.

Tool Calls:
  search_tool (call_abc123)
   Args:
     query: current weather in Zurich
================================ Tool Message ================================

[{'title': 'Zurich Weather - MeteoSwiss', 'content': 'Today: Overcast, 64.9°F (18.3°C)...'}]

================================ AI Message ================================

Now I have the weather: overcast, 64.9°F. I'll recommend clothing based on that.

Tool Calls:
  recommend_clothing (call_def456)
   Args:
     weather: overcast 64.9°F
================================ Tool Message ================================

"A light jacket should be fine."

================================ AI Message ================================

The weather in Zurich is currently overcast with a temperature of 64.9°F (18.3°C).  
A light jacket should be comfortable for this weather.
📁 Repository Structure
text
.
├── README.md                # This file
├── react_agent.ipynb        # Full Jupyter notebook with code and explanations
├── requirements.txt         # Python dependencies
└── .env.example             # Example environment variables file
🔮 Possible Extensions
Add more tools (e.g., calculator, database query, image generator).

Use a different LLM (Claude, Llama) via LangChain integrations.

Persist conversation state across sessions.

Add human‑in‑the‑loop approval for sensitive actions.

📄 License
MIT – feel free to use and adapt for your own projects.

🙏 Acknowledgements
LangGraph – for building stateful, multi‑actor LLM applications.

Tavily – for the search API.

OpenAI – for the GPT‑4o‑mini model.





