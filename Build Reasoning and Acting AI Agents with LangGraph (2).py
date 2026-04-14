
# 

# # **ReAct: Build Reasoning and Acting AI Agents with LangGraph**
# 

# 
# ## What is ReAct?
# 
# **ReAct** stands for **Reasoning + Acting**. It's a framework that combines:
# 
# 1. **Reasoning**: The agent thinks through problems step by step, maintaining an internal dialogue about what it needs to do.
# 2. **Acting**: The agent can use external tools (search engines, calculators, APIs) to gather information or perform actions.
# 3. **Observing**: The agent processes the results from its actions and incorporates them into its reasoning.
# 
# This creates a powerful loop: **Think → Act → Observe → Think → Act → ...**



# ## Setup & Installation
# 


# In[1]:


get_ipython().system('pip install -U langgraph langchain-openai')


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install langgraph==0.3.34 langchain-openai==0.3.14 langchainhub==0.1.21 langchain==0.3.24 pygraphviz==1.14 langchain-community==0.3.23\n')



# In[3]:


import warnings 
warnings.filterwarnings('ignore')

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
import os
import json

os.environ["TAVILY_API_KEY"] = "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"

# Initialize the Tavily search tool
search = TavilySearchResults()

@tool
def search_tool(query: str):
    """
    Search the web for information using Tavily API.

    :param query: The search query string
    :return: Search results related to the query
    """
    return search.invoke(query)


# ### Theory behind Web Search Tools:
# - Enable real-time information retrieval
# - Overcome the knowledge cutoff limitation of language models
# - Return structured data that the agent can process and reason about
# 
# ### Testing the Search Tool
# 

# In[4]:


search_tool.invoke("What's the weather like in Tokyo today?")


# This test demonstrates how the agent can access current information that wasn't available during training.
# 
# #### 2. Clothing Recommendation Tool
# 

# In[5]:


@tool
def recommend_clothing(weather: str) -> str:
    """
    Returns a clothing recommendation based on the provided weather description.

    This function examines the input string for specific keywords or temperature indicators 
    (e.g., "snow", "freezing", "rain", "85°F") to suggest appropriate attire. It handles 
    common weather conditions like snow, rain, heat, and cold by providing simple and practical 
    clothing advice.

    :param weather: A brief description of the weather (e.g., "Overcast, 64.9°F")
    :return: A string with clothing recommendations suitable for the weather
    """
    weather = weather.lower()
    if "snow" in weather or "freezing" in weather:
        return "Wear a heavy coat, gloves, and boots."
    elif "rain" in weather or "wet" in weather:
        return "Bring a raincoat and waterproof shoes."
    elif "hot" in weather or "85" in weather:
        return "T-shirt, shorts, and sunscreen recommended."
    elif "cold" in weather or "50" in weather:
        return "Wear a warm jacket or sweater."
    else:
        return "A light jacket should be fine."


# **Why this Tool Matters:**
# - Demonstrates domain-specific reasoning
# - Shows how tools can process and interpret data from other tools
# - Illustrates the composability of ReAct systems
# 
# #### Creating the Tool Registry
# 

# In[6]:


tools=[search_tool,recommend_clothing]

tools_by_name={ tool.name:tool for tool in tools}


# 
# ## Setting Up the Language Model
# 
# ### Initializing the AI Model
# 

# In[7]:


from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

model = ChatOpenAI(model="gpt-4o-mini")


# We're using GPT-4o-mini as our reasoning engine. This model will:
# - Analyze user queries
# - Decide which tools to use
# - Process tool results
# - Generate final responses
# 
# ### Creating the System Prompt
# 

# In[8]:


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage,SystemMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful AI assistant that thinks step-by-step and uses tools when needed.

When responding to queries:
1. First, think about what information you need
2. Use available tools if you need current data or specific capabilities  
3. Provide clear, helpful responses based on your reasoning and any tool results

Always explain your thinking process to help users understand your approach.
"""),
    MessagesPlaceholder(variable_name="scratch_pad")
])


# **The System Prompt's Role:**
# - Defines the agent's behavior and personality
# - Establishes the reasoning pattern (think → act → observe)
# - Encourages transparency in the decision-making process
# 
# ### Binding Tools to the Model
# 

# In[9]:


model_react=chat_prompt|model.bind_tools(tools)


# This creates a model that can:
# - Understand when to use tools
# - Generate properly formatted tool calls
# - Process tool results in context
# 
# ## Understanding Agent State
# 
# ### What is Agent State?
# 
# In ReAct, state management is crucial, as the agent must maintain context across multiple reasoning and acting steps.
# 

# In[10]:


from typing import (Annotated,Sequence,TypedDict)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]



# In[11]:


# Example conversation flow:
state: AgentState = {"messages": []}

# append a message using the reducer properly
state["messages"] = add_messages(state["messages"], [HumanMessage(content="Hi")])
print("After greeting:", state["messages"])

# add another message (e.g. a question)
state["messages"] = add_messages(state["messages"], [HumanMessage(content="Weather in NYC?")])
print("After question:", state)


# This demonstrates how the state accumulates context over the conversation.
# 

# ## Manual ReAct Execution (Understanding the Flow)
# 
# Before building the automated graph, let's manually step through a ReAct cycle to understand what happens:
# 
# ### Step 1: Initial Query Processing
# 

# In[12]:


dummy_state: AgentState = {
    "messages": [HumanMessage( "What's the weather like in Zurich, and what should I wear based on the temperature?")]}

response = model_react.invoke({"scratch_pad":dummy_state["messages"]})

dummy_state["messages"]=add_messages(dummy_state["messages"],[response])



# ### Step 2: Tool Execution
# 

# In[13]:


tool_call = response.tool_calls[-1]
print("Tool call:", tool_call)

tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
print("Tool result preview:", tool_result[0]['title'])

tool_message = ToolMessage(
    content=json.dumps(tool_result),
    name=tool_call["name"],
    tool_call_id=tool_call["id"]
)
dummy_state["messages"] = add_messages(dummy_state["messages"], [tool_message])


# **What Happens Here:**
# 1. Extract the tool call from the model's response.
# 2. Execute the tool using the specified arguments.
# 3. Create a ToolMessage containing the results.
# 4. Add the tool result to the conversation state.
# 

# ### Step 3: Processing Results and Next Action
# 

# In[14]:


response = model_react.invoke({"scratch_pad": dummy_state["messages"]})
dummy_state['messages'] = add_messages(dummy_state['messages'], [response])

# check if the model wants to use another tool
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
    tool_message = ToolMessage(
        content=json.dumps(tool_result),
        name=tool_call["name"],
        tool_call_id=tool_call["id"]
    )
    dummy_state['messages'] = add_messages(dummy_state['messages'], [tool_message])


# **What Happens Here:**
# 1. The model processes the search results.
# 2. It realizes it needs to use the clothing recommendation tool.
# 3. It extracts weather information and calls the clothing tool.
# 4. It receives clothing recommendations based on the weather data.
# 

# ### Step 4: Final Response Generation
# 

# In[15]:


response = model_react.invoke({"scratch_pad": dummy_state["messages"]})
print("Final response generated:", response.content is not None)
print("More tools needed:", bool(response.tool_calls))


# **What Happens Here:**
# 1. The model has all necessary information.
# 2. It synthesizes weather data and clothing recommendations.
# 3. It generates a comprehensive response to the user.
# 4. No more tool calls needed—the reasoning cycle is complete.
# 

# ## Automating ReAct with Graphs
# 
# ### Why Use Graphs?
# 
# Manual ReAct execution is educational but impractical for real applications. LangGraph automates this process with a state machine that handles the reasoning loop automatically.
# 
# ### Building the Core Functions
# 
# #### Tool Execution Node
# 

# In[16]:


def tool_node(state: AgentState):
    """Execute all tool calls from the last message in the state."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


# **Function Purpose:**
# - Automatically execute all tool calls from the model
# - Handle multiple simultaneous tool calls
# - Return properly formatted tool messages
# 

# #### Model Invocation Node
# 

# In[17]:


def call_model(state: AgentState):
    """Invoke the model with the current conversation state."""
    response = model_react.invoke({"scratch_pad": state["messages"]})
    return {"messages": [response]}


# **Function Purpose:**
# - Call the ReAct-enabled model
# - Pass the full conversation context
# - Return the model's response (which may include tool calls)
# 
# #### Decision Logic
# 

# In[18]:


def should_continue(state: AgentState):
    """Determine whether to continue with tool use or end the conversation."""
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# **Function Purpose:**
# - Implement the control flow logic
# - Decide whether the agent needs to use more tools
# - Route the conversation to either tool execution or completion
# 
# ### Constructing the State Graph
# 

# In[19]:


from langgraph.graph import StateGraph, END

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Add edges between nodes
workflow.add_edge("tools", "agent")  # After tools, always go back to agent

# Add conditional logic
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",  # If tools needed, go to tools node
        "end": END,          # If done, end the conversation
    },
)

# Set entry point
workflow.set_entry_point("agent")

# Compile the graph
graph = workflow.compile()


# **Graph Structure Explained:**
# 1. **Agent Node**: Where reasoning happens and tool calls are generated.
# 2. **Tools Node**: Where tool execution occurs.
# 3. **Conditional Edge**: Determines whether to continue or finish.
# 4. **Entry Point**: Conversation always starts with the agent reasoning.
# ### Visualizing the Graph
# 

# In[20]:


from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


# This visualization shows the flow: Agent → Decision → Tools → Agent → Decision → End
# 

# ## Running the Complete ReAct Agent
# 
# ### Final Execution
# 

# In[21]:


def print_stream(stream):
    """Helper function for formatting the stream nicely."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [HumanMessage(content="What's the weather like in Zurich, and what should I wear based on the temperature?")]}

print_stream(graph.stream(inputs, stream_mode="values"))



