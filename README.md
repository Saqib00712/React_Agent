# ReAct Agent with LangGraph
> Reasoning + Acting AI agent that thinks through problems, calls external tools, observes results, and loops until a final answer is reached — built with LangGraph and GPT-4o-mini.

---

## What is ReAct?

ReAct (**Re**asoning + **Act**ing) is an agent framework that combines three phases in a continuous loop:

Think → Act → Observe → Think → Act → ...

| Phase | What happens |
|-------|-------------|
| **Reasoning** | The agent thinks step-by-step about what it needs to do |
| **Acting** | The agent calls external tools (web search, APIs, calculators) |
| **Observing** | The agent reads tool results and updates its reasoning |

---

## Demo

**Query:** *"What's the weather like in Zurich, and what should I wear?"*
Agent → calls search_tool("weather in Zurich")
→ observes: "Overcast, 64.9°F"
→ calls recommend_clothing("Overcast, 64.9°F")
→ observes: "Wear a warm jacket or sweater"
→ Final answer: "It's 64.9°F and overcast in Zurich. I recommend wearing a warm jacket..."

## Architecture
User Query
↓
[Agent Node] ← reasoning with GPT-4o-mini
↓
Should continue?
├── YES → [Tools Node] → execute tool → back to Agent
└── NO  → Final Answer


Built with **LangGraph StateGraph** — a state machine that automates the reasoning loop.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| `search_tool` | Real-time web search via Tavily API — overcomes LLM knowledge cutoff |
| `recommend_clothing` | Domain-specific reasoning based on weather conditions |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3.34-teal?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3.24-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/GPT--4o--mini-OpenAI-black?style=flat-square)

- **LangGraph** — StateGraph for automated agent loop
- **LangChain** — Tool binding, prompt templates, message types
- **GPT-4o-mini** — Reasoning engine
- **Tavily Search API** — Real-time web search
- **Python** — Core language

---

## Project Structure
React_Agent/
│
├── react_agent.ipynb       # Main notebook with full implementation
├── requirements.txt        # All dependencies
├── .env.example            # Required API keys (template)
└── README.md

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Saqib00712/React_Agent.git
cd React_Agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
```bash
cp .env.example .env
```
Edit `.env` and add your keys:

OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here

### 4. Run the notebook
```bash
jupyter notebook react_agent.ipynb
```

---

## Key Concepts Covered

- **AgentState** with LangGraph's `add_messages` reducer for context management
- **Tool binding** — attaching custom tools to a language model
- **Conditional edges** — routing logic between agent and tool nodes
- **Manual vs automated** ReAct loop — understanding both approaches
- **StateGraph compilation** — building production-ready agent pipelines

---

## What I Learned

Building this project deepened my understanding of how LLM agents manage multi-step reasoning. The key insight is that the agent doesn't "know" the answer upfront — it discovers it by iterating through the Think → Act → Observe loop, which mirrors how humans solve problems with incomplete information.

---

## Related Certifications

This project was built as part of the IBM **Building AI Agents and Agentic Workflows Specialization** on Coursera.

[![IBM Badge](https://img.shields.io/badge/IBM-AI%20Agents%20Specialization-blue?style=flat-square)](https://www.credly.com/users/muhammad-saqib.361f9b8c)

---

## Author

**Muhammad Saqib**
- GitHub: [@Saqib00712](https://github.com/Saqib00712)
- LinkedIn: [muhammad-saqib](https://www.linkedin.com/in/muhammad-saqib-68b9b3374/)
- Email: saqibkhosa649@gmail.com
- Credly: [15x IBM Certified](https://www.credly.com/users/muhammad-saqib.361f9b8c)




