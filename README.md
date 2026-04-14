# React_Agent
Built a ReAct (Reasoning + Acting) agent using LangGraph's StateGraph that autonomously thinks through problems, calls external tools, observes results, and loops until a final answer is reached. Implemented two custom tools — a Tavily web search tool and a clothing recommendation tool — connected via conditional edges. 

# 🤖 ReAct Agent with LangGraph

## 📌 Overview

This project implements a ReAct (Reasoning + Acting) AI Agent using LangGraph.

The agent follows a structured reasoning loop (Think → Act → Observe) and uses external tools such as web search and recommendation engines to solve multi-step problems.

This project demonstrates how modern AI agents perform reasoning, tool usage, and workflow orchestration.

---

## 🚀 Features

* ReAct-based reasoning agent
* Tool integration (Web Search + Clothing Recommendation)
* LangGraph workflow automation
* Stateful conversation management
* Multi-step reasoning and action execution
* Extensible tool architecture

---

## 🛠 Tech Stack

* Python
* LangGraph
* LangChain
* OpenAI GPT Models
* Tavily Search API

---

## 📂 Project Structure

notebook/react_agent_langgraph.ipynb — Main implementation
tools/custom_tools.py — Custom tools
requirements.txt — Dependencies
screenshots/demo.png — Example output

---

## ▶️ How to Run

Install dependencies:

pip install -r requirements.txt

Run notebook:

jupyter notebook

Open:

react_agent_langgraph.ipynb

Run all cells.

---

## 📸 Demo

(Add screenshot here)

---

## 🎯 Future Improvements

* Add calculator tool
* Add news summarization tool
* Build Streamlit web interface
* Add memory persistence

---

## 👨‍💻 Author

Muhammad Saqib 
GitHub: https://github.com/Saqib00712


