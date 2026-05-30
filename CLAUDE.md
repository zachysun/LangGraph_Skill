# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **skill** for Claude Code and other AI coding platforms — not a traditional application. When loaded, it instructs the AI on how to build LangGraph/LangChain agents, using the reference examples in `.skill/langgraph-for-agents/references/` as implementation templates.

## Architecture

```
.skill/langgraph-for-agents/
├── SKILL.md                              # Skill definition — when to use, design process, build philosophy
└── references/                           # Reference code organized by complexity level
    ├── README.md                         # Index of all reference files with descriptions
    ├── openai_call_llm.py                # Raw OpenAI API (non-streaming)
    ├── openai_call_llm_stream.py         # Raw OpenAI API (streaming)
    ├── openai_structured_output.py       # OpenAI structured output
    ├── langchain_chatmodel.py            # LangChain basic chat model call
    ├── langchain_chatmodel_multi_msg.py  # Multiple message forms (system, human, AI)
    ├── langchain_chatmodel_rag.py        # Simple RAG flow
    ├── langchain_chatmodel_integ_tool.py # Chat model with built-in integration tool
    ├── langchain_chatmodel_custom_tool.py # Chat model with custom tool definition
    ├── langchain_react_agent.py          # ReAct agent via create_agent
    ├── langchain_react_agent_integ_tool.py # ReAct agent with integration tool
    ├── langgraph_workflow_one_node.py    # Minimal single-node graph
    ├── langgraph_workflow_prompt_chaining.py # Sequential node chaining
    ├── langgraph_workflow_parallelization.py # Parallel nodes with result joining
    ├── langgraph_workflow_routing.py     # Conditional routing between nodes
    ├── langgraph_workflow_orch_worker.py # Orchestrator + multiple workers pattern
    ├── langgraph_workflow_eval_optim.py  # Evaluator-optimizer loop pattern
    ├── langgraph_human_in_the_loop.py    # Human interruption and feedback
    ├── langgraph_workflow_st_mem.py      # Short-term memory (in-session state)
    ├── langgraph_workflow_lt_mem.py      # Long-term memory (cross-session persistence)
    └── langgraph_streaming.py            # Streaming graph output
```

`.local_references/` mirrors the same files for local development convenience.

## How the skill works

1. When invoked, Claude Code reads `SKILL.md` for the design process:
   - **Step 1:** Determine system level (single-agent vs multi-agent)
   - **Step 2:** Choose framework (LangGraph for stateful workflows, LangChain for standard tool-calling agents)
   - **Step 3:** Design the specific implementation using the reference examples as templates

2. The skill directs Claude Code to use the reference files as implementation templates — starting from the relevant pattern and adapting it to the user's specific use case.

3. For external resources, the skill instructs searching LangGraph/LangChain docs and the Context7 API for up-to-date framework documentation.

## Key design principles (from SKILL.md)

- **Prefer Native:** Check if a tool or integration already exists in LangChain before custom building.
- **Single File First:** Keep core logic in one file initially to simplify debugging.
- **Single-file for demos, multi-file structure for production:** For production apps, use `app/api/`, `app/backend/`, `app/frontend/` separation with `.env.example` and `requirements.txt`.
