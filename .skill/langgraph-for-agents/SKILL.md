---
name: langgraph-for-agents
description: Use LangGraph to build agents
---

# LangGraph for Agents

## When to use
- Use this skill when the user asks to build agents or multi-agent systems using LangGraph.

## How to refer
### Integrated Reference Examples
Read the examples in "./references/" to understand common patterns. 
Start with "./references/README.md" for an overview, then read the target file, it will show more details.

### External Resources
[Search]
If the "search" tool is available, you can refine the query keywords and execute the search.

[Browse]
If the "browse" tool is available, you can visit the following three websites:
- LangGraph Official GitHub Repository (https://github.com/langchain-ai/langgraph)
- LangGraph Official Documentation (https://docs.langchain.com/oss/python/langgraph/overview)
- LangChain Official Documentation (https://docs.langchain.com/oss/python/langchain/overview)

[Fetch]
If the "fetch" tool is available, you can retrieve content from the following URL:
- Context-7 LangGraph (https://context7.com/websites/langchain_oss_python_langgraph/llms.txt?tokens=10000)
You may adjust the number of tokens by modifying the `tokens` parameter in the URL. The default value is 10,000.

## Project Structure
For demos or tests, use a single .py file. For production-grade applications, use:
```
├── app/                      
│   ├── api/        # API endpoints
│   ├── backend/    # LangGraph logic
│   └── frontend/   # User interface
├── .env.example
├── requirements.txt
└── README.md
```

## Build Philosophy
- Prefer Native: Check if a tool or integration already exists in LangChain before custom building.
- Single File First: Keep core logic in one file initially to simplify debugging.
- Clean Code: Provide only essential comments and use clear, descriptive variable names.
- Real Data: Use actual API URLs and schemas whenever possible.
