# Framework Implementations

Individual AI framework implementations of the Daily Stock News Agent.

## 📋 Implementation Files

| Framework | File | Pattern | Complexity |
|-----------|------|---------|------------|
| **OpenAI Assistants** | `openai_assistants_agent.py` | Thread-based | Simple |
| **LangChain** | `langchain_agent.py` | Sequential chains | Complex |
| **CrewAI** | `crewai_agent.py` | Multi-agent teams | Medium |
| **AutoGen** | `autogen_agent.py` | Conversations | Medium |
| **LangGraph** | `langgraph_agent.py` | State machines | Complex |
| **PydanticAI** | `pydanticai_agent.py` | Type-safe | Simple |
| **Swarm** | `swarm_agent.py` | Handoff patterns | Simple |
| **Semantic Kernel** | `semantic_kernel_agent.py` | Enterprise plugins | Complex |
| **Haystack** | `haystack_agent.py` | NLP pipelines | Complex |

## 🚀 Usage

Each implementation follows the same interface:

```python
# Example usage pattern (all frameworks)
import asyncio
from <framework>_agent import <Framework>StockNewsSystem

async def main():
    # Initialize with OpenAI API key
    system = <Framework>StockNewsSystem(api_key="your_openai_key")
    
    # Process daily news
    result = await system.process_daily_news(
        channels=["moneypurse", "daytradertelugu"],
        date="2025-01-22"
    )
    
    if result["success"]:
        print(f"✅ Processing completed!")
        print(f"📄 Report: {result['final_report']}")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Quick Test

Test any framework implementation:

```bash
# Set your API key
export OPENAI_API_KEY="your_key_here"

# Run specific framework
python openai_assistants_agent.py
python langchain_agent.py
python crewai_agent.py
# ... etc
```

## 📊 Common Interface

All implementations provide:
- **Same method signatures** for easy comparison
- **Consistent return formats** for analysis
- **Similar error handling** patterns
- **Comparable logging** and monitoring

---

*Choose the implementation that best fits your use case and requirements.*
