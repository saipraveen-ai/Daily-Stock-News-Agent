# Daily Stock News Agent - Framework Comparison

This directory contains implementations of the same problem using 9 different agent fr## Additional Resources

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions and troubleshooting
- **[DETAILED_FRAMEWORK_COMPARISON.md](DETAILED_FRAMEWORK_COMPARISON.md)** - In-depth technical analysis 
- **[MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md)** - Configure frameworks with different LLM providers (Anthropic, Google, Azure, etc.)
- **[WHY_OPENAI_ONLY.md](WHY_OPENAI_ONLY.md)** - Comprehensive explanation of provider choice strategy
- **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)** - Visual diagrams, class diagrams, and Mermaid charts
- **[visual_comparison.py](visual_comparison.py)** - Generate framework comparison charts
- **[generate_diagrams.py](generate_diagrams.py)** - Generate architectural diagrams and visualizationsks to demonstrate the benefits of using established frameworks over custom implementations.

## Problem Statement
Process Telugu YouTube videos from stock channels → Transcribe content → Analyze for stock insights → Generate comprehensive reports

## Framework Implementations

### 1. 🦜 LangChain
- **File**: `langchain_agent.py`
- **Strengths**: Mature ecosystem, extensive tools, great documentation
- **Best For**: Production applications with complex workflows

### 2. 🤝 CrewAI
- **File**: `crewai_agent.py`
- **Strengths**: Multi-agent collaboration, role-based agents
- **Best For**: Teams of specialized agents working together

### 3. 🏗️ AutoGen
- **File**: `autogen_agent.py`
- **Strengths**: Conversational agents, multi-agent discussions
- **Best For**: Complex reasoning requiring agent collaboration

### 4. 🧠 LangGraph
- **File**: `langgraph_agent.py`
- **Strengths**: State management, complex workflows, visual graphs
- **Best For**: Complex state-driven workflows

### 5. 🎯 PydanticAI
- **File**: `pydanticai_agent.py`
- **Strengths**: Type safety, validation, clean APIs
- **Best For**: Type-safe applications with data validation

### 6. 🚀 Swarm (OpenAI)
- **File**: `swarm_agent.py`
- **Strengths**: Lightweight, OpenAI-native, simple coordination
- **Best For**: Simple multi-agent coordination

### 7. 🔧 Semantic Kernel (Microsoft)
- **File**: `semantic_kernel_agent.py`
- **Strengths**: AI orchestration, plugin architecture, planning
- **Best For**: AI-first applications with Microsoft ecosystem integration

### 8. 🔍 Haystack (Deepset)
- **File**: `haystack_agent.py`
- **Strengths**: Production NLP pipelines, RAG, document processing
- **Best For**: NLP-heavy applications requiring advanced document processing

### 9. 🤖 OpenAI Assistants API (Official)
- **File**: `openai_assistants_agent.py`
- **Strengths**: Persistent threads, built-in tools, stateful memory
- **Best For**: Official OpenAI integration with persistent conversations

## Visual Workflow
```
YouTube Videos → Transcription → Analysis → Report Generation
     ↓              ↓              ↓            ↓
[Video Agent] → [STT Agent] → [Analysis Agent] → [Report Agent]
```

## Comparison Matrix

| Framework | Setup Complexity | Learning Curve | Production Ready | Multi-Agent | Type Safety | Special Features |
|-----------|------------------|----------------|------------------|-------------|-------------|------------------|
| LangChain | Medium | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Tool ecosystem |
| CrewAI | Low | Low | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Role-based agents |
| AutoGen | High | High | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Conversational AI |
| LangGraph | Medium | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | State management |
| PydanticAI | Low | Low | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Type safety |
| Swarm | Very Low | Very Low | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Lightweight |
| Semantic Kernel | Medium | Medium | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | AI orchestration |
| Haystack | Medium | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | NLP pipelines |
| OpenAI Assistants | Low | Low | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Persistent memory |

## Installation

All dependencies are managed in the main `requirements.txt` file. Install the base dependencies:

```bash
# Install base dependencies
pip install -r requirements.txt

# Then uncomment and install specific frameworks you want to test:
# For LangChain:
pip install langchain langchain-openai langchain-community

# For CrewAI:
pip install crewai

# For AutoGen:
pip install pyautogen

# For LangGraph:
pip install langgraph langchain-core

# For PydanticAI:
pip install pydantic-ai pydantic

# For Swarm:
pip install git+https://github.com/openai/swarm.git

# For Semantic Kernel:
pip install semantic-kernel

# For Haystack:
pip install haystack-ai sentence-transformers

# For OpenAI Assistants API:
pip install openai>=1.0.0
```

## Running Examples

Each implementation can be run independently:

```bash
# Set your OpenAI API key first
export OPENAI_API_KEY="your-api-key-here"

# Run specific frameworks
python framework_comparison/langchain_agent.py
python framework_comparison/crewai_agent.py
python framework_comparison/autogen_agent.py
python framework_comparison/langgraph_agent.py
python framework_comparison/pydanticai_agent.py
python framework_comparison/swarm_agent.py
python framework_comparison/semantic_kernel_agent.py
python framework_comparison/haystack_agent.py
python framework_comparison/openai_assistants_agent.py

# Or run the visual comparison
python framework_comparison/visual_comparison.py
```

## Additional Resources

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions and troubleshooting
- **[DETAILED_FRAMEWORK_COMPARISON.md](DETAILED_FRAMEWORK_COMPARISON.md)** - In-depth technical analysis 
- **[MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md)** - Configure frameworks with different LLM providers (Anthropic, Google, Azure, etc.)
- **[visual_comparison.py](visual_comparison.py)** - Generate framework comparison charts

## Why OpenAI for All Frameworks?

All implementations use OpenAI's GPT-4 to ensure **fair framework comparison**:
- Same AI model quality across all implementations
- Focus on framework orchestration differences, not AI model differences  
- Universal support - every framework has excellent OpenAI integration
- Single API key simplifies setup and testing

> 💡 Most frameworks support multiple providers (Anthropic, Google, Azure, local models). See [MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md) for alternative configurations.
