# Framework Setup Guide

This guide explains what API keys and setup steps are needed to run each of the 9 AI framework implementations.

## 🔑 Required API Keys

**ALL frameworks require OpenAI API Key:**
- Get your API key from: https://platform.openai.com/api-keys
- Set environment variable: `OPENAI_API_KEY=your_api_key_here`

### Why Only OpenAI API Key?
- **Fair Comparison**: Same GPT-4 model across all frameworks = comparing framework capabilities, not AI models
- **Universal Support**: Every framework has excellent OpenAI integration
- **Consistency**: Eliminates model performance as a variable in framework comparison
- **Simplicity**: One API key for all 9 implementations

> 💡 **Note**: Most frameworks support multiple providers (Anthropic, Google, Azure, etc.). We use OpenAI for consistent comparison. See [MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md) for alternative configurations.

## 📋 Quick Setup

### 1. Environment Variables
Create a `.env` file in the `framework_comparison/` directory:

```bash
# .env file  
OPENAI_API_KEY=your_openai_api_key_here
```

**OR** copy the example file:
```bash
cd framework_comparison/
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 2. Install Dependencies

**Core Project Dependencies:**
```bash
pip install -r requirements.txt
```

**Framework Dependencies (choose based on compatibility):**

⚠️ **Important**: Some frameworks have conflicting dependencies. Choose one installation approach:

**Option A: Modern LangChain Ecosystem**
```bash
pip install "langchain>=0.1.0" "langchain-openai>=0.1.0" "langchain-community>=0.0.1" "langgraph>=0.1.0" "langchain-core>=0.1.0"
```

**Option B: CrewAI Compatible** 
```bash
pip install "crewai>=0.1.0" "langchain==0.0.354" "langchain-openai==0.0.2"
```

**Option C: Standalone Frameworks**
```bash
pip install "pyautogen>=0.2.0" "pydantic-ai>=0.1.0" "pydantic>=2.0.0" "semantic-kernel>=0.4.0" "haystack-ai>=2.0.0"
```

**Option D: Maximum Compatibility (excludes CrewAI)**
```bash
pip install "langchain>=0.1.0" "langchain-openai>=0.1.0" "langchain-community>=0.0.1" "langgraph>=0.1.0" "langchain-core>=0.1.0" "pyautogen>=0.2.0" "pydantic-ai>=0.1.0" "pydantic>=2.0.0" "semantic-kernel>=0.4.0" "haystack-ai>=2.0.0"
```

### 3. Run Any Framework
```bash
cd framework_comparison/implementations/
python langchain_agent.py      # LangChain implementation
python crewai_agent.py         # CrewAI implementation  
python autogen_agent.py        # AutoGen implementation
python langgraph_agent.py      # LangGraph implementation
python pydanticai_agent.py     # PydanticAI implementation
python swarm_agent.py          # OpenAI Swarm implementation
python semantic_kernel_agent.py # Semantic Kernel implementation
python haystack_agent.py       # Haystack implementation
python openai_assistants_agent.py # OpenAI Assistants API implementation
```

## 🔧 Framework-Specific Setup Details

### 1. LangChain
- **API Keys**: OpenAI API Key
- **Dependencies**: `langchain>=0.1.0`, `langchain-openai>=0.1.0`, `langchain-community>=0.0.1`
- **Setup Time**: ~2-5 minutes
- **Special Notes**: Most mature ecosystem, extensive tool library

### 2. CrewAI  
- **API Keys**: OpenAI API Key
- **Dependencies**: `crewai>=0.1.0`
- **Setup Time**: ~3-7 minutes
- **Special Notes**: Multi-agent collaboration framework

### 3. AutoGen
- **API Keys**: OpenAI API Key
- **Dependencies**: `pyautogen>=0.2.0`
- **Setup Time**: ~2-5 minutes
- **Special Notes**: Microsoft's multi-agent conversation framework

### 4. LangGraph
- **API Keys**: OpenAI API Key
- **Dependencies**: `langgraph>=0.1.0`, `langchain-core>=0.1.0`
- **Setup Time**: ~3-6 minutes
- **Special Notes**: State machine approach to AI workflows

### 5. PydanticAI
- **API Keys**: OpenAI API Key
- **Dependencies**: `pydantic-ai>=0.1.0`, `pydantic>=2.0.0`
- **Setup Time**: ~2-4 minutes
- **Special Notes**: Type-safe AI framework with Pydantic validation

### 6. OpenAI Swarm
- **API Keys**: OpenAI API Key
- **Dependencies**: `git+https://github.com/openai/swarm.git`
- **Setup Time**: ~3-5 minutes
- **Special Notes**: Lightweight multi-agent orchestration

### 7. Semantic Kernel
- **API Keys**: OpenAI API Key
- **Dependencies**: `semantic-kernel>=0.4.0`
- **Setup Time**: ~5-10 minutes
- **Special Notes**: Microsoft's AI orchestration platform

### 8. Haystack
- **API Keys**: OpenAI API Key
- **Dependencies**: `haystack-ai>=2.0.0`
- **Setup Time**: ~5-12 minutes
- **Special Notes**: NLP pipeline framework, good for RAG applications

### 9. OpenAI Assistants API
- **API Keys**: OpenAI API Key
- **Dependencies**: `openai>=1.3.0`
- **Setup Time**: ~1-3 minutes
- **Special Notes**: OpenAI's official persistent assistant platform

## 🚀 Installation Commands

### One-Command Setup
```bash
# Clone and setup (if needed)
git clone <repository-url>
cd Daily-Stock-News-Agent/framework_comparison

# Install all dependencies
pip install -r requirements.txt

# Set environment variable (Linux/Mac)
export OPENAI_API_KEY="your_api_key_here"

# Or set environment variable (Windows)
set OPENAI_API_KEY=your_api_key_here
```

### Alternative: Python dotenv approach
All frameworks support loading from `.env` file:

```python
from dotenv import load_dotenv
load_dotenv()  # This loads OPENAI_API_KEY from .env file
```

## 📊 Cost Considerations

**OpenAI GPT-4 Pricing (as of 2024):**
- Input tokens: $0.03 per 1K tokens
- Output tokens: $0.06 per 1K tokens

**Estimated costs per framework run:**
- **Light processing** (demo mode): ~$0.10-0.30
- **Full processing** (real videos): ~$2-8 depending on content length
- **Development/Testing**: Budget $10-20/day for active development

**Framework-specific cost factors:**
- **OpenAI Assistants API**: Lowest cost (efficient built-in tools)
- **LangChain/LangGraph**: Medium cost (good optimization)
- **CrewAI/AutoGen**: Higher cost (multiple agent conversations)
- **Swarm**: Low cost (lightweight coordination)
- **PydanticAI**: Low cost (efficient structured outputs)
- **Semantic Kernel**: Medium cost (planning overhead)
- **Haystack**: Medium cost (pipeline processing)

## 🏆 Framework Setup Ranking (by ease)

**Ranked from easiest to most complex setup:**

1. **🥇 OpenAI Assistants API** - 1-3 min setup
   - Just `pip install openai>=1.3.0` + API key
   - Official OpenAI platform, minimal dependencies

2. **🥈 PydanticAI** - 2-4 min setup  
   - `pip install pydantic-ai pydantic>=2.0.0`
   - Clean, type-safe API design

3. **🥉 LangChain** - 2-5 min setup
   - `pip install langchain langchain-openai langchain-community`
   - Mature ecosystem, well-documented

4. **AutoGen** - 2-5 min setup
   - `pip install pyautogen>=0.2.0`
   - Microsoft framework, good documentation

5. **Swarm** - 3-5 min setup
   - `pip install git+https://github.com/openai/swarm.git`
   - Lightweight, requires Git installation

6. **LangGraph** - 3-6 min setup
   - `pip install langgraph langchain-core`
   - State management adds complexity

7. **CrewAI** - 3-7 min setup
   - `pip install crewai>=0.1.0`
   - Multi-agent coordination setup

8. **Semantic Kernel** - 5-10 min setup
   - `pip install semantic-kernel>=0.4.0`
   - Microsoft ecosystem, more configuration

9. **Haystack** - 5-12 min setup
   - `pip install haystack-ai>=2.0.0`
   - Complex NLP pipeline dependencies

## 🔒 Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables or .env files**
3. **Rotate API keys regularly**
4. **Set usage limits in OpenAI dashboard**
5. **Monitor API usage and costs**

## 🐛 Common Issues & Solutions

### Issue: "No module named 'framework_name'"
```bash
# Solution: Install the specific framework
pip install langchain crewai pyautogen langgraph pydantic-ai semantic-kernel haystack-ai
pip install git+https://github.com/openai/swarm.git
```

### Issue: "OpenAI API key not found"
```bash
# Solution: Set environment variable
export OPENAI_API_KEY="your_key_here"

# Or create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Issue: "Module version conflicts"
```bash
# Solution: Use virtual environment
python -m venv framework_env
source framework_env/bin/activate  # Linux/Mac
# framework_env\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Issue: Swarm installation fails
```bash
# Solution: Install directly from GitHub
pip install git+https://github.com/openai/swarm.git

# Alternative: Clone and install locally
git clone https://github.com/openai/swarm.git
cd swarm
pip install -e .
```

## 📖 Framework Selection Guide

**Choose based on your needs:**

- **🚀 Quickest Setup**: OpenAI Assistants API (1-3 min)
- **📚 Most Documentation**: LangChain (mature ecosystem)  
- **👥 Multi-Agent Focus**: CrewAI, AutoGen, Swarm
- **🏗️ Complex Workflows**: LangGraph (state machines)
- **🔒 Type Safety**: PydanticAI (Pydantic validation)
- **🏢 Enterprise**: Semantic Kernel (Microsoft ecosystem)
- **🔍 NLP Pipelines**: Haystack (RAG applications)

## 🔄 Alternative LLM Providers

While all implementations use OpenAI for consistency, most frameworks support multiple providers:

### Quick Provider Examples
```python
# LangChain - Multiple options
from langchain_openai import ChatOpenAI           # OpenAI
from langchain_anthropic import ChatAnthropic     # Anthropic Claude
from langchain_google import ChatGoogleGenerativeAI # Google Gemini

# CrewAI - Provider flexibility
llm = LLM(model="gpt-4", api_key=openai_key)                    # OpenAI
llm = LLM(model="anthropic/claude-3-opus", api_key=claude_key)  # Anthropic
llm = LLM(model="ollama/llama2", base_url="http://localhost:11434") # Local

# PydanticAI - Model abstraction
model = OpenAIModel('gpt-4', api_key=openai_key)      # OpenAI
model = AnthropicModel('claude-3-opus', api_key=key)  # Anthropic
model = GoogleModel('gemini-pro', api_key=key)        # Google
```

### Why OpenAI for This Comparison?
1. **Consistent Baseline**: Same model quality reveals framework differences
2. **Universal Support**: All 9 frameworks have excellent OpenAI integration
3. **Fair Evaluation**: Framework orchestration comparison, not AI model comparison
4. **Practical**: Single API key simplifies setup and testing

> 📖 **Detailed Guide**: See [MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md) for complete multi-provider configurations across all frameworks.

## 🎯 Next Steps

1. **Set your OpenAI API key** in environment variables or `.env` file
2. **Install dependencies** using `pip install -r requirements.txt`
3. **Run the simplest framework first**: `python openai_assistants_agent.py`
4. **Compare different approaches** by running multiple frameworks
5. **Customize for your use case** by modifying the implementations
6. **Explore alternatives** in [MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md) for different AI providers

## 📚 Additional Resources

- **[README.md](README.md)** - Overview and comparison matrix
- **[DETAILED_FRAMEWORK_COMPARISON.md](DETAILED_FRAMEWORK_COMPARISON.md)** - In-depth technical analysis
- **[MULTI_PROVIDER_GUIDE.md](MULTI_PROVIDER_GUIDE.md)** - Alternative LLM provider configurations
- **[WHY_OPENAI_ONLY.md](WHY_OPENAI_ONLY.md)** - Comprehensive explanation of provider choice strategy
- **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)** - Visual diagrams, class diagrams, and Mermaid charts
- **[visual_comparison.py](visual_comparison.py)** - Generate comparison charts
- **[generate_diagrams.py](generate_diagrams.py)** - Generate architectural diagrams and visualizations

All frameworks are ready to run with just an OpenAI API key - no additional setup required!
