# Multi-Provider Configuration Examples

This document shows how each framework can be configured with different LLM providers beyond OpenAI.

## 🔄 Alternative LLM Provider Examples

### 1. LangChain - Multiple Providers
```python
# OpenAI (current)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(anthropic_api_key=api_key, model="claude-3-opus")

# Google Gemini
from langchain_google import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro")

# Azure OpenAI
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key=api_key,
    api_version="2024-02-15-preview"
)
```

### 2. CrewAI - Multi-Provider Support
```python
from crewai import LLM, Agent

# OpenAI (current)
llm = LLM(model="gpt-4", api_key=openai_key)

# Anthropic
llm = LLM(model="anthropic/claude-3-opus", api_key=anthropic_key)

# Google
llm = LLM(model="google/gemini-pro", api_key=google_key)

# Ollama (local)
llm = LLM(model="ollama/llama2", base_url="http://localhost:11434")
```

### 3. AutoGen - Provider Flexibility
```python
# OpenAI (current)
config = {
    "model": "gpt-4",
    "api_key": openai_key,
    "base_url": "https://api.openai.com/v1"
}

# Azure OpenAI
config = {
    "model": "gpt-4",
    "api_key": azure_key,
    "base_url": "https://your-resource.openai.azure.com/",
    "api_version": "2024-02-15-preview"
}

# Local LLM via OpenAI-compatible API
config = {
    "model": "llama2",
    "api_key": "not-needed",
    "base_url": "http://localhost:8000/v1"
}
```

### 4. PydanticAI - Model Providers
```python
from pydantic_ai import Agent
from pydantic_ai.models import OpenAIModel, AnthropicModel, GoogleModel

# OpenAI (current)
model = OpenAIModel('gpt-4', api_key=openai_key)

# Anthropic
model = AnthropicModel('claude-3-opus', api_key=anthropic_key)

# Google
model = GoogleModel('gemini-pro', api_key=google_key)

agent = Agent(model=model)
```

### 5. Haystack - Generator Flexibility
```python
# OpenAI (current)
from haystack.components.generators import OpenAIGenerator
generator = OpenAIGenerator(api_key=openai_key, model="gpt-4")

# Anthropic
from haystack.components.generators import AnthropicGenerator
generator = AnthropicGenerator(api_key=anthropic_key, model="claude-3-opus")

# Google
from haystack.components.generators import GoogleAIGeminiGenerator
generator = GoogleAIGeminiGenerator(api_key=google_key, model="gemini-pro")

# Hugging Face
from haystack.components.generators import HuggingFaceAPIGenerator
generator = HuggingFaceAPIGenerator(api_token=hf_token, model="microsoft/DialoGPT-large")
```

### 6. Semantic Kernel - Service Configuration
```python
import semantic_kernel as sk

# OpenAI (current)
kernel.add_text_completion_service(
    "openai",
    OpenAIChatCompletion("gpt-4", api_key=openai_key)
)

# Azure OpenAI
kernel.add_text_completion_service(
    "azure_openai", 
    AzureChatCompletion(
        deployment_name="gpt-4",
        endpoint="https://your-resource.openai.azure.com/",
        api_key=azure_key
    )
)

# Hugging Face
kernel.add_text_completion_service(
    "huggingface",
    HuggingFaceTextCompletion("microsoft/DialoGPT-large", api_key=hf_key)
)
```

## 🎯 Why OpenAI for This Comparison?

### Technical Reasons
1. **Consistent Baseline**: Same model quality across all frameworks
2. **Feature Completeness**: Function calling, structured outputs, streaming
3. **Reliability**: 99.9% uptime, mature API
4. **Performance**: Fast inference, good throughput

### Practical Reasons
1. **Universal Support**: Every framework has excellent OpenAI integration
2. **Documentation**: Best-documented provider across frameworks
3. **Cost Transparency**: Clear, predictable pricing
4. **Developer Experience**: Excellent tooling and debugging

### Comparison Fairness
- **Apples-to-Apples**: Framework orchestration comparison, not model comparison
- **Eliminates Variables**: Same AI capability reveals framework differences
- **Focus on Architecture**: Compare agents vs chains vs graphs vs pipelines
- **Real-World Relevance**: OpenAI is the most widely used enterprise AI provider

### Cost & Accessibility
- **Reasonable Pricing**: $0.03 input/$0.06 output per 1K tokens for GPT-4
- **No Enterprise Barriers**: Single API key, no complex enterprise setup
- **Predictable Costs**: Clear token-based pricing model
- **Global Availability**: Works worldwide without regional restrictions
- **Reduced Variables**: Eliminates model performance as a variable
- **Clear Attribution**: Differences come from framework design, not AI capability

## 💡 Multi-Provider Implementation Strategy

If you wanted to support multiple providers, here's the pattern:

```python
class UniversalStockNewsAgent:
    def __init__(self, provider_config: dict):
        self.provider = provider_config["provider"]  # "openai", "anthropic", "google"
        self.api_key = provider_config["api_key"]
        self.model = provider_config.get("model", self._default_model())
        
        self.llm = self._create_llm()
    
    def _default_model(self):
        defaults = {
            "openai": "gpt-4",
            "anthropic": "claude-3-opus", 
            "google": "gemini-pro",
            "azure": "gpt-4"
        }
        return defaults[self.provider]
    
    def _create_llm(self):
        if self.provider == "openai":
            return ChatOpenAI(openai_api_key=self.api_key, model=self.model)
        elif self.provider == "anthropic":
            return ChatAnthropic(anthropic_api_key=self.api_key, model=self.model)
        elif self.provider == "google":
            return ChatGoogleGenerativeAI(google_api_key=self.api_key, model=self.model)
        # ... etc
```

## 🔄 Easy Provider Switching

To run with different providers, you'd just change:

```bash
# OpenAI (current)
export OPENAI_API_KEY="sk-..."
python langchain_agent.py

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python langchain_agent_anthropic.py

# Google
export GOOGLE_API_KEY="AIza..."
python langchain_agent_google.py
```

## 🎭 The Real Comparison Value

Using OpenAI across all frameworks reveals:

1. **Framework Orchestration Differences**: How each handles multi-step workflows
2. **Developer Experience**: Setup complexity, debugging, maintenance
3. **Architecture Patterns**: Agents vs chains vs graphs vs pipelines
4. **Flexibility**: How easily can you modify the workflow?
5. **Production Readiness**: Error handling, monitoring, scaling

The **framework comparison** is the valuable insight, not the AI model comparison!

Would you like me to create alternative implementations using different providers for any specific framework?
