# 🤖 Daily Stock News Agent

*Autonomous agent that processes daily Telugu YouTube stock market videos and generates comprehensive English summary reports*

## 🎯 Project Overview

This project features an **autonomous agent architecture** that intelligently processes Telugu stock market analysis videos from YouTube channels and generates structured English reports:

1. **🤖 Autonomous Agent**: Natural language request processing with intelligent tool orchestration
2. **🔧 Modular Tools**: YouTube processing, transcription, analysis, and report generation
3. **🧠 Multi-Provider Speech-to-Text**: OpenAI Whisper (free/local), Google, AssemblyAI, Rev.ai
4. **📋 Intelligent Workflows**: Context-aware processing and analysis
5. **📝 Automated Blog Generation**: Daily summary blog posts with structured insights
6. **⏰ Scheduled Processing**: Automatic daily processing with 10 PM delivery

## 📚 Documentation

Comprehensive documentation and visual guides are available in the **[`documentation/`](./documentation/)** folder:

- **[Setup Guide](./documentation/guides/SETUP_GUIDE.md)** - Complete setup instructions and API requirements
- **[Framework Comparison](./documentation/guides/DETAILED_FRAMEWORK_COMPARISON.md)** - Analysis of 9 AI frameworks
- **[Multi-Provider Guide](./documentation/guides/MULTI_PROVIDER_GUIDE.md)** - Configure multiple LLM providers
- **[Strategic Analysis](./documentation/guides/WHY_OPENAI_ONLY.md)** - Provider choice reasoning
- **[Visual Diagrams](./documentation/diagrams/)** - Architecture diagrams and performance analysis

## 🏗️ Architecture Overview

### Autonomous Agent System
```
🤖 Daily Stock News Agent
├── 🧠 Natural Language Processing
├── 📋 Intelligent Workflow Planning  
├── 🔧 Dynamic Tool Orchestration
└── 🎯 Context-Aware Execution

🔧 Modular Tool Ecosystem
├── 📺 YouTube Processing Tool (Video Download)
├── 🎙️ Speech-to-Text Tool (Multi-Provider)
├── 🔍 Content Analysis Tool (Stock Insights)
├── 📊 Report Generation Tool (Structured Reports)
├── 📝 Blog Generation Tool (Daily Summaries)
└── 🔌 Extensible Tool Registry
```

## 📁 Project Structure

```
📦 Daily-Stock-News-Agent/
├── 🤖 Autonomous Agent Core:
│   └── autonomous_stock_news_agent.py    # Main intelligent agent
├── 🔧 Modular Tool System:
│   ├── tools/
│   │   ├── base_tool.py                  # Tool interface & registry
│   │   ├── youtube_processing_tool.py    # YouTube video download/processing
│   │   ├── speech_to_text_tool.py        # Multi-provider transcription
│   │   ├── content_analysis_tool.py      # Stock market analysis
│   │   ├── report_generation_tool.py     # Structured report creation
│   │   └── blog_generation_tool.py       # Blog post generation
├── 📊 Data Storage:
│   ├── videos/                           # Downloaded video files
│   ├── transcripts/                      # Speech-to-text outputs
│   ├── reports/                          # Daily structured reports
│   └── blogs/                            # Generated blog posts
├── 🔐 Configuration:
│   ├── .env                              # API keys and settings
│   ├── .env.example                      # Environment template
│   ├── config.yaml                       # Channel and processing config
│   └── requirements.txt                  # Python dependencies
├── � Documentation:
│   └── documentation/                    # Comprehensive guides and diagrams
│       ├── guides/                       # Setup and framework guides
│       └── diagrams/                     # Architecture and visual diagrams
├── 🔬 Framework Comparison:
│   └── framework_comparison/             # 9 AI framework implementations
│       ├── implementations/              # Agent implementations  
│       ├── examples/                     # Usage examples
│       └── utils/                        # Analysis and comparison tools
└── 🔄 Legacy/Utils:
    ├── manual_processor.py               # Manual video processing
    └── batch_processor.py                # Batch historical processing
```

## 🎯 Key Features

### **🤖 Intelligent Processing**
- **Autonomous Workflow**: Natural language requests with smart tool selection
- **Multi-Channel Support**: Processes both MoneyPurse and Day Trader Telugu channels
- **Segment Detection**: Automatically identifies 20-30 content segments per video
- **Context-Aware Analysis**: Understands stock market terminology and Telugu context

### **🎙️ Advanced Transcription**
- **Multi-Provider Support**: OpenAI Whisper (free/local), Google, AssemblyAI, Rev.ai
- **Telugu-to-English**: Automatic translation and transcription
- **High Accuracy**: Optimized for financial terminology and Telugu accents
- **Cost-Effective**: Prioritizes free and low-cost options

### **📊 Comprehensive Analysis**
- **Stock Recommendations**: Extracts buy/sell/hold signals
- **Market Sentiment**: Analyzes overall market mood and trends
- **Sector Performance**: Identifies sector-wise insights and opportunities
- **Technical Analysis**: Captures chart patterns and technical indicators
- **Key Movements**: Highlights significant stock price movements and news

### **📝 Structured Reporting**
- **Daily Summary Reports**: Comprehensive structured insights
- **Blog Post Generation**: Engaging daily blog entries
- **Comparison Analysis**: Side-by-side insights from both channels
- **Actionable Insights**: Prioritized recommendations and key takeaways
- **Historical Tracking**: Maintains record of daily analyses

### **⏰ Automated Operations**
- **Daily Scheduling**: Automatic processing after 7 PM IST
- **Smart Monitoring**: Checks for new uploads with retry logic
- **Timely Delivery**: Summary reports ready before 10 PM
- **Error Handling**: Robust failure recovery and notification
- **Local Storage**: All data stored securely on local system

## 🚀 Quick Start

### 1. **Installation**
```bash
git clone <repository-url>
cd Daily-Stock-News-Agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. **Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys (optional - Whisper works without keys)
```

### 3. **Manual Processing**
```python
from autonomous_stock_news_agent import StockNewsAgent

# Process today's videos
agent = StockNewsAgent()
results = agent.process_daily_videos()
```

### 4. **Automated Scheduling**
```bash
# Set up daily automation (see SCHEDULER_GUIDE.md)
python setup_scheduler.py
```

## 📋 Supported Channels

| Channel | URL | Typical Upload | Content Focus |
|---------|-----|---------------|---------------|
| **MoneyPurse** | [@MoneyPurse](https://www.youtube.com/@MoneyPurse) | ~7 PM IST | Market analysis, stock picks |
| **Day Trader Telugu** | [@daytradertelugu](https://www.youtube.com/@daytradertelugu) | ~7 PM IST | Day trading, technical analysis |

## 🔧 Provider Options

### **Speech-to-Text Providers**
| Provider | Cost | Quality | Telugu Support | Setup |
|----------|------|---------|---------------|-------|
| **OpenAI Whisper** | FREE | ⭐⭐⭐⭐⭐ | Excellent | Local |
| **Google Speech-to-Text** | Free tier | ⭐⭐⭐⭐ | Good | API Key |
| **AssemblyAI** | 5h free | ⭐⭐⭐⭐ | Good | API Key |
| **Rev.ai** | 5h free | ⭐⭐⭐⭐ | Good | API Key |

**Recommended**: Start with OpenAI Whisper (completely free and runs locally)

## 📊 Sample Output

### **Daily Report Structure**
```markdown
# Daily Stock Market Analysis - July 21, 2025

## 📈 Market Overview
- **Market Sentiment**: Bullish/Bearish/Neutral
- **Key Themes**: [Major market themes from both channels]

## 🎯 Stock Recommendations
### MoneyPurse Channel
- **Buy**: [Stock symbols with rationale]
- **Sell**: [Stock symbols with rationale]
- **Hold**: [Stock symbols with rationale]

### Day Trader Telugu Channel  
- **Buy**: [Stock symbols with rationale]
- **Sell**: [Stock symbols with rationale]
- **Hold**: [Stock symbols with rationale]

## 🏭 Sector Analysis
- **Top Performing Sectors**: [Analysis]
- **Underperforming Sectors**: [Analysis]
- **Sector Rotation Insights**: [Analysis]

## 📊 Technical Analysis
- **Chart Patterns**: [Key patterns discussed]
- **Support/Resistance Levels**: [Important levels]
- **Market Indicators**: [RSI, MACD, etc.]

## 📰 Key News & Events
- **Market Moving News**: [Important announcements]
- **Earnings Highlights**: [Key earnings reports]
- **Economic Indicators**: [GDP, inflation, etc.]

## 🎯 Action Items
1. **Immediate Actions**: [Today's actionable insights]
2. **Watch List**: [Stocks to monitor]
3. **Risk Factors**: [Key risks to consider]

## 📝 Channel Summary Comparison
[Side-by-side comparison of both channels' perspectives]
```

---

**Version**: 1.0.0  
**Last Updated**: July 21, 2025  
**Status**: Ready for Development ✅
