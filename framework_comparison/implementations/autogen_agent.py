"""
Daily Stock News Agent - AutoGen Implementation

This implementation uses Microsoft's AutoGen framework for conversational 
multi-agent systems with complex reasoning and collaboration.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.capabilities import transforms


class AutoGenStockNewsSystem:
    """AutoGen-based conversational multi-agent stock news system"""
    
    def __init__(self, openai_api_key: str):
        # Configuration for all agents
        self.config_list = [
            {
                "model": "gpt-4",
                "api_key": openai_api_key,
                "temperature": 0.1
            }
        ]
        
        self.llm_config = {
            "config_list": self.config_list,
            "timeout": 120,
        }
        
        self._create_agents()
        self._setup_group_chat()
    
    def _create_agents(self):
        """Create specialized agents for different tasks"""
        
        # Video Processing Agent
        self.video_agent = AssistantAgent(
            name="VideoProcessor",
            system_message="""You are a YouTube video processing specialist. Your responsibilities:
            1. Download videos from specified Telugu stock channels
            2. Validate video quality and duration
            3. Organize files for processing
            4. Report download status and metadata
            
            Always provide detailed status updates and handle errors gracefully.""",
            llm_config=self.llm_config,
        )
        
        # Transcription Agent
        self.transcription_agent = AssistantAgent(
            name="TranscriptionExpert",
            system_message="""You are a transcription expert specializing in Telugu financial content.
            Your responsibilities:
            1. Transcribe Telugu audio/video to text using OpenAI Whisper
            2. Translate Telugu content to English while preserving financial terminology
            3. Ensure high accuracy and confidence scores
            4. Handle regional accents and financial jargon
            
            Provide confidence scores and flag any uncertain translations.""",
            llm_config=self.llm_config,
        )
        
        # Stock Analysis Agent
        self.analyst_agent = AssistantAgent(
            name="StockAnalyst", 
            system_message="""You are a senior stock market analyst with expertise in Indian markets.
            Your responsibilities:
            1. Analyze transcribed content for stock mentions and recommendations
            2. Identify market sentiment (BULLISH/BEARISH/NEUTRAL)
            3. Extract actionable investment insights
            4. Assess confidence levels for each recommendation
            5. Identify sector trends and themes
            
            Focus on: Stock symbols, price targets, buy/sell signals, risk factors.
            Always provide confidence scores (0-1) for your analysis.""",
            llm_config=self.llm_config,
        )
        
        # Report Generation Agent
        self.reporter_agent = AssistantAgent(
            name="ReportWriter",
            system_message="""You are a professional financial report writer. Your responsibilities:
            1. Synthesize analysis from all agents into coherent reports
            2. Create executive summaries and detailed recommendations
            3. Format reports in professional markdown
            4. Include risk disclaimers and confidence levels
            5. Ensure reports are actionable for investors
            
            Structure: Executive Summary, Market Overview, Key Recommendations, Risk Assessment.""",
            llm_config=self.llm_config,
        )
        
        # Quality Assurance Agent
        self.qa_agent = AssistantAgent(
            name="QualityAssurance",
            system_message="""You are a quality assurance specialist for financial reports.
            Your responsibilities:
            1. Review all agent outputs for accuracy and completeness
            2. Validate stock analysis against market data
            3. Ensure report quality and professional standards
            4. Flag any inconsistencies or missing information
            5. Approve final deliverables
            
            Be thorough and constructive in your feedback.""",
            llm_config=self.llm_config,
        )
        
        # Coordinator Agent (acts as user proxy)
        self.coordinator = UserProxyAgent(
            name="ProcessCoordinator",
            system_message="""You coordinate the entire stock news processing workflow.
            Guide the conversation to ensure all steps are completed in order:
            1. Video download and processing
            2. Transcription and translation  
            3. Stock analysis and insights
            4. Report generation
            5. Quality assurance and approval
            
            Keep the team focused and ensure deliverables meet requirements.""",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )
    
    def _setup_group_chat(self):
        """Setup group chat with conversation flow"""
        
        # Define agent conversation order
        self.agents = [
            self.coordinator,
            self.video_agent,
            self.transcription_agent,
            self.analyst_agent,
            self.reporter_agent,
            self.qa_agent
        ]
        
        # Custom speaker selection function
        def custom_speaker_selection(last_speaker, groupchat):
            """Custom logic for selecting next speaker"""
            messages = groupchat.messages
            
            if len(messages) <= 1:
                return self.video_agent
            
            last_message = messages[-1]["content"].lower()
            
            # Route based on conversation context
            if "download" in last_message or "video" in last_message:
                if last_speaker == self.coordinator:
                    return self.video_agent
                elif last_speaker == self.video_agent:
                    return self.transcription_agent
            
            elif "transcrib" in last_message or "translate" in last_message:
                if last_speaker == self.video_agent:
                    return self.transcription_agent
                elif last_speaker == self.transcription_agent:
                    return self.analyst_agent
            
            elif "analy" in last_message or "stock" in last_message:
                if last_speaker == self.transcription_agent:
                    return self.analyst_agent
                elif last_speaker == self.analyst_agent:
                    return self.reporter_agent
            
            elif "report" in last_message:
                if last_speaker == self.analyst_agent:
                    return self.reporter_agent
                elif last_speaker == self.reporter_agent:
                    return self.qa_agent
            
            elif "quality" in last_message or "review" in last_message:
                return self.qa_agent
            
            # Default fallback
            return self.coordinator
        
        # Create group chat
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=20,
            speaker_selection_method=custom_speaker_selection
        )
        
        # Create group chat manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
            system_message="""You are managing a stock news processing workflow.
            Ensure all agents contribute effectively and the conversation stays on track.
            Guide the discussion through these phases:
            1. Video Processing
            2. Transcription 
            3. Analysis
            4. Report Generation
            5. Quality Assurance"""
        )
    
    async def process_daily_news(self, channels: List[str], date: str = None) -> Dict[str, Any]:
        """Process daily news using conversational agents"""
        
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🏗️ AutoGen Conversational Processing for {date}")
        print("=" * 60)
        
        # Initial message to start the workflow
        initial_message = f"""
        Let's process daily stock news for {date}.
        
        Channels to process: {', '.join(channels)}
        
        Please coordinate the following workflow:
        1. VideoProcessor: Download videos from the specified channels
        2. TranscriptionExpert: Transcribe and translate all videos  
        3. StockAnalyst: Analyze content for stock insights and recommendations
        4. ReportWriter: Generate comprehensive investment report
        5. QualityAssurance: Review and approve final deliverables
        
        Let's begin with video processing. VideoProcessor, please start downloading.
        """
        
        try:
            # Start the conversation
            chat_result = self.coordinator.initiate_chat(
                recipient=self.manager,
                message=initial_message
            )
            
            # Extract results from conversation
            messages = self.group_chat.messages
            final_message = messages[-1]["content"] if messages else "No conversation recorded"
            
            # Save conversation log
            log_file = f"./data/logs/autogen_conversation_{date.replace('-', '')}.txt"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            with open(log_file, 'w') as f:
                for msg in messages:
                    f.write(f"[{msg['name']}]: {msg['content']}\n\n")
            
            return {
                "success": True,
                "conversation_summary": final_message,
                "total_messages": len(messages),
                "agents_participated": len(set(msg["name"] for msg in messages)),
                "conversation_log": log_file,
                "date": date,
                "channels": channels
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "date": date
            }
    
    def create_specialized_workflow(self, workflow_type: str):
        """Create specialized workflows for different use cases"""
        
        if workflow_type == "real_time_analysis":
            # Create agents for real-time market monitoring
            real_time_agent = AssistantAgent(
                name="RealTimeMonitor",
                system_message="""You monitor live market feeds and breaking news.
                Immediately flag urgent stock movements and breaking financial news.""",
                llm_config=self.llm_config
            )
            return real_time_agent
        
        elif workflow_type == "risk_assessment":
            # Create specialized risk assessment agent
            risk_agent = AssistantAgent(
                name="RiskAssessor", 
                system_message="""You specialize in financial risk assessment.
                Evaluate portfolio risk, market volatility, and recommendation confidence.""",
                llm_config=self.llm_config
            )
            return risk_agent
    
    def get_conversation_workflow_visualization(self) -> str:
        """Return ASCII visualization of conversation workflow"""
        return """
🏗️ AutoGen Conversational Multi-Agent Workflow:

                        ┌─────────────────────┐
                        │   Group Chat        │
                        │   Manager           │
                        └──────────┬──────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
                   ▼               ▼               ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │ Coordinator     │ │ Video Processor │ │ Transcription   │
        │                 │ │                 │ │ Expert          │
        │ • Orchestrate   │ │ • Download      │ │                 │
        │ • Guide Flow    │ │ • Validate      │ │ • Telugu→English│
        │ • Monitor       │ │ • Organize      │ │ • Financial     │
        │   Progress      │ │   Content       │ │   Terminology   │
        └─────────┬───────┘ └─────────┬───────┘ └─────────┬───────┘
                  │                   │                   │
                  └─────────┬─────────┼─────────┬─────────┘
                            │         │         │
                            ▼         ▼         ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │ Stock Analyst   │ │ Report Writer   │ │ Quality         │
        │                 │ │                 │ │ Assurance       │
        │ • Market        │ │ • Synthesize    │ │                 │
        │   Analysis      │ │ • Professional  │ │ • Review All    │
        │ • Sentiment     │ │   Reports       │ │ • Validate      │
        │ • Recommendations│ │ • Executive     │ │ • Approve       │
        │ • Confidence    │ │   Summary       │ │   Deliverables  │
        └─────────────────┘ └─────────────────┘ └─────────────────┘

Conversation Features:
✅ Dynamic speaker selection based on context
✅ Conversational reasoning and collaboration  
✅ Complex multi-turn discussions
✅ Adaptive workflow based on conversation flow
✅ Quality control through peer review
✅ Complete conversation logging
        """


# Example usage
async def main():
    """Demonstrate AutoGen conversational workflow"""
    
    print("🏗️ AutoGen Conversational Multi-Agent System")
    print("=" * 50)
    
    # Initialize system
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    system = AutoGenStockNewsSystem(api_key)
    
    # Show workflow visualization
    print(system.get_conversation_workflow_visualization())
    
    # Process daily news
    channels = ["moneypurse", "daytradertelugu"]
    result = await system.process_daily_news(channels)
    
    if result["success"]:
        print(f"\n✅ Conversational processing completed!")
        print(f"📅 Date: {result['date']}")
        print(f"📺 Channels: {', '.join(result['channels'])}")
        print(f"💬 Messages: {result['total_messages']}")
        print(f"🤖 Agents: {result['agents_participated']}")
        print(f"📄 Log: {result['conversation_log']}")
        print(f"📋 Summary: {result['conversation_summary'][:200]}...")
    else:
        print(f"❌ Processing failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
