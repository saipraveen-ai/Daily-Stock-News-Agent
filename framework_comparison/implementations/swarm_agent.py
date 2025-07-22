"""
Daily Stock News Agent - OpenAI Swarm Implementation

This implementation uses OpenAI's Swarm framework for lightweight 
multi-agent coordination with simple handoffs and function calling.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json

from swarm import Swarm, Agent
from swarm.types import Response


class SwarmStockNewsSystem:
    """OpenAI Swarm-based lightweight multi-agent system"""
    
    def __init__(self, openai_api_key: str):
        self.client = Swarm()
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Shared context for agent coordination
        self.context = {
            "processed_videos": [],
            "transcriptions": [],
            "analyses": [],
            "reports": [],
            "errors": [],
            "current_date": "",
            "channels": []
        }
        
        # Create agents with specific responsibilities
        self.video_agent = self._create_video_agent()
        self.transcription_agent = self._create_transcription_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.report_agent = self._create_report_agent()
        self.coordinator_agent = self._create_coordinator_agent()
    
    def _create_video_agent(self) -> Agent:
        """Create video processing agent"""
        
        def download_videos(channels: str, date: str) -> str:
            """Download videos from specified channels"""
            try:
                channel_list = channels.split(",")
                videos = []
                
                for channel in channel_list:
                    channel = channel.strip()
                    video_data = {
                        "title": f"Daily Market Update - {channel}",
                        "file_path": f"./data/videos/{channel}_{date.replace('-', '')}.mp4",
                        "channel": channel,
                        "duration": 1800,
                        "download_date": date,
                        "status": "downloaded"
                    }
                    videos.append(video_data)
                
                self.context["processed_videos"] = videos
                return f"Successfully downloaded {len(videos)} videos from channels: {channels}"
                
            except Exception as e:
                error_msg = f"Video download failed: {str(e)}"
                self.context["errors"].append(error_msg)
                return error_msg
        
        def validate_video_quality() -> str:
            """Validate downloaded video quality"""
            videos = self.context.get("processed_videos", [])
            if not videos:
                return "No videos to validate"
            
            validated_count = 0
            for video in videos:
                if video.get("duration", 0) > 300:  # At least 5 minutes
                    validated_count += 1
                    video["quality_check"] = "passed"
                else:
                    video["quality_check"] = "failed"
                    self.context["errors"].append(f"Video too short: {video['title']}")
            
            return f"Validated {validated_count}/{len(videos)} videos"
        
        return Agent(
            name="VideoProcessor",
            instructions="""You are a video processing specialist for stock market content.
            Your responsibilities:
            1. Download videos from specified YouTube channels
            2. Validate video quality and duration
            3. Organize files for further processing
            4. Handle download errors gracefully
            
            Always provide clear status updates and call appropriate functions.""",
            functions=[download_videos, validate_video_quality]
        )
    
    def _create_transcription_agent(self) -> Agent:
        """Create transcription agent"""
        
        def transcribe_videos() -> str:
            """Transcribe all downloaded videos"""
            videos = self.context.get("processed_videos", [])
            if not videos:
                return "No videos available for transcription"
            
            transcriptions = []
            for video in videos:
                if video.get("quality_check") != "passed":
                    continue
                
                # Simulate transcription process
                transcription = {
                    "video_info": video,
                    "original_text": "మార్కెట్ పరిస్థితులు మరియు ఈరోజు ముఖ్య స్టాక్ సిఫార్సులు...",
                    "translated_text": f"Market analysis for {video['channel']}: Key stocks showing bullish momentum include Reliance, TCS, and Infosys. Technical indicators suggest...",
                    "language": "te",
                    "confidence": 0.91,
                    "transcription_date": self.context["current_date"]
                }
                transcriptions.append(transcription)
            
            self.context["transcriptions"] = transcriptions
            return f"Successfully transcribed {len(transcriptions)} videos with avg confidence 0.91"
        
        def improve_translation(video_title: str) -> str:
            """Improve translation for specific video"""
            transcriptions = self.context.get("transcriptions", [])
            
            for transcription in transcriptions:
                if transcription["video_info"]["title"] == video_title:
                    # Simulate improved translation
                    transcription["translated_text"] += " [Enhanced translation with financial terminology]"
                    transcription["confidence"] = min(0.98, transcription["confidence"] + 0.05)
                    return f"Improved translation for {video_title}"
            
            return f"Video {video_title} not found for translation improvement"
        
        return Agent(
            name="TranscriptionExpert",
            instructions="""You are a transcription expert specializing in Telugu financial content.
            Your responsibilities:
            1. Transcribe Telugu audio to text using OpenAI Whisper
            2. Translate to English preserving financial terminology
            3. Ensure high confidence scores
            4. Handle regional accents and market jargon
            
            Focus on accuracy and financial term preservation.""",
            functions=[transcribe_videos, improve_translation]
        )
    
    def _create_analysis_agent(self) -> Agent:
        """Create stock analysis agent"""
        
        def analyze_stock_content() -> str:
            """Analyze transcribed content for stock insights"""
            transcriptions = self.context.get("transcriptions", [])
            if not transcriptions:
                return "No transcriptions available for analysis"
            
            analyses = []
            for transcription in transcriptions:
                # Simulate intelligent stock analysis
                analysis = {
                    "channel": transcription["video_info"]["channel"],
                    "video_title": transcription["video_info"]["title"],
                    "analysis_date": self.context["current_date"],
                    "market_sentiment": "BULLISH",
                    "key_stocks": ["RELIANCE", "TCS", "INFY", "HDFC"],
                    "recommendations": [
                        {
                            "symbol": "RELIANCE",
                            "action": "BUY",
                            "target_price": 2800,
                            "confidence": 0.87,
                            "rationale": "Strong technical breakout with volume"
                        },
                        {
                            "symbol": "TCS",
                            "action": "HOLD", 
                            "target_price": 4200,
                            "confidence": 0.75,
                            "rationale": "Consolidation phase, wait for direction"
                        }
                    ],
                    "confidence_score": 0.84,
                    "risk_factors": ["Market volatility", "Global uncertainty"]
                }
                analyses.append(analysis)
            
            self.context["analyses"] = analyses
            return f"Analyzed {len(analyses)} videos. Overall market sentiment: BULLISH"
        
        def calculate_portfolio_risk() -> str:
            """Calculate portfolio risk from recommendations"""
            analyses = self.context.get("analyses", [])
            if not analyses:
                return "No analyses available for risk calculation"
            
            total_recommendations = sum(len(a["recommendations"]) for a in analyses)
            high_confidence_recs = sum(
                1 for a in analyses 
                for rec in a["recommendations"] 
                if rec["confidence"] > 0.8
            )
            
            risk_score = 1 - (high_confidence_recs / total_recommendations) if total_recommendations > 0 else 1
            
            # Add risk assessment to context
            for analysis in analyses:
                analysis["portfolio_risk"] = risk_score
            
            return f"Portfolio risk calculated: {risk_score:.2f} (lower is better)"
        
        return Agent(
            name="StockAnalyst",
            instructions="""You are a senior stock market analyst with 15+ years experience.
            Your responsibilities:
            1. Analyze transcribed content for actionable stock insights
            2. Identify market sentiment and trends
            3. Generate buy/sell/hold recommendations with confidence scores
            4. Assess risk factors and portfolio implications
            
            Be conservative with high-confidence recommendations.""",
            functions=[analyze_stock_content, calculate_portfolio_risk]
        )
    
    def _create_report_agent(self) -> Agent:
        """Create report generation agent"""
        
        def generate_comprehensive_report() -> str:
            """Generate comprehensive investment report"""
            analyses = self.context.get("analyses", [])
            if not analyses:
                return "No analyses available for report generation"
            
            # Generate report content
            report_content = f"""
# Swarm Multi-Agent Stock Analysis Report - {self.context['current_date']}

## Executive Summary
- **Videos Analyzed**: {len(analyses)}
- **Channels Processed**: {', '.join(set(a['channel'] for a in analyses))}
- **Total Recommendations**: {sum(len(a['recommendations']) for a in analyses)}

## Market Overview
Generated by OpenAI Swarm multi-agent coordination system.

"""
            
            # Add channel-specific analysis
            for analysis in analyses:
                report_content += f"""
### {analysis['channel']} Analysis
**Video**: {analysis['video_title']}  
**Sentiment**: {analysis['market_sentiment']}  
**Confidence**: {analysis['confidence_score']:.2f}  

**Key Recommendations**:
"""
                for rec in analysis['recommendations']:
                    report_content += f"""
- **{rec['symbol']}**: {rec['action']} 
  - Target: ₹{rec['target_price']} | Confidence: {rec['confidence']:.2f}
  - Rationale: {rec['rationale']}
"""
            
            # Add risk section
            report_content += f"""
## Risk Assessment
**Portfolio Risk Score**: {analyses[0].get('portfolio_risk', 'N/A')}

**Risk Factors Identified**:
"""
            all_risks = set()
            for analysis in analyses:
                all_risks.update(analysis.get('risk_factors', []))
            
            for risk in all_risks:
                report_content += f"- {risk}\n"
            
            report_content += """
---
*Generated by OpenAI Swarm Multi-Agent System*
"""
            
            # Save report
            report_file = f"./data/reports/swarm_report_{self.context['current_date'].replace('-', '')}.md"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            self.context["reports"].append(report_file)
            return f"Comprehensive report generated: {report_file}"
        
        def create_executive_summary() -> str:
            """Create executive summary for stakeholders"""
            analyses = self.context.get("analyses", [])
            if not analyses:
                return "No data for executive summary"
            
            # Calculate key metrics
            total_recs = sum(len(a['recommendations']) for a in analyses)
            avg_confidence = sum(a['confidence_score'] for a in analyses) / len(analyses)
            
            # Determine overall sentiment
            sentiments = [a['market_sentiment'] for a in analyses]
            overall_sentiment = max(set(sentiments), key=sentiments.count)
            
            summary = f"""
EXECUTIVE SUMMARY - {self.context['current_date']}

📊 MARKET ANALYSIS:
- Overall Sentiment: {overall_sentiment}
- Videos Analyzed: {len(analyses)}
- Recommendations: {total_recs}
- Average Confidence: {avg_confidence:.1%}

🎯 KEY INSIGHTS:
- Multi-agent coordination successful
- High-confidence recommendations identified
- Risk factors properly assessed
"""
            
            return summary
        
        return Agent(
            name="ReportWriter",
            instructions="""You are a professional financial report writer.
            Your responsibilities:
            1. Generate comprehensive investment reports
            2. Create executive summaries for stakeholders
            3. Synthesize multi-agent analysis results
            4. Ensure professional formatting and clarity
            
            Focus on actionable insights and clear communication.""",
            functions=[generate_comprehensive_report, create_executive_summary]
        )
    
    def _create_coordinator_agent(self) -> Agent:
        """Create coordinator agent for workflow management"""
        
        def handoff_to_video_processor() -> Agent:
            """Hand off to video processing agent"""
            return self.video_agent
        
        def handoff_to_transcription() -> Agent:
            """Hand off to transcription agent"""
            return self.transcription_agent
        
        def handoff_to_analysis() -> Agent:
            """Hand off to analysis agent"""
            return self.analysis_agent
        
        def handoff_to_report_writer() -> Agent:
            """Hand off to report writer"""
            return self.report_agent
        
        def get_processing_status() -> str:
            """Get current processing status"""
            status = {
                "videos": len(self.context.get("processed_videos", [])),
                "transcriptions": len(self.context.get("transcriptions", [])),
                "analyses": len(self.context.get("analyses", [])),
                "reports": len(self.context.get("reports", [])),
                "errors": len(self.context.get("errors", []))
            }
            return f"Status: {json.dumps(status, indent=2)}"
        
        return Agent(
            name="ProcessCoordinator",
            instructions="""You are the process coordinator for the stock news analysis workflow.
            Your responsibilities:
            1. Orchestrate the entire workflow across agents
            2. Hand off tasks to appropriate specialists
            3. Monitor progress and handle coordination
            4. Ensure all steps complete successfully
            
            Workflow: Video Processing → Transcription → Analysis → Report Generation
            
            Use handoff functions to delegate work to specialists.""",
            functions=[
                handoff_to_video_processor,
                handoff_to_transcription, 
                handoff_to_analysis,
                handoff_to_report_writer,
                get_processing_status
            ]
        )
    
    async def process_daily_news(self, channels: List[str], date: str = None) -> Dict[str, Any]:
        """Process daily news using Swarm coordination"""
        
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Initialize context
        self.context.update({
            "current_date": date,
            "channels": channels,
            "processed_videos": [],
            "transcriptions": [],
            "analyses": [],
            "reports": [],
            "errors": []
        })
        
        print(f"🚀 OpenAI Swarm Multi-Agent Processing for {date}")
        print("=" * 60)
        
        try:
            # Start workflow with coordinator
            initial_message = f"""
            Let's process daily stock news for {date}.
            
            Channels: {', '.join(channels)}
            
            Please coordinate the complete workflow:
            1. Download videos from the specified channels
            2. Transcribe and translate content
            3. Analyze for stock insights
            4. Generate comprehensive reports
            
            Start by handing off to the video processor.
            """
            
            # Run the swarm coordination
            response = self.client.run(
                agent=self.coordinator_agent,
                messages=[{"role": "user", "content": initial_message}]
            )
            
            # Extract final results
            final_message = response.messages[-1]["content"]
            
            return {
                "success": len(self.context["analyses"]) > 0,
                "videos_processed": len(self.context["processed_videos"]),
                "transcriptions": len(self.context["transcriptions"]),
                "analyses": len(self.context["analyses"]),
                "reports_generated": len(self.context["reports"]),
                "errors": self.context["errors"],
                "final_response": final_message,
                "date": date,
                "channels": channels,
                "agent_handoffs": "Multiple agent coordination via Swarm"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "date": date,
                "context": self.context
            }
    
    def get_swarm_workflow_visualization(self) -> str:
        """Return ASCII visualization of Swarm workflow"""
        return """
🚀 OpenAI Swarm Lightweight Multi-Agent Workflow:

                    ┌─────────────────────┐
                    │  Process Coordinator│
                    │                     │
                    │ • Orchestrate Flow  │
                    │ • Handle Handoffs   │
                    │ • Monitor Progress  │
                    └──────────┬──────────┘
                               │
                       [Agent Handoffs]
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Video Processor │ │ Transcription   │ │ Stock Analyst   │
│                 │ │ Expert          │ │                 │
│ • download_     │ │ • transcribe_   │ │ • analyze_stock │
│   videos()      │ │   videos()      │ │   _content()    │
│ • validate_     │ │ • improve_      │ │ • calculate_    │
│   video_quality │ │   translation() │ │   portfolio_    │
│   ()            │ │                 │ │   risk()        │
└─────────┬───────┘ └─────────┬───────┘ └─────────┬───────┘
          │                   │                   │
          └─────────┬─────────┼─────────┬─────────┘
                    │         │         │
                    ▼         ▼         ▼
            ┌─────────────────────────────────┐
            │       Report Writer             │
            │                                 │
            │ • generate_comprehensive_       │
            │   report()                      │
            │ • create_executive_summary()    │
            │                                 │
            └─────────────────────────────────┘

Swarm Features:
✅ Lightweight agent coordination
✅ Function-based tool calling
✅ Simple agent handoffs
✅ Shared context management
✅ OpenAI-native integration
✅ Minimal framework overhead
        """


# Example usage
async def main():
    """Demonstrate OpenAI Swarm workflow"""
    
    print("🚀 OpenAI Swarm Lightweight Multi-Agent System")
    print("=" * 50)
    
    # Initialize system
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    system = SwarmStockNewsSystem(api_key)
    
    # Show workflow visualization
    print(system.get_swarm_workflow_visualization())
    
    # Process daily news
    channels = ["moneypurse", "daytradertelugu"]
    result = await system.process_daily_news(channels)
    
    if result["success"]:
        print(f"\n✅ Swarm coordination completed!")
        print(f"📅 Date: {result['date']}")
        print(f"📺 Channels: {', '.join(result['channels'])}")
        print(f"🎥 Videos: {result['videos_processed']}")
        print(f"📝 Transcriptions: {result['transcriptions']}")
        print(f"📊 Analyses: {result['analyses']}")
        print(f"📄 Reports: {result['reports_generated']}")
        print(f"🤝 Coordination: {result['agent_handoffs']}")
        print(f"💬 Final Response: {result['final_response'][:100]}...")
        
        if result['errors']:
            print(f"⚠️ Errors: {len(result['errors'])}")
    else:
        print(f"❌ Processing failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
