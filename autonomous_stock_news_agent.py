"""
Autonomous Stock News Agent

Main orchestrator for the Daily Stock News Agent system.
This agent intelligently coordinates all tools to process YouTube videos,
transcribe content, analyze stock insights, and generate comprehensive reports.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from tools import (
    BaseTool, ToolResult, ToolConfig, ToolCategory, ToolPriority,
    YouTubeProcessingTool, SpeechToTextTool, ContentAnalysisTool, ReportGenerationTool,
    tool_registry, register_tool
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/stock_news_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("StockNewsAgent")


@dataclass
class ProcessingRequest:
    """Request for processing videos"""
    operation: str
    date: Optional[str] = None
    channels: Optional[List[str]] = None
    video_urls: Optional[List[str]] = None
    output_formats: Optional[List[str]] = None
    enable_translation: bool = True
    ai_provider: str = "auto"
    stt_provider: str = "auto"


@dataclass
class ProcessingResult:
    """Result from processing operation"""
    success: bool
    processed_videos: List[Dict[str, Any]]
    transcriptions: List[Dict[str, Any]]
    analyses: List[Dict[str, Any]]
    reports: List[str]
    errors: List[str]
    processing_time: float
    metadata: Dict[str, Any]


class AutonomousStockNewsAgent:
    """
    Autonomous agent that orchestrates the entire stock news processing pipeline.
    
    Features:
    - Intelligent workflow planning and execution
    - Tool coordination and error handling
    - Natural language request processing
    - Automatic fallback and retry mechanisms
    - Progress tracking and reporting
    - Multi-channel video processing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the autonomous stock news agent.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.tools = {}
        self.processing_history = []
        
        # Create logs directory
        os.makedirs('./logs', exist_ok=True)
        
        self.logger = logging.getLogger("StockNewsAgent")
        self.logger.info("Initializing Autonomous Stock News Agent")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment variables"""
        config = {
            # Default settings
            "youtube": {
                "download_path": os.getenv("YOUTUBE_DOWNLOAD_PATH", "./data/videos"),
                "quality": os.getenv("YOUTUBE_QUALITY", "best[height<=720]"),
                "max_duration": int(os.getenv("YOUTUBE_MAX_DURATION", "3600"))
            },
            "speech_to_text": {
                "default_provider": os.getenv("DEFAULT_STT_PROVIDER", "whisper"),
                "whisper_model": os.getenv("WHISPER_MODEL_SIZE", "base"),
                "google_api_key": os.getenv("GOOGLE_CLOUD_API_KEY"),
                "assemblyai_api_key": os.getenv("ASSEMBLYAI_API_KEY"),
                "output_path": os.getenv("TRANSCRIPTS_PATH", "./data/transcripts"),
                "enable_translation": os.getenv("ENABLE_TRANSLATION", "true").lower() == "true"
            },
            "content_analysis": {
                "ai_provider": os.getenv("DEFAULT_AI_PROVIDER", "local"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "google_ai_key": os.getenv("GOOGLE_AI_API_KEY"),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "output_path": os.getenv("ANALYSIS_PATH", "./data/analysis")
            },
            "report_generation": {
                "output_path": os.getenv("REPORTS_PATH", "./data/reports"),
                "default_formats": os.getenv("DEFAULT_OUTPUT_FORMATS", "markdown,json").split(","),
                "template_path": os.getenv("BLOG_TEMPLATE_PATH", "./templates")
            },
            "automation": {
                "processing_time": os.getenv("DAILY_PROCESSING_TIME", "19:30"),
                "delivery_time": os.getenv("REPORT_DELIVERY_TIME", "22:00"),
                "auto_enabled": os.getenv("AUTO_PROCESSING_ENABLED", "true").lower() == "true",
                "max_retries": int(os.getenv("MAX_RETRIES", "3")),
                "retry_delay": int(os.getenv("RETRY_DELAY", "300"))
            }
        }
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config file {config_path}: {e}")
        
        return config
    
    async def initialize(self) -> bool:
        """Initialize all tools and prepare the agent for operation"""
        try:
            self.logger.info("Initializing tools...")
            
            # Initialize YouTube Processing Tool
            youtube_config = ToolConfig(
                name="youtube_processor",
                category=ToolCategory.YOUTUBE,
                priority=ToolPriority.HIGH,
                settings=self.config["youtube"]
            )
            self.tools["youtube"] = YouTubeProcessingTool(youtube_config)
            register_tool(self.tools["youtube"])
            
            # Initialize Speech-to-Text Tool
            stt_config = ToolConfig(
                name="speech_to_text",
                category=ToolCategory.TRANSCRIPTION,
                priority=ToolPriority.CRITICAL,
                settings=self.config["speech_to_text"]
            )
            self.tools["stt"] = SpeechToTextTool(stt_config)
            register_tool(self.tools["stt"])
            
            # Initialize Content Analysis Tool
            analysis_config = ToolConfig(
                name="content_analysis",
                category=ToolCategory.ANALYSIS,
                priority=ToolPriority.HIGH,
                settings=self.config["content_analysis"]
            )
            self.tools["analysis"] = ContentAnalysisTool(analysis_config)
            register_tool(self.tools["analysis"])
            
            # Initialize Report Generation Tool
            report_config = ToolConfig(
                name="report_generation",
                category=ToolCategory.GENERATION,
                priority=ToolPriority.MEDIUM,
                settings=self.config["report_generation"]
            )
            self.tools["report"] = ReportGenerationTool(report_config)
            register_tool(self.tools["report"])
            
            # Initialize all tools
            initialization_tasks = [
                tool.initialize() for tool in self.tools.values()
            ]
            
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check initialization results
            failed_tools = []
            for tool_name, result in zip(self.tools.keys(), results):
                if isinstance(result, Exception):
                    failed_tools.append(f"{tool_name}: {result}")
                elif not result.success:
                    failed_tools.append(f"{tool_name}: {result.error_message}")
            
            if failed_tools:
                self.logger.error(f"Failed to initialize tools: {failed_tools}")
                return False
            
            self.logger.info("All tools initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            return False
    
    async def process_daily_videos(self, date: Optional[str] = None) -> ProcessingResult:
        """
        Process daily videos from all monitored channels.
        
        Args:
            date: Date to process (YYYY-MM-DD format, defaults to today)
            
        Returns:
            ProcessingResult with comprehensive processing results
        """
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        request = ProcessingRequest(
            operation="daily_processing",
            date=date,
            channels=["moneypurse", "daytradertelugu"],
            output_formats=self.config["report_generation"]["default_formats"],
            enable_translation=self.config["speech_to_text"]["enable_translation"]
        )
        
        return await self.execute_processing_request(request)
    
    async def execute_processing_request(self, request: ProcessingRequest) -> ProcessingResult:
        """
        Execute a processing request with intelligent workflow orchestration.
        
        Args:
            request: ProcessingRequest with operation details
            
        Returns:
            ProcessingResult with execution results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting processing request: {request.operation}")
        
        try:
            # Phase 1: Video Discovery and Download
            video_results = await self._process_videos(request)
            
            if not video_results.success or not video_results.data.get("downloaded_videos"):
                return ProcessingResult(
                    success=False,
                    processed_videos=[],
                    transcriptions=[],
                    analyses=[],
                    reports=[],
                    errors=[f"Video processing failed: {video_results.error_message}"],
                    processing_time=0,
                    metadata={"phase": "video_download"}
                )
            
            downloaded_videos = video_results.data["downloaded_videos"]
            self.logger.info(f"Downloaded {len(downloaded_videos)} videos")
            
            # Phase 2: Speech-to-Text Transcription
            transcription_results = await self._transcribe_videos(downloaded_videos, request)
            
            if not transcription_results:
                return ProcessingResult(
                    success=False,
                    processed_videos=downloaded_videos,
                    transcriptions=[],
                    analyses=[],
                    reports=[],
                    errors=["No transcriptions generated"],
                    processing_time=0,
                    metadata={"phase": "transcription"}
                )
            
            self.logger.info(f"Transcribed {len(transcription_results)} videos")
            
            # Phase 3: Content Analysis
            analysis_results = await self._analyze_content(transcription_results, request)
            
            if not analysis_results:
                return ProcessingResult(
                    success=False,
                    processed_videos=downloaded_videos,
                    transcriptions=transcription_results,
                    analyses=[],
                    reports=[],
                    errors=["No analyses generated"],
                    processing_time=0,
                    metadata={"phase": "analysis"}
                )
            
            self.logger.info(f"Analyzed {len(analysis_results)} videos")
            
            # Phase 4: Report Generation
            report_results = await self._generate_reports(analysis_results, request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                success=True,
                processed_videos=downloaded_videos,
                transcriptions=transcription_results,
                analyses=analysis_results,
                reports=report_results,
                errors=[],
                processing_time=processing_time,
                metadata={
                    "request": request.__dict__,
                    "completion_time": datetime.now().isoformat()
                }
            )
            
            # Save processing history
            self.processing_history.append(result)
            
            self.logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Processing failed: {e}")
            
            return ProcessingResult(
                success=False,
                processed_videos=[],
                transcriptions=[],
                analyses=[],
                reports=[],
                errors=[str(e)],
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    async def _process_videos(self, request: ProcessingRequest) -> ToolResult:
        """Phase 1: Process and download videos"""
        youtube_tool = self.tools["youtube"]
        
        if request.video_urls:
            # Process specific video URLs
            results = []
            for url in request.video_urls:
                result = await youtube_tool.execute(
                    operation="download_url",
                    video_url=url
                )
                if result.success:
                    results.append(result.data)
            
            return ToolResult(
                success=len(results) > 0,
                data={"downloaded_videos": results}
            )
        
        else:
            # Process daily videos from monitored channels
            return await youtube_tool.execute(
                operation="download_latest",
                date=request.date.replace('-', '') if request.date else None
            )
    
    async def _transcribe_videos(
        self, 
        downloaded_videos: List[Any], 
        request: ProcessingRequest
    ) -> List[Dict[str, Any]]:
        """Phase 2: Transcribe downloaded videos"""
        stt_tool = self.tools["stt"]
        transcription_results = []
        
        for video in downloaded_videos:
            try:
                video_file = video.file_path if hasattr(video, 'file_path') else video.get('file_path')
                
                if not video_file or not os.path.exists(video_file):
                    self.logger.warning(f"Video file not found: {video_file}")
                    continue
                
                result = await stt_tool.execute(
                    audio_file=video_file,
                    provider=request.stt_provider,
                    enable_translation=request.enable_translation,
                    output_format="json"
                )
                
                if result.success:
                    transcription_data = result.data["transcription"]
                    transcription_results.append({
                        "video_info": video,
                        "transcription": transcription_data,
                        "transcript_file": result.data["output_file"]
                    })
                else:
                    self.logger.error(f"Transcription failed for {video_file}: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Error transcribing video: {e}")
                continue
        
        return transcription_results
    
    async def _analyze_content(
        self, 
        transcription_results: List[Dict[str, Any]], 
        request: ProcessingRequest
    ) -> List[Dict[str, Any]]:
        """Phase 3: Analyze transcribed content"""
        analysis_tool = self.tools["analysis"]
        analysis_results = []
        
        for transcription_result in transcription_results:
            try:
                video_info = transcription_result["video_info"]
                transcription = transcription_result["transcription"]
                
                # Extract channel name
                channel = getattr(video_info, 'channel', 'Unknown') if hasattr(video_info, 'channel') else video_info.get('channel', 'Unknown')
                video_title = getattr(video_info, 'title', 'Unknown') if hasattr(video_info, 'title') else video_info.get('title', 'Unknown')
                
                # Use translated text if available, otherwise original
                transcript_text = transcription.translated_text or transcription.original_text
                
                result = await analysis_tool.execute(
                    transcript_text=transcript_text,
                    channel=channel,
                    video_title=video_title,
                    analysis_types=["all"]
                )
                
                if result.success:
                    analysis_results.append(result.data["analysis"])
                else:
                    self.logger.error(f"Analysis failed for {channel}: {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing content: {e}")
                continue
        
        return analysis_results
    
    async def _generate_reports(
        self, 
        analysis_results: List[Dict[str, Any]], 
        request: ProcessingRequest
    ) -> List[str]:
        """Phase 4: Generate comprehensive reports"""
        report_tool = self.tools["report"]
        
        try:
            result = await report_tool.execute(
                analysis_data=analysis_results,
                report_date=request.date or datetime.now().strftime('%Y-%m-%d'),
                output_formats=request.output_formats or ["markdown", "json"],
                include_comparison=len(analysis_results) > 1
            )
            
            if result.success:
                return list(result.data["output_files"].values())
            else:
                self.logger.error(f"Report generation failed: {result.error_message}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            return []
    
    async def process_request_from_natural_language(self, user_request: str) -> ProcessingResult:
        """
        Process a natural language request and execute appropriate workflow.
        
        Args:
            user_request: Natural language description of what to do
            
        Returns:
            ProcessingResult from executed workflow
        """
        self.logger.info(f"Processing natural language request: {user_request}")
        
        # Simple keyword-based request parsing (can be enhanced with NLP)
        request = self._parse_natural_language_request(user_request)
        
        return await self.execute_processing_request(request)
    
    def _parse_natural_language_request(self, user_request: str) -> ProcessingRequest:
        """Parse natural language request into ProcessingRequest"""
        request_lower = user_request.lower()
        
        # Determine operation type
        if "today" in request_lower or "daily" in request_lower:
            operation = "daily_processing"
            date = datetime.now().strftime('%Y-%m-%d')
        elif "yesterday" in request_lower:
            operation = "daily_processing"
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            operation = "daily_processing"
            date = None
        
        # Determine channels
        channels = []
        if "moneypurse" in request_lower or "money purse" in request_lower:
            channels.append("moneypurse")
        if "daytrader" in request_lower or "day trader" in request_lower:
            channels.append("daytradertelugu")
        
        if not channels:
            channels = ["moneypurse", "daytradertelugu"]  # Default to both
        
        # Determine output formats
        output_formats = ["markdown", "json"]  # Default
        if "html" in request_lower:
            output_formats.append("html")
        if "pdf" in request_lower:
            output_formats.append("pdf")
        
        return ProcessingRequest(
            operation=operation,
            date=date,
            channels=channels,
            output_formats=output_formats,
            enable_translation=True,
            ai_provider="auto",
            stt_provider="auto"
        )
    
    async def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and tool health"""
        status = {
            "agent_status": "ready",
            "tools_status": {},
            "last_processing": None,
            "processing_history_count": len(self.processing_history)
        }
        
        # Check tool health
        for tool_name, tool in self.tools.items():
            health_result = await tool.health_check()
            status["tools_status"][tool_name] = {
                "healthy": health_result.success,
                "info": tool.get_info()
            }
        
        # Get last processing info
        if self.processing_history:
            last_processing = self.processing_history[-1]
            status["last_processing"] = {
                "success": last_processing.success,
                "processing_time": last_processing.processing_time,
                "videos_processed": len(last_processing.processed_videos),
                "reports_generated": len(last_processing.reports),
                "completion_time": last_processing.metadata.get("completion_time")
            }
        
        return status
    
    async def cleanup(self) -> bool:
        """Clean up all tools and resources"""
        try:
            cleanup_tasks = [tool.cleanup() for tool in self.tools.values()]
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            failed_cleanups = []
            for tool_name, result in zip(self.tools.keys(), results):
                if isinstance(result, Exception) or not result.success:
                    failed_cleanups.append(tool_name)
            
            if failed_cleanups:
                self.logger.warning(f"Failed to cleanup tools: {failed_cleanups}")
            
            self.logger.info("Agent cleanup completed")
            return len(failed_cleanups) == 0
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False


# Example usage and CLI interface
async def main():
    """Main entry point for the autonomous agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Stock News Agent")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--operation", default="daily", 
                       choices=["daily", "yesterday", "custom"],
                       help="Operation to perform")
    parser.add_argument("--date", help="Specific date to process (YYYY-MM-DD)")
    parser.add_argument("--channels", nargs="+", 
                       choices=["moneypurse", "daytradertelugu"],
                       help="Specific channels to process")
    parser.add_argument("--formats", nargs="+",
                       choices=["markdown", "html", "json", "pdf"],
                       default=["markdown", "json"],
                       help="Output formats")
    parser.add_argument("--request", help="Natural language processing request")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = AutonomousStockNewsAgent(args.config)
    
    if not await agent.initialize():
        print("❌ Failed to initialize agent")
        return
    
    print("✅ Agent initialized successfully")
    
    try:
        # Process request
        if args.request:
            # Natural language request
            result = await agent.process_request_from_natural_language(args.request)
        else:
            # Structured request
            if args.operation == "daily":
                result = await agent.process_daily_videos()
            elif args.operation == "yesterday":
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                result = await agent.process_daily_videos(yesterday)
            elif args.operation == "custom":
                request = ProcessingRequest(
                    operation="custom_processing",
                    date=args.date,
                    channels=args.channels,
                    output_formats=args.formats
                )
                result = await agent.execute_processing_request(request)
        
        # Display results
        if result.success:
            print(f"✅ Processing completed successfully!")
            print(f"📹 Videos processed: {len(result.processed_videos)}")
            print(f"📝 Transcriptions: {len(result.transcriptions)}")
            print(f"📊 Analyses: {len(result.analyses)}")
            print(f"📋 Reports generated: {len(result.reports)}")
            print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
            
            if result.reports:
                print("\n📁 Generated reports:")
                for report in result.reports:
                    print(f"   - {report}")
        else:
            print(f"❌ Processing failed:")
            for error in result.errors:
                print(f"   - {error}")
    
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
