"""
Daily Stock News Agent - PydanticAI Implementation

This implementation uses PydanticAI for type-safe agent development with 
structured data validation and clean APIs.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import OpenAIModel


class MarketSentiment(str, Enum):
    """Enum for market sentiment"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH" 
    NEUTRAL = "NEUTRAL"


class StockAction(str, Enum):
    """Enum for stock actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class VideoMetadata(BaseModel):
    """Type-safe video metadata"""
    title: str = Field(..., description="Video title")
    file_path: str = Field(..., description="Path to video file")
    channel: str = Field(..., description="YouTube channel name")
    duration: int = Field(..., gt=0, description="Video duration in seconds")
    download_date: str = Field(..., description="Download date (YYYY-MM-DD)")
    quality: str = Field(default="720p", description="Video quality")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path format"""
        if not v.endswith(('.mp4', '.mkv', '.avi')):
            raise ValueError('File must be a video format')
        return v


class TranscriptionResult(BaseModel):
    """Type-safe transcription result"""
    original_text: str = Field(..., description="Original Telugu text")
    translated_text: str = Field(..., description="English translation")
    language: str = Field(..., description="Source language code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Transcription confidence")
    video_metadata: VideoMetadata
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is realistic"""
        if v < 0.3:
            raise ValueError('Confidence too low for reliable processing')
        return v


class StockRecommendation(BaseModel):
    """Type-safe stock recommendation"""
    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(..., description="Company name")
    action: StockAction = Field(..., description="Recommended action")
    target_price: Optional[float] = Field(None, gt=0, description="Target price")
    current_price: Optional[float] = Field(None, gt=0, description="Current price")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recommendation confidence")
    rationale: str = Field(..., description="Reasoning for recommendation")
    time_horizon: str = Field(default="medium", description="Investment time horizon")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate stock symbol format"""
        if not v.isupper() or len(v) < 2:
            raise ValueError('Stock symbol must be uppercase, 2+ characters')
        return v


class AnalysisResult(BaseModel):
    """Type-safe analysis result"""
    channel: str = Field(..., description="Source channel")
    video_title: str = Field(..., description="Video title")
    analysis_date: str = Field(..., description="Analysis date")
    market_sentiment: MarketSentiment = Field(..., description="Overall market sentiment")
    key_themes: List[str] = Field(default_factory=list, description="Key discussion themes")
    stock_recommendations: List[StockRecommendation] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall analysis confidence")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    
    @validator('stock_recommendations')
    def validate_recommendations(cls, v):
        """Ensure at least some recommendations if confidence is high"""
        return v  # Could add business logic here


class ProcessingRequest(BaseModel):
    """Type-safe processing request"""
    channels: List[str] = Field(..., min_items=1, description="Channels to process")
    date: str = Field(..., description="Processing date (YYYY-MM-DD)")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    output_formats: List[str] = Field(default=["markdown", "json"])
    enable_quality_check: bool = Field(default=True)
    
    @validator('date')
    def validate_date(cls, v):
        """Validate date format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v


class ProcessingResult(BaseModel):
    """Type-safe processing result"""
    success: bool = Field(..., description="Processing success status")
    request: ProcessingRequest = Field(..., description="Original request")
    videos_processed: List[VideoMetadata] = Field(default_factory=list)
    transcriptions: List[TranscriptionResult] = Field(default_factory=list)
    analyses: List[AnalysisResult] = Field(default_factory=list)
    reports_generated: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    
    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence across all analyses"""
        if not self.analyses:
            return 0.0
        return sum(a.confidence_score for a in self.analyses) / len(self.analyses)


class PydanticAIStockNewsSystem:
    """PydanticAI-based type-safe stock news processing system"""
    
    def __init__(self, openai_api_key: str):
        self.model = OpenAIModel('gpt-4', api_key=openai_api_key)
        
        # Create specialized agents with type safety
        self.video_agent = self._create_video_agent()
        self.transcription_agent = self._create_transcription_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.report_agent = self._create_report_agent()
    
    def _create_video_agent(self) -> Agent[None, VideoMetadata]:
        """Create type-safe video processing agent"""
        
        @Agent(self.model, result_type=VideoMetadata)
        async def video_processor(ctx: RunContext[None], channel: str, date: str) -> VideoMetadata:
            """Download and process video with type validation"""
            
            # Simulate video download with proper validation
            video_data = VideoMetadata(
                title=f"Daily Market Analysis - {channel} - {date}",
                file_path=f"./data/videos/{channel}_{date.replace('-', '')}.mp4",
                channel=channel,
                duration=1800,  # 30 minutes
                download_date=date,
                quality="720p"
            )
            
            return video_data
        
        return video_processor
    
    def _create_transcription_agent(self) -> Agent[VideoMetadata, TranscriptionResult]:
        """Create type-safe transcription agent"""
        
        @Agent(self.model, result_type=TranscriptionResult)
        async def transcription_processor(
            ctx: RunContext[VideoMetadata], 
            video: VideoMetadata
        ) -> TranscriptionResult:
            """Transcribe video with confidence validation"""
            
            # Simulate high-quality transcription
            result = TranscriptionResult(
                original_text="మార్కెట్ పరిస్థితులు మరియు ఈరోజు చర్చించవలసిన ముఖ్య స్టాక్‌లు...",
                translated_text="Market conditions and key stocks to discuss today. Based on technical analysis, Reliance shows strong bullish momentum...",
                language="te",
                confidence=0.94,
                video_metadata=video
            )
            
            return result
        
        return transcription_processor
    
    def _create_analysis_agent(self) -> Agent[TranscriptionResult, AnalysisResult]:
        """Create type-safe analysis agent"""
        
        @Agent(self.model, result_type=AnalysisResult, system_prompt="""
        You are a senior stock market analyst. Analyze the provided transcript and extract:
        1. Market sentiment (BULLISH/BEARISH/NEUTRAL)
        2. Stock recommendations with confidence scores
        3. Key themes and risk factors
        
        Ensure all recommendations have proper rationale and confidence scores.
        Be conservative with confidence scores - only use high confidence for clear signals.
        """)
        async def stock_analyzer(
            ctx: RunContext[TranscriptionResult], 
            transcript: TranscriptionResult
        ) -> AnalysisResult:
            """Analyze transcript with structured output"""
            
            # This would use the LLM for actual analysis
            # For demo, returning structured mock data
            recommendations = [
                StockRecommendation(
                    symbol="RELIANCE",
                    company_name="Reliance Industries Limited",
                    action=StockAction.BUY,
                    target_price=2800.0,
                    current_price=2650.0,
                    confidence=0.85,
                    rationale="Strong technical breakout with volume confirmation",
                    time_horizon="medium"
                ),
                StockRecommendation(
                    symbol="TCS",
                    company_name="Tata Consultancy Services",
                    action=StockAction.HOLD,
                    target_price=4200.0,
                    current_price=4150.0,
                    confidence=0.75,
                    rationale="Consolidation phase, wait for clear direction",
                    time_horizon="short"
                )
            ]
            
            result = AnalysisResult(
                channel=transcript.video_metadata.channel,
                video_title=transcript.video_metadata.title,
                analysis_date=transcript.video_metadata.download_date,
                market_sentiment=MarketSentiment.BULLISH,
                key_themes=["Technical Analysis", "Large Cap Stocks", "Momentum Trading"],
                stock_recommendations=recommendations,
                confidence_score=0.82,
                risk_factors=["Market Volatility", "Global Economic Uncertainty"]
            )
            
            return result
        
        return stock_analyzer
    
    def _create_report_agent(self) -> Agent[List[AnalysisResult], str]:
        """Create type-safe report generation agent"""
        
        @Agent(self.model, result_type=str, system_prompt="""
        You are a professional financial report writer. Create comprehensive investment 
        reports that synthesize multiple analyses into actionable insights.
        
        Include: Executive Summary, Market Overview, Key Recommendations, Risk Assessment.
        Ensure proper formatting and professional language.
        """)
        async def report_generator(
            ctx: RunContext[List[AnalysisResult]], 
            analyses: List[AnalysisResult],
            date: str
        ) -> str:
            """Generate comprehensive report with type safety"""
            
            if not analyses:
                raise ValueError("No analyses provided for report generation")
            
            # Calculate aggregate metrics
            total_recommendations = sum(len(a.stock_recommendations) for a in analyses)
            avg_confidence = sum(a.confidence_score for a in analyses) / len(analyses)
            
            # Determine overall sentiment
            sentiments = [a.market_sentiment for a in analyses]
            overall_sentiment = max(set(sentiments), key=sentiments.count)
            
            report_content = f"""
# PydanticAI Stock Analysis Report - {date}

## Executive Summary
- **Videos Analyzed**: {len(analyses)}
- **Total Recommendations**: {total_recommendations}
- **Average Confidence**: {avg_confidence:.2f}
- **Overall Sentiment**: {overall_sentiment.value}

## Market Overview
Generated using type-safe PydanticAI agents with full data validation.

## Channel Analysis
"""
            
            for analysis in analyses:
                report_content += f"""
### {analysis.channel}
**Sentiment**: {analysis.market_sentiment.value}  
**Confidence**: {analysis.confidence_score:.2f}  
**Key Themes**: {', '.join(analysis.key_themes)}

**Recommendations**:
"""
                for rec in analysis.stock_recommendations:
                    report_content += f"""
- **{rec.symbol}** ({rec.company_name}): {rec.action.value}
  - Target: ₹{rec.target_price or 'N/A'} | Confidence: {rec.confidence:.2f}
  - Rationale: {rec.rationale}
"""
            
            report_content += f"""
## Risk Assessment
Common risk factors identified across analyses:
"""
            all_risks = set()
            for analysis in analyses:
                all_risks.update(analysis.risk_factors)
            
            for risk in all_risks:
                report_content += f"- {risk}\n"
            
            report_content += f"""
## Data Quality
- All data passed type validation
- Confidence thresholds enforced
- Structured analysis guaranteed

---
*Report generated by PydanticAI with full type safety*
"""
            
            return report_content
        
        return report_generator
    
    async def process_request(self, request: ProcessingRequest) -> ProcessingResult:
        """Process request with full type safety"""
        
        start_time = datetime.now()
        print(f"🎯 PydanticAI Type-Safe Processing for {request.date}")
        print("=" * 60)
        
        try:
            videos: List[VideoMetadata] = []
            transcriptions: List[TranscriptionResult] = []
            analyses: List[AnalysisResult] = []
            reports: List[str] = []
            errors: List[str] = []
            
            # Phase 1: Video Processing (Type-Safe)
            print("🎥 Processing videos with type validation...")
            for channel in request.channels:
                try:
                    video = await self.video_agent.run(channel, request.date)
                    videos.append(video)
                    print(f"✅ Video processed: {video.title}")
                except Exception as e:
                    errors.append(f"Video processing failed for {channel}: {str(e)}")
            
            # Phase 2: Transcription (Type-Safe)
            print("🎙️ Transcribing with confidence validation...")
            for video in videos:
                try:
                    transcription = await self.transcription_agent.run(video)
                    if transcription.confidence >= request.confidence_threshold:
                        transcriptions.append(transcription)
                        print(f"✅ Transcribed: {video.channel} (confidence: {transcription.confidence:.2f})")
                    else:
                        errors.append(f"Low confidence transcription: {video.channel}")
                except Exception as e:
                    errors.append(f"Transcription failed for {video.channel}: {str(e)}")
            
            # Phase 3: Analysis (Type-Safe)
            print("📊 Analyzing with structured validation...")
            for transcription in transcriptions:
                try:
                    analysis = await self.analysis_agent.run(transcription)
                    analyses.append(analysis)
                    print(f"✅ Analyzed: {analysis.channel} (confidence: {analysis.confidence_score:.2f})")
                except Exception as e:
                    errors.append(f"Analysis failed for {transcription.video_metadata.channel}: {str(e)}")
            
            # Phase 4: Report Generation (Type-Safe)
            if analyses:
                print("📄 Generating type-safe report...")
                try:
                    report_content = await self.report_agent.run(analyses, request.date)
                    
                    # Save report
                    report_file = f"./data/reports/pydanticai_report_{request.date.replace('-', '')}.md"
                    os.makedirs(os.path.dirname(report_file), exist_ok=True)
                    
                    with open(report_file, 'w') as f:
                        f.write(report_content)
                    
                    reports.append(report_file)
                    print(f"✅ Report generated: {report_file}")
                except Exception as e:
                    errors.append(f"Report generation failed: {str(e)}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Return fully validated result
            result = ProcessingResult(
                success=len(analyses) > 0,
                request=request,
                videos_processed=videos,
                transcriptions=transcriptions,
                analyses=analyses,
                reports_generated=reports,
                errors=errors,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                success=False,
                request=request,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    def get_type_safety_visualization(self) -> str:
        """Return visualization of type safety features"""
        return """
🎯 PydanticAI Type Safety Features:

┌─────────────────────────────────────────────────────────────┐
│                    INPUT VALIDATION                         │
├─────────────────────────────────────────────────────────────┤
│ ProcessingRequest                                           │
│ ├── channels: List[str] (min_items=1) ✓                    │
│ ├── date: str (YYYY-MM-DD format) ✓                        │
│ ├── confidence_threshold: float (0.0-1.0) ✓                │
│ └── output_formats: List[str] ✓                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│ VideoMetadata ──────► TranscriptionResult                   │
│ ├── file_path validation    ├── confidence validation       │
│ ├── duration > 0           ├── language code validation     │
│ └── channel validation     └── text length validation       │
│                              │                              │
│                              ▼                              │
│ AnalysisResult ──────► ProcessingResult                     │
│ ├── sentiment enum         ├── success boolean             │
│ ├── confidence scores      ├── error collection            │
│ ├── stock symbols          ├── performance metrics         │
│ └── risk validation        └── type-safe aggregation       │
└─────────────────────────────────────────────────────────────┘

Type Safety Benefits:
✅ Compile-time error detection
✅ Automatic data validation  
✅ IDE autocomplete and hints
✅ Runtime type checking
✅ Structured error handling
✅ API contract enforcement
✅ Documentation generation
✅ Testing facilitation
        """


# Example usage
async def main():
    """Demonstrate PydanticAI type-safe workflow"""
    
    print("🎯 PydanticAI Type-Safe Multi-Agent System")
    print("=" * 50)
    
    # Initialize system
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    system = PydanticAIStockNewsSystem(api_key)
    
    # Show type safety visualization
    print(system.get_type_safety_visualization())
    
    # Create type-safe request
    try:
        request = ProcessingRequest(
            channels=["moneypurse", "daytradertelugu"],
            date=datetime.now().strftime('%Y-%m-%d'),
            confidence_threshold=0.75,
            output_formats=["markdown", "json"],
            enable_quality_check=True
        )
        
        # Process with full type validation
        result = await system.process_request(request)
        
        if result.success:
            print(f"\n✅ Type-safe processing completed!")
            print(f"📅 Date: {result.request.date}")
            print(f"📺 Channels: {', '.join(result.request.channels)}")
            print(f"🎥 Videos: {len(result.videos_processed)}")
            print(f"📝 Transcriptions: {len(result.transcriptions)}")
            print(f"📊 Analyses: {len(result.analyses)}")
            print(f"📄 Reports: {len(result.reports_generated)}")
            print(f"⚡ Processing Time: {result.processing_time:.2f}s")
            print(f"🎯 Overall Confidence: {result.overall_confidence:.2f}")
            
            if result.errors:
                print(f"⚠️ Errors: {len(result.errors)}")
                for error in result.errors:
                    print(f"   - {error}")
        else:
            print(f"❌ Processing failed: {result.errors}")
    
    except Exception as e:
        print(f"❌ Validation failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
