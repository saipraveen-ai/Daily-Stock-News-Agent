"""
Content Analysis Tool for Daily Stock News Agent

This tool analyzes transcribed content from Telugu stock market videos,
extracts key insights, and structures the information for report generation.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_tool import BaseTool, ToolResult, ToolConfig, ToolCategory, ToolPriority

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of content analysis"""
    STOCK_RECOMMENDATIONS = "stock_recommendations"
    MARKET_SENTIMENT = "market_sentiment"
    SECTOR_ANALYSIS = "sector_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    NEWS_EVENTS = "news_events"
    KEY_SEGMENTS = "key_segments"


@dataclass
class StockRecommendation:
    """Stock recommendation data structure"""
    symbol: str
    company_name: str
    action: str  # BUY, SELL, HOLD
    target_price: Optional[float] = None
    current_price: Optional[float] = None
    rationale: str = ""
    confidence: float = 0.0
    time_horizon: str = "short"  # short, medium, long
    risk_level: str = "medium"  # low, medium, high


@dataclass
class MarketSegment:
    """Market analysis segment"""
    timestamp: str
    topic: str
    content: str
    category: AnalysisType
    importance: float = 0.5  # 0.0 to 1.0
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


@dataclass
class AnalysisResult:
    """Result from content analysis"""
    channel: str
    video_title: str
    analysis_date: str
    market_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    key_themes: List[str]
    stock_recommendations: List[StockRecommendation]
    sector_insights: Dict[str, Any]
    technical_signals: Dict[str, Any]
    market_segments: List[MarketSegment]
    chapter_summary: List[Dict[str, Any]] = None
    confidence_score: float = 0.0
    video_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.chapter_summary is None:
            self.chapter_summary = []


class ContentAnalysisTool(BaseTool):
    """
    Tool for analyzing transcribed stock market content.
    
    Features:
    - Extract stock recommendations (buy/sell/hold)
    - Identify market sentiment and themes
    - Parse sector-wise analysis
    - Extract technical analysis insights
    - Segment content into key topics
    - Multi-language support (Telugu/English)
    """
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        
        # Load environment variables if dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        self.ai_provider = config.settings.get('ai_provider', 'local')
        # Get API key from config first, then fallback to environment variable
        self.ai_api_key = config.settings.get('ai_api_key') or os.getenv('OPENAI_API_KEY')
        self.openai_client = None
        
    async def initialize(self) -> ToolResult:
        """Initialize the content analysis tool"""
        try:
            # Initialize OpenAI client if available and configured
            if OPENAI_AVAILABLE and self.ai_provider == 'openai' and self.ai_api_key:
                self.openai_client = openai.OpenAI(api_key=self.ai_api_key)
                self.logger.info("OpenAI client initialized successfully")
            elif self.ai_provider != 'local' and self.ai_api_key:
                self.logger.info(f"AI provider {self.ai_provider} configured")
            else:
                self.logger.warning("OpenAI API key not configured - tool will require OpenAI for analysis")
            
            self._is_initialized = True
            
            return ToolResult(
                success=True,
                data={"message": "Content analysis tool initialized"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize content analysis tool: {e}")
            return ToolResult(
                success=False,
                error_message=f"Initialization failed: {str(e)}"
            )
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute content analysis on transcribed text.
        
        Args:
            transcript_file: str - Path to transcript file
            transcript_text: str - Direct transcript text
            channel: str - Channel name
            video_title: str - Video title
            analysis_types: List[str] - Types of analysis to perform
            video_metadata: Dict[str, Any] - Video metadata for enhanced analysis
            target_date: str - Target date for analysis (YYYY-MM-DD format)
            
        Returns:
            ToolResult with analysis results
        """
        transcript_text = kwargs.get('transcript_text')
        transcript_file = kwargs.get('transcript_file')
        video_metadata = kwargs.get('video_metadata', {})
        target_date = kwargs.get('target_date')  # Add target_date parameter
        
        # Get transcript text
        if transcript_file and os.path.exists(transcript_file):
            transcript_text = await self._load_transcript(transcript_file)
        
        if not transcript_text:
            return ToolResult(
                success=False,
                error_message="No transcript text provided"
            )
        
        channel = kwargs.get('channel', 'Unknown')
        video_title = kwargs.get('video_title', 'Unknown')
        analysis_types = kwargs.get('analysis_types', ['all'])
        
        try:
            # Perform comprehensive analysis
            analysis_result = await self._analyze_content(
                transcript_text, channel, video_title, analysis_types, video_metadata, target_date
            )
            
            # Save analysis results
            output_file = await self._save_analysis(analysis_result)
            
            return ToolResult(
                success=True,
                data={
                    "analysis": analysis_result,
                    "output_file": output_file
                },
                metadata={
                    "channel": channel,
                    "video_title": video_title
                }
            )
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return ToolResult(
                success=False,
                error_message=f"Analysis failed: {str(e)}"
            )
    
    async def _load_transcript(self, transcript_file: str) -> str:
        """Load transcript from file"""
        try:
            if transcript_file.endswith('.json'):
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('translated_text') or data.get('original_text', '')
            else:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load transcript: {e}")
            return ""
    
    async def _analyze_content(
        self, 
        text: str, 
        channel: str, 
        video_title: str,
        analysis_types: List[str],
        video_metadata: Dict[str, Any] = None,
        target_date: str = None
    ) -> AnalysisResult:
        """Perform comprehensive content analysis with video metadata"""
        
        if video_metadata is None:
            video_metadata = {}
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        # Extract market segments
        segments = await self._extract_segments(cleaned_text)
        
        # Perform different types of analysis
        stock_recommendations = []
        sector_insights = {}
        technical_signals = {}
        market_sentiment = "NEUTRAL"
        key_themes = []
        
        # Require OpenAI for LLM analysis
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        
        # Step 1: Clean up the transcript using metadata context
        cleaned_transcript = await self._cleanup_transcript_with_llm(text, channel, video_title, video_metadata)
        if not cleaned_transcript:
            self.logger.warning("Transcript cleanup failed, using original text")
            cleaned_transcript = cleaned_text
        
        # Step 2: Perform analysis on the cleaned transcript
        llm_analysis = await self._analyze_with_llm(cleaned_transcript, channel, video_title, video_metadata)
        if llm_analysis:
            stock_recommendations = llm_analysis.get('stock_recommendations', [])
            market_sentiment = llm_analysis.get('market_sentiment', 'NEUTRAL')
            key_themes = llm_analysis.get('key_themes', [])
            sector_insights = llm_analysis.get('sector_insights', {})
            technical_signals = llm_analysis.get('technical_signals', {})
            chapter_summary = llm_analysis.get('chapter_summary', [])
        else:
            self.logger.error("LLM analysis failed and no fallback available")
            raise RuntimeError("Failed to analyze content with LLM. Please check your OpenAI API configuration.")
        
        # Calculate overall confidence score
        confidence_score = await self._calculate_confidence(
            stock_recommendations, segments, text
        )
        
        return AnalysisResult(
            channel=channel,
            video_title=video_title,
            analysis_date=target_date or self._get_current_date(),
            market_sentiment=market_sentiment,
            key_themes=key_themes,
            stock_recommendations=stock_recommendations,
            sector_insights=sector_insights,
            technical_signals=technical_signals,
            market_segments=segments,
            chapter_summary=chapter_summary,
            confidence_score=confidence_score,
            video_metadata=video_metadata  # Include metadata in results
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Convert to lowercase for pattern matching
        return text.lower()
    
    async def _extract_segments(self, text: str) -> List[MarketSegment]:
        """Extract key content segments from the text"""
        segments = []
        
        # Split text into paragraphs or sentences
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) < 20:  # Skip very short segments
                continue
            
            # Classify segment type
            category = self._classify_segment(sentence)
            importance = self._calculate_segment_importance(sentence)
            keywords = self._extract_keywords(sentence)
            
            segment = MarketSegment(
                timestamp=f"{i:03d}",
                topic=self._generate_topic_title(sentence),
                content=sentence.strip(),
                category=category,
                importance=importance,
                keywords=keywords
            )
            
            segments.append(segment)
        
        # Sort by importance and return top segments
        segments.sort(key=lambda x: x.importance, reverse=True)
        return segments[:30]  # Top 30 segments
    
    def _classify_segment(self, text: str) -> AnalysisType:
        """Classify a text segment by type using simple heuristics"""
        text_lower = text.lower()
        
        # Check for stock-related terms
        stock_terms = ['stock', 'share', 'buy', 'sell', 'invest', 'target', 'price']
        if any(term in text_lower for term in stock_terms):
            return AnalysisType.STOCK_RECOMMENDATIONS
        
        # Check for technical analysis terms
        technical_terms = ['support', 'resistance', 'breakout', 'chart', 'trend']
        if any(term in text_lower for term in technical_terms):
            return AnalysisType.TECHNICAL_ANALYSIS
        
        # Check for sector terms
        sector_terms = ['sector', 'banking', 'it', 'pharma', 'auto', 'fmcg']
        if any(term in text_lower for term in sector_terms):
            return AnalysisType.SECTOR_ANALYSIS
        
        # Check for sentiment terms
        sentiment_terms = ['bullish', 'bearish', 'positive', 'negative', 'market']
        if any(term in text_lower for term in sentiment_terms):
            return AnalysisType.MARKET_SENTIMENT
        
        return AnalysisType.NEWS_EVENTS
    
    def _calculate_segment_importance(self, text: str) -> float:
        """Calculate importance score for a segment using simple metrics"""
        importance = 0.0
        text_lower = text.lower()
        
        # Numbers (prices, percentages) increase importance
        if re.search(r'\d+(?:\.\d+)?%', text):
            importance += 0.3  # Percentages are important
        elif re.search(r'\d+', text):
            importance += 0.1  # Any numbers
        
        # Financial terms increase importance
        financial_terms = ['stock', 'price', 'target', 'profit', 'loss', 'growth', 'revenue']
        for term in financial_terms:
            if term in text_lower:
                importance += 0.1
        
        # Company mentions increase importance
        if re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', text):  # Capitalized words (potential company names)
            importance += 0.2
        
        # Longer segments get slight boost (more information)
        if len(text) > 100:
            importance += 0.1
        
        return min(importance, 1.0)  # Cap at 1.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text using simple patterns"""
        keywords = []
        
        # Extract potential stock symbols (all caps, 3-10 characters)
        stock_symbols = re.findall(r'\b[A-Z]{3,10}\b', text)
        keywords.extend(stock_symbols)
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        keywords.extend(percentages)
        
        # Extract important financial terms
        text_lower = text.lower()
        financial_keywords = ['buy', 'sell', 'hold', 'target', 'support', 'resistance', 'bullish', 'bearish']
        for keyword in financial_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
    
    def _generate_topic_title(self, text: str) -> str:
        """Generate a topic title for a segment"""
        # Simple approach: take first 50 characters
        title = text[:50].strip()
        if len(text) > 50:
            title += "..."
        return title.capitalize()
    
    def _get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d')
    
    async def _calculate_confidence(
        self, 
        recommendations: List[StockRecommendation],
        segments: List[MarketSegment],
        text: str
    ) -> float:
        """Calculate overall confidence score for the analysis"""
        # For LLM-based analysis, confidence comes primarily from the LLM itself
        # and the quality of the input text
        confidence = 0.5  # Base confidence for LLM analysis
        
        # Additional confidence from text length (more context = better analysis)
        if len(text) > 2000:
            confidence += 0.3
        elif len(text) > 1000:
            confidence += 0.2
        elif len(text) > 500:
            confidence += 0.1
        
        # Additional confidence from number of segments (more structure = better analysis)
        if len(segments) > 10:
            confidence += 0.2
        elif len(segments) > 5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _cleanup_transcript_with_llm(self, text: str, channel: str, video_title: str, video_metadata: Dict[str, Any]) -> Optional[str]:
        """Clean up the transcript using LLM with metadata context, handling long transcripts with chunking"""
        try:
            print(f"\n🧹 CLEANING UP TRANSCRIPT FOR {channel.upper()}")
            print(f"📊 Original Length: {len(text)} characters")
            
            # Handle very long transcripts with chunking
            if len(text) > 12000:
                print(f"📚 LONG TRANSCRIPT DETECTED - Using chunked preprocessing")
                return await self._cleanup_long_transcript_chunked(text, channel, video_title, video_metadata)
            
            # Standard preprocessing for shorter transcripts
            cleanup_prompt = self._create_cleanup_prompt(text, channel, video_title, video_metadata)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert transcript editor specializing in Telugu financial content. Clean up garbled speech-to-text transcripts to make them coherent and useful for financial analysis while preserving all important details."
                    },
                    {
                        "role": "user",
                        "content": cleanup_prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent cleanup
                max_tokens=4000   # Allow more tokens for cleaned transcript
            )
            
            cleaned_transcript = response.choices[0].message.content.strip()
            
            print(f"✨ Cleaned Length: {len(cleaned_transcript)} characters")
            print(f"📈 Length Improvement: {len(cleaned_transcript) - len(text):+d} characters")
            
            # Save the cleaned transcript for debugging
            await self._save_cleaned_transcript(cleaned_transcript, channel, video_title)
            
            return cleaned_transcript
            
        except Exception as e:
            self.logger.error(f"Transcript cleanup failed: {e}")
            print(f"❌ TRANSCRIPT CLEANUP ERROR: {e}")
            return None
    
    def _create_cleanup_prompt(self, text: str, channel: str, video_title: str, video_metadata: Dict[str, Any]) -> str:
        """Create prompt for transcript cleanup"""
        
        # Extract metadata context
        metadata_context = ""
        if video_metadata:
            video_info = video_metadata.get('video_info', {})
            duration = video_info.get('duration', 'Unknown')
            description = video_info.get('description', '')
            
            # Extract chapters for context
            chapters_info = ""
            content_structure = video_metadata.get('content_structure', {})
            chapters = content_structure.get('chapters', [])
            if chapters:
                chapter_list = []
                for chapter in chapters:
                    title = chapter.get('title', '')
                    start_time = chapter.get('start_formatted', chapter.get('start_time', ''))
                    chapter_list.append(f"  - {start_time}: {title}")
                chapters_info = f"\nVideo Chapters:\n" + "\n".join(chapter_list)
            
            # Extract description timestamps
            description_context = ""
            if description and len(description) > 100:
                timestamp_pattern = r'(\d{1,2}:\d{2}[^\n]*)'
                timestamps = re.findall(timestamp_pattern, description)
                if timestamps:
                    description_context = f"\nDescription Highlights:\n" + "\n".join(f"  - {ts}" for ts in timestamps[:15])
            
            metadata_context = f"""
Video Metadata:
- Duration: {duration} seconds
- Channel: {channel}
- Title: {video_title}{chapters_info}{description_context}
"""
        
        # For standard cleanup, use first 10000 characters (increased from 8000)
        cleanup_text = text[:10000] + "..." if len(text) > 10000 else text
        
        prompt = f"""
You are cleaning up a garbled speech-to-text transcript from a Telugu financial YouTube channel. The transcript contains many errors from automatic transcription, but it has valuable stock market information.

{metadata_context}

Original Garbled Transcript:
{cleanup_text}

CLEANUP INSTRUCTIONS:
1. **Fix speech-to-text errors** - Correct obvious misheard words and phrases
2. **Structure the content** - Organize into coherent paragraphs based on the chapter information
3. **Preserve financial data** - Keep ALL stock names, numbers, percentages, prices, targets
4. **Maintain context** - Use the video metadata and chapters to understand the flow
5. **Keep it comprehensive** - Don't shorten significantly, just make it readable
6. **Focus on stock content** - Emphasize stock recommendations, market analysis, company discussions

WHAT TO PRESERVE:
- All company names and stock symbols
- All numerical data (prices, percentages, targets, revenues)
- Market sentiment and analysis
- Investment recommendations (buy/sell/hold)
- Technical analysis terms
- Financial metrics and ratios

WHAT TO FIX:
- Grammar and sentence structure
- Obvious speech-to-text errors
- Repetitive or garbled sections
- Organize by topics/chapters when possible

Please provide a cleaned up version that maintains the original length and detail but is much more readable and coherent for financial analysis.
"""
        return prompt
    
    async def _save_cleaned_transcript(self, cleaned_transcript: str, channel: str, video_title: str) -> None:
        """Save cleaned transcript for debugging"""
        try:
            debug_dir = os.path.join('./data/debug', self._get_current_date())
            os.makedirs(debug_dir, exist_ok=True)
            
            cleaned_file = os.path.join(debug_dir, f"{channel}_cleaned_transcript.txt")
            with open(cleaned_file, 'w', encoding='utf-8') as f:
                f.write(f"=== CLEANED TRANSCRIPT FOR {channel.upper()} ===\n")
                f.write(f"Video Title: {video_title}\n")
                f.write(f"Cleaned Length: {len(cleaned_transcript)} characters\n")
                f.write(f"Timestamp: {self._get_current_date()}\n")
                f.write("="*80 + "\n\n")
                f.write(cleaned_transcript)
            
            print(f"🧹 CLEANED TRANSCRIPT SAVED: {cleaned_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save cleaned transcript: {e}")

    async def _cleanup_long_transcript_chunked(self, text: str, channel: str, video_title: str, video_metadata: Dict[str, Any]) -> Optional[str]:
        """Clean up very long transcripts by processing in chunks"""
        try:
            # Extract metadata context for consistent processing
            metadata_context = self._extract_metadata_context(video_metadata, channel, video_title)
            
            # Split transcript into chunks (8000 chars each with overlap)
            chunk_size = 8000
            overlap = 500  # Overlap to maintain context
            chunks = []
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
            
            print(f"📚 Processing {len(chunks)} chunks of ~{chunk_size} characters each")
            
            cleaned_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                
                chunk_prompt = self._create_chunk_cleanup_prompt(chunk, metadata_context, i+1, len(chunks))
                
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert transcript editor specializing in Telugu financial content. Clean up garbled speech-to-text transcript chunks while preserving financial information."
                            },
                            {
                                "role": "user",
                                "content": chunk_prompt
                            }
                        ],
                        temperature=0.1,
                        max_tokens=2000
                    )
                    
                    cleaned_chunk = response.choices[0].message.content.strip()
                    cleaned_chunks.append(cleaned_chunk)
                    print(f"✅ Chunk {i+1} cleaned: {len(chunk)} → {len(cleaned_chunk)} chars")
                    
                except Exception as e:
                    print(f"❌ Failed to clean chunk {i+1}: {e}")
                    # Use original chunk if cleaning fails
                    cleaned_chunks.append(chunk)
            
            # Combine all cleaned chunks
            full_cleaned_transcript = "\n\n".join(cleaned_chunks)
            
            print(f"✨ CHUNKED CLEANUP COMPLETE:")
            print(f"   Original: {len(text)} characters")
            print(f"   Cleaned: {len(full_cleaned_transcript)} characters")
            print(f"   Improvement: {len(full_cleaned_transcript) - len(text):+d} characters")
            
            # Save the cleaned transcript
            await self._save_cleaned_transcript(full_cleaned_transcript, channel, video_title)
            
            return full_cleaned_transcript
            
        except Exception as e:
            self.logger.error(f"Chunked transcript cleanup failed: {e}")
            print(f"❌ CHUNKED CLEANUP ERROR: {e}")
            return None

    def _extract_metadata_context(self, video_metadata: Dict[str, Any], channel: str, video_title: str) -> str:
        """Extract metadata context for use in chunk processing"""
        metadata_context = f"Channel: {channel}\nTitle: {video_title}\n"
        
        if video_metadata:
            video_info = video_metadata.get('video_info', {})
            duration = video_info.get('duration', 'Unknown')
            
            # Extract key chapters for context
            content_structure = video_metadata.get('content_structure', {})
            chapters = content_structure.get('chapters', [])
            if chapters:
                chapter_list = []
                for chapter in chapters[:15]:  # Top 15 chapters
                    title = chapter.get('title', '')
                    start_time = chapter.get('start_formatted', chapter.get('start_time', ''))
                    chapter_list.append(f"  - {start_time}: {title}")
                
                metadata_context += f"\nVideo Duration: {duration} seconds\n"
                metadata_context += f"Key Chapters:\n" + "\n".join(chapter_list)
        
        return metadata_context

    def _create_chunk_cleanup_prompt(self, chunk: str, metadata_context: str, chunk_num: int, total_chunks: int) -> str:
        """Create cleanup prompt for individual chunks"""
        prompt = f"""
You are cleaning up a garbled Telugu financial video transcript chunk {chunk_num} of {total_chunks}.

{metadata_context}

CHUNK {chunk_num}/{total_chunks} TO CLEAN:
{chunk}

CLEANUP INSTRUCTIONS:
1. **Fix speech-to-text errors** - Correct obvious misheard words
2. **Preserve financial data** - Keep ALL stock names, numbers, percentages, prices
3. **Maintain sentence flow** - Make it readable but don't shorten significantly  
4. **Keep technical terms** - Preserve financial and stock market terminology
5. **Structure logically** - Organize into coherent paragraphs

PRESERVE:
- Company names and stock symbols
- All numerical data (prices, percentages, targets)
- Market analysis and recommendations
- Technical terms and financial metrics

FOCUS: Make this chunk coherent and useful for financial analysis while preserving all important details.

Return the cleaned chunk only:
"""
        return prompt

    async def _analyze_with_llm(self, text: str, channel: str, video_title: str, video_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform comprehensive analysis using OpenAI LLM"""
        try:
            # Create comprehensive prompt for stock analysis
            prompt = self._create_stock_analysis_prompt(text, channel, video_title, video_metadata)
            
            # Save the complete transcript and prompt for debugging
            await self._save_debug_files(text, prompt, channel, video_title, video_metadata)
            
            # DEBUG: Log the exact prompt being sent to LLM
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: LLM ANALYSIS INPUT FOR {channel.upper()}")
            print(f"{'='*80}")
            print(f"📝 Video Title: {video_title}")
            print(f"📊 Text Length: {len(text)} characters")
            print(f"📋 Metadata: {video_metadata}")
            print(f"\n🎯 PROMPT BEING SENT TO LLM:")
            print(f"{'−'*50}")
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print(f"{'−'*50}")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Indian stock market analyst with deep knowledge of Telugu financial channels. Provide detailed, actionable analysis from video transcripts. Always respond with valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000
            )
            
            # Parse the LLM response
            llm_response = response.choices[0].message.content
            
            # DEBUG: Log the exact LLM response
            print(f"\n🤖 LLM RAW RESPONSE:")
            print(f"{'−'*50}")
            print(llm_response[:2000] + "..." if len(llm_response) > 2000 else llm_response)
            print(f"{'−'*50}")
            
            # Save the LLM response to debug files
            await self._save_llm_response(llm_response, channel, video_title)
            
            analysis_data = json.loads(llm_response)
            
            # DEBUG: Log parsed JSON data
            print(f"\n📋 PARSED JSON DATA:")
            print(f"{'−'*50}")
            print(json.dumps(analysis_data, indent=2)[:1500] + "..." if len(str(analysis_data)) > 1500 else json.dumps(analysis_data, indent=2))
            print(f"{'='*80}\n")
            
            # Convert LLM response to our format
            return self._convert_llm_response(analysis_data)
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            print(f"❌ LLM ANALYSIS ERROR: {e}")
            return None
    
    def _create_stock_analysis_prompt(self, text: str, channel: str, video_title: str, video_metadata: Dict[str, Any]) -> str:
        """Create comprehensive prompt for stock analysis"""
        
        # Include metadata context if available
        metadata_context = ""
        if video_metadata:
            # Extract basic video info
            video_info = video_metadata.get('video_info', {})
            duration = video_info.get('duration', video_metadata.get('duration', 'Unknown'))
            view_count = video_info.get('view_count', video_metadata.get('view_count', 'Unknown'))
            upload_date = video_info.get('upload_date', video_metadata.get('upload_date', 'Unknown'))
            description = video_info.get('description', '')
            
            # Extract engagement metrics
            engagement = video_metadata.get('engagement_metrics', {})
            like_count = engagement.get('like_count', 'Unknown')
            comment_count = engagement.get('comment_count', 'Unknown')
            
            # Extract chapters/content structure for context
            chapters_info = ""
            content_structure = video_metadata.get('content_structure', {})
            chapters = content_structure.get('chapters', [])
            if chapters:
                chapter_list = []
                for chapter in chapters[:10]:  # First 10 chapters
                    title = chapter.get('title', '')
                    start_time = chapter.get('start_formatted', chapter.get('start_time', ''))
                    chapter_list.append(f"  - {start_time}: {title}")
                chapters_info = f"\nVideo Chapters:\n" + "\n".join(chapter_list)
            
            # Extract key description snippets (timestamps and stock mentions)
            description_context = ""
            if description and len(description) > 100:
                # Extract timestamp sections from description
                timestamp_pattern = r'(\d{1,2}:\d{2}[^\n]*)'
                timestamps = re.findall(timestamp_pattern, description)
                if timestamps:
                    description_context = f"\nDescription Highlights:\n" + "\n".join(f"  - {ts}" for ts in timestamps[:10])
            
            metadata_context = f"""
Video Metadata:
- Duration: {duration} seconds
- Views: {view_count}
- Upload Date: {upload_date}
- Likes: {like_count}
- Comments: {comment_count}{chapters_info}{description_context}
"""
        
        # Truncate text if too long (keep first 4000 chars for context)
        analysis_text = text[:4000] + "..." if len(text) > 4000 else text
        
        prompt = f"""
You are an expert Indian stock market analyst. Analyze this Telugu YouTube video transcript for investment insights.

Channel: {channel}
Title: {video_title}
{metadata_context}

Transcript:
{analysis_text}

Analyze the content and extract:

1. **Specific stock recommendations** with company names (NSE/BSE symbols if mentioned)
2. **Price targets, percentage gains, or numerical predictions**
3. **Market sentiment and reasoning**
4. **Investment timeframe** (short/medium/long term)
5. **Risk factors or warnings** mentioned
6. **Sector analysis or themes**
7. **Technical analysis signals** (support, resistance, breakouts, etc.)
8. **Chapter-wise summary** of the video content based on the provided metadata chapters

IMPORTANT CONTEXT:
- Use the video metadata (duration, engagement, chapters) to understand the content structure
- If chapters are provided, correlate stock mentions with specific time segments and create summaries for each chapter
- High engagement (views, likes) may indicate important market insights
- Pay attention to timestamp information in the description for specific stock discussions
- Provide a comprehensive chapter-wise breakdown that maps the transcript content to the video structure

Focus on extracting actionable investment information. If no clear recommendations exist, indicate that in the response.

Respond in this exact JSON format:
{{
  "stock_recommendations": [
    {{
      "symbol": "STOCK_SYMBOL",
      "company_name": "Full Company Name",
      "action": "BUY/SELL/HOLD",
      "target_price": "if mentioned or null",
      "rationale": "reason for recommendation",
      "confidence": 0.0-1.0,
      "timeframe": "short/medium/long"
    }}
  ],
  "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
  "confidence_score": 0.0-1.0,
  "key_themes": ["theme1", "theme2", "theme3"],
  "sector_insights": {{
    "IT": {{"mentions": 0, "sentiment": "NEUTRAL", "keywords_found": []}},
    "Banking": {{"mentions": 0, "sentiment": "NEUTRAL", "keywords_found": []}}
  }},
  "technical_signals": {{
    "support": {{"found": false, "keywords": [], "context": []}},
    "resistance": {{"found": false, "keywords": [], "context": []}}
  }},
  "chapter_summary": [
    {{
      "chapter_title": "Chapter Name",
      "time_range": "00:00 - 08:53",
      "key_points": ["point1", "point2", "point3"],
      "stocks_mentioned": ["STOCK1", "STOCK2"],
      "main_topics": ["topic1", "topic2"],
      "summary": "Brief summary of what was discussed in this chapter"
    }}
  ],
  "risks_mentioned": ["risk1", "risk2"],
  "price_targets_mentioned": ["target1", "target2"],
  "numerical_predictions": ["prediction1", "prediction2"]
}}

Make sure the JSON is valid and complete. If a section has no relevant information, use empty arrays or appropriate null values.
"""
        return prompt
    
    def _convert_llm_response(self, llm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM response format to our internal format"""
        try:
            # Convert stock recommendations to our StockRecommendation format
            recommendations = []
            for rec in llm_data.get('stock_recommendations', []):
                stock_rec = StockRecommendation(
                    symbol=rec.get('symbol', 'UNKNOWN'),
                    company_name=rec.get('company_name', rec.get('symbol', 'UNKNOWN')),
                    action=rec.get('action', 'HOLD'),
                    target_price=rec.get('target_price'),
                    rationale=rec.get('rationale', ''),
                    confidence=rec.get('confidence', 0.5),
                    time_horizon=rec.get('timeframe', 'medium')
                )
                recommendations.append(stock_rec)
            
            return {
                'stock_recommendations': recommendations,
                'market_sentiment': llm_data.get('market_sentiment', 'NEUTRAL'),
                'key_themes': llm_data.get('key_themes', []),
                'sector_insights': llm_data.get('sector_insights', {}),
                'technical_signals': llm_data.get('technical_signals', {}),
                'chapter_summary': llm_data.get('chapter_summary', []),
                'confidence_score': llm_data.get('confidence_score', 0.5)
            }
        except Exception as e:
            self.logger.error(f"Failed to convert LLM response: {e}")
            return None
    
    async def _save_debug_files(self, transcript_text: str, prompt: str, channel: str, video_title: str, video_metadata: Dict[str, Any]) -> None:
        """Save debug files with complete transcript and prompt for analysis"""
        try:
            # Create debug directory
            debug_dir = os.path.join('./data/debug', self._get_current_date())
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save complete original transcript
            transcript_file = os.path.join(debug_dir, f"{channel}_original_transcript.txt")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"=== ORIGINAL TRANSCRIPT FOR {channel.upper()} ===\n")
                f.write(f"Video Title: {video_title}\n")
                f.write(f"Transcript Length: {len(transcript_text)} characters\n")
                f.write(f"Timestamp: {self._get_current_date()}\n")
                f.write("="*80 + "\n\n")
                f.write(transcript_text)
            
            # Save complete prompt
            prompt_file = os.path.join(debug_dir, f"{channel}_llm_prompt.txt")
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(f"=== COMPLETE LLM PROMPT FOR {channel.upper()} ===\n")
                f.write(f"Video Title: {video_title}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write(f"Timestamp: {self._get_current_date()}\n")
                f.write("="*80 + "\n\n")
                f.write(prompt)
            
            # Save metadata
            metadata_file = os.path.join(debug_dir, f"{channel}_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "channel": channel,
                    "video_title": video_title,
                    "original_transcript_length": len(transcript_text),
                    "prompt_length": len(prompt),
                    "metadata": video_metadata,
                    "timestamp": self._get_current_date()
                }, f, ensure_ascii=False, indent=2)
            
            print(f"🐛 DEBUG FILES SAVED:")
            print(f"   - Original Transcript: {transcript_file}")
            print(f"   - Analysis Prompt: {prompt_file}")
            print(f"   - Metadata: {metadata_file}")
            print(f"   - LLM Response: Will be saved after analysis")
            
        except Exception as e:
            self.logger.error(f"Failed to save debug files: {e}")
            print(f"❌ DEBUG FILE SAVE ERROR: {e}")

    async def _save_llm_response(self, llm_response: str, channel: str, video_title: str) -> None:
        """Save LLM response to debug files"""
        try:
            debug_dir = os.path.join('./data/debug', self._get_current_date())
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save complete LLM response
            response_file = os.path.join(debug_dir, f"{channel}_llm_response.json")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"=== LLM RAW RESPONSE FOR {channel.upper()} ===\n")
                f.write(f"Video Title: {video_title}\n")
                f.write(f"Response Length: {len(llm_response)} characters\n")
                f.write(f"Timestamp: {self._get_current_date()}\n")
                f.write("="*80 + "\n\n")
                f.write(llm_response)
            
            print(f"🤖 LLM RESPONSE SAVED: {response_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save LLM response: {e}")
            print(f"❌ LLM RESPONSE SAVE ERROR: {e}")

    async def _save_analysis(self, analysis: AnalysisResult) -> str:
        """Save analysis results to file with date-based organization"""
        # Use the analysis date for organization
        analysis_date = analysis.analysis_date  # Should be YYYY-MM-DD format
        filename = f"{analysis.channel}_analysis.json"
        
        output_path = self.config.settings.get('output_path', './data/analyses')
        # Create date-based directory structure
        date_dir = os.path.join(output_path, analysis_date)
        os.makedirs(date_dir, exist_ok=True)
        
        output_file = os.path.join(date_dir, filename)
        
        # Convert to dictionary for JSON serialization
        analysis_dict = {
            "channel": analysis.channel,
            "video_title": analysis.video_title,
            "analysis_date": analysis.analysis_date,
            "market_sentiment": analysis.market_sentiment,
            "key_themes": analysis.key_themes,
            "stock_recommendations": [
                {
                    "symbol": rec.symbol,
                    "company_name": rec.company_name,
                    "action": rec.action,
                    "target_price": rec.target_price,
                    "current_price": rec.current_price,
                    "rationale": rec.rationale,
                    "confidence": rec.confidence,
                    "time_horizon": rec.time_horizon,
                    "risk_level": rec.risk_level
                }
                for rec in analysis.stock_recommendations
            ],
            "sector_insights": analysis.sector_insights,
            "technical_signals": analysis.technical_signals,
            "chapter_summary": analysis.chapter_summary,
            "market_segments": [
                {
                    "timestamp": seg.timestamp,
                    "topic": seg.topic,
                    "content": seg.content,
                    "category": seg.category.value,
                    "importance": seg.importance,
                    "keywords": seg.keywords
                }
                for seg in analysis.market_segments
            ],
            "confidence_score": analysis.confidence_score
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, ensure_ascii=False, indent=2)
        
        return output_file
    
    async def cleanup(self) -> ToolResult:
        """Clean up resources"""
        return ToolResult(success=True, data={"message": "Cleanup completed"})
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters"""
        transcript_text = kwargs.get('transcript_text')
        transcript_file = kwargs.get('transcript_file')
        
        if not transcript_text and not transcript_file:
            return False
        
        if transcript_file and not os.path.exists(transcript_file):
            return False
        
        return True
