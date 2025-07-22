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
    confidence_score: float = 0.0


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
        
        # Stock market keywords and patterns
        self.stock_patterns = {
            # Telugu stock-related terms (in English transliteration)
            'buy_signals': [
                'konali', 'buy', 'purchase', 'invest', 'accumulate',
                'good entry', 'strong buy', 'recommended', 'positive'
            ],
            'sell_signals': [
                'ammali', 'sell', 'book profit', 'exit', 'avoid',
                'negative', 'dump', 'risky', 'dangerous'
            ],
            'hold_signals': [
                'hold', 'wait', 'watch', 'observe', 'monitor',
                'long term', 'patience', 'steady'
            ],
            'sentiment_positive': [
                'bullish', 'positive', 'good', 'strong', 'upward',
                'growth', 'profit', 'gain', 'success'
            ],
            'sentiment_negative': [
                'bearish', 'negative', 'bad', 'weak', 'downward',
                'loss', 'decline', 'fall', 'crash'
            ]
        }
        
        # Sector keywords
        self.sector_keywords = {
            'IT': ['TCS', 'Infosys', 'Wipro', 'HCL', 'Tech Mahindra', 'software', 'technology'],
            'Banking': ['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'banking', 'finance'],
            'Pharma': ['Sun Pharma', 'Dr Reddy', 'Cipla', 'pharmaceutical', 'healthcare'],
            'Auto': ['Maruti', 'Hyundai', 'Tata Motors', 'Bajaj', 'automotive', 'automobile'],
            'FMCG': ['HUL', 'ITC', 'Nestle', 'consumer goods', 'FMCG'],
            'Energy': ['Reliance', 'ONGC', 'NTPC', 'Power Grid', 'energy', 'oil', 'gas'],
            'Metals': ['Tata Steel', 'JSW', 'Hindalco', 'metals', 'steel', 'aluminum']
        }
        
        # Technical analysis keywords
        self.technical_keywords = {
            'support': ['support', 'floor', 'bottom', 'base'],
            'resistance': ['resistance', 'ceiling', 'top', 'barrier'],
            'breakout': ['breakout', 'break above', 'break through'],
            'breakdown': ['breakdown', 'break below', 'break down'],
            'trend': ['uptrend', 'downtrend', 'sideways', 'consolidation'],
            'patterns': ['head and shoulders', 'double top', 'double bottom', 'triangle']
        }
        
        self.ai_provider = config.settings.get('ai_provider', 'local')
        self.ai_api_key = config.settings.get('ai_api_key')
        
    async def initialize(self) -> ToolResult:
        """Initialize the content analysis tool"""
        try:
            # Test AI provider if configured
            if self.ai_provider != 'local' and self.ai_api_key:
                # Initialize AI provider (OpenAI, Gemini, etc.)
                self.logger.info(f"AI provider {self.ai_provider} configured")
            else:
                self.logger.info("Using local pattern-based analysis")
            
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
            
        Returns:
            ToolResult with analysis results
        """
        transcript_text = kwargs.get('transcript_text')
        transcript_file = kwargs.get('transcript_file')
        
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
                transcript_text, channel, video_title, analysis_types
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
        analysis_types: List[str]
    ) -> AnalysisResult:
        """Perform comprehensive content analysis"""
        
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
        
        if 'all' in analysis_types or 'stock_recommendations' in analysis_types:
            stock_recommendations = await self._extract_stock_recommendations(cleaned_text)
        
        if 'all' in analysis_types or 'market_sentiment' in analysis_types:
            market_sentiment = await self._analyze_market_sentiment(cleaned_text)
        
        if 'all' in analysis_types or 'sector_analysis' in analysis_types:
            sector_insights = await self._analyze_sectors(cleaned_text)
        
        if 'all' in analysis_types or 'technical_analysis' in analysis_types:
            technical_signals = await self._extract_technical_signals(cleaned_text)
        
        if 'all' in analysis_types or 'key_segments' in analysis_types:
            key_themes = await self._extract_key_themes(segments)
        
        # Calculate overall confidence score
        confidence_score = await self._calculate_confidence(
            stock_recommendations, segments, text
        )
        
        return AnalysisResult(
            channel=channel,
            video_title=video_title,
            analysis_date=self._get_current_date(),
            market_sentiment=market_sentiment,
            key_themes=key_themes,
            stock_recommendations=stock_recommendations,
            sector_insights=sector_insights,
            technical_signals=technical_signals,
            market_segments=segments,
            confidence_score=confidence_score
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
        """Classify a text segment by type"""
        text_lower = text.lower()
        
        # Check for stock recommendations
        if any(signal in text_lower for signal in 
               self.stock_patterns['buy_signals'] + 
               self.stock_patterns['sell_signals'] + 
               self.stock_patterns['hold_signals']):
            return AnalysisType.STOCK_RECOMMENDATIONS
        
        # Check for technical analysis
        if any(keyword in text_lower for keywords in self.technical_keywords.values() 
               for keyword in keywords):
            return AnalysisType.TECHNICAL_ANALYSIS
        
        # Check for sector analysis
        if any(keyword in text_lower for keywords in self.sector_keywords.values()
               for keyword in keywords):
            return AnalysisType.SECTOR_ANALYSIS
        
        # Check for sentiment
        if any(keyword in text_lower for keyword in 
               self.stock_patterns['sentiment_positive'] + 
               self.stock_patterns['sentiment_negative']):
            return AnalysisType.MARKET_SENTIMENT
        
        return AnalysisType.NEWS_EVENTS
    
    def _calculate_segment_importance(self, text: str) -> float:
        """Calculate importance score for a segment"""
        importance = 0.0
        text_lower = text.lower()
        
        # Stock names and symbols increase importance
        if any(sector in text_lower for sector_stocks in self.sector_keywords.values() 
               for sector in sector_stocks):
            importance += 0.3
        
        # Action words increase importance
        if any(signal in text_lower for signals in self.stock_patterns.values()
               for signal in signals):
            importance += 0.2
        
        # Numbers (prices, percentages) increase importance
        if re.search(r'\d+', text):
            importance += 0.1
        
        # Technical terms increase importance
        if any(keyword in text_lower for keywords in self.technical_keywords.values()
               for keyword in keywords):
            importance += 0.2
        
        # Longer segments get slight boost
        if len(text) > 100:
            importance += 0.1
        
        return min(importance, 1.0)  # Cap at 1.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        keywords = []
        text_lower = text.lower()
        
        # Extract stock symbols/names
        for sector_stocks in self.sector_keywords.values():
            for stock in sector_stocks:
                if stock.lower() in text_lower:
                    keywords.append(stock)
        
        # Extract action words
        for signal_type, signals in self.stock_patterns.items():
            for signal in signals:
                if signal in text_lower:
                    keywords.append(signal)
        
        return list(set(keywords))  # Remove duplicates
    
    def _generate_topic_title(self, text: str) -> str:
        """Generate a topic title for a segment"""
        # Simple approach: take first 50 characters
        title = text[:50].strip()
        if len(text) > 50:
            title += "..."
        return title.capitalize()
    
    async def _extract_stock_recommendations(self, text: str) -> List[StockRecommendation]:
        """Extract stock recommendations from text"""
        recommendations = []
        
        # This is a simplified pattern-based approach
        # In production, you'd want to use NLP/AI for better accuracy
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Look for stock names and action words together
            for sector, stocks in self.sector_keywords.items():
                for stock in stocks:
                    if stock.lower() in sentence_lower:
                        action = self._determine_action(sentence_lower)
                        
                        if action:
                            recommendation = StockRecommendation(
                                symbol=stock.upper(),
                                company_name=stock,
                                action=action,
                                rationale=sentence.strip(),
                                confidence=0.6  # Pattern-based confidence
                            )
                            recommendations.append(recommendation)
        
        return recommendations
    
    def _determine_action(self, text: str) -> Optional[str]:
        """Determine buy/sell/hold action from text"""
        buy_score = sum(1 for signal in self.stock_patterns['buy_signals'] 
                       if signal in text)
        sell_score = sum(1 for signal in self.stock_patterns['sell_signals'] 
                        if signal in text)
        hold_score = sum(1 for signal in self.stock_patterns['hold_signals'] 
                        if signal in text)
        
        if buy_score > sell_score and buy_score > hold_score:
            return "BUY"
        elif sell_score > buy_score and sell_score > hold_score:
            return "SELL"
        elif hold_score > 0:
            return "HOLD"
        
        return None
    
    async def _analyze_market_sentiment(self, text: str) -> str:
        """Analyze overall market sentiment"""
        positive_score = sum(1 for keyword in self.stock_patterns['sentiment_positive']
                           if keyword in text)
        negative_score = sum(1 for keyword in self.stock_patterns['sentiment_negative']
                           if keyword in text)
        
        if positive_score > negative_score * 1.2:  # 20% threshold
            return "BULLISH"
        elif negative_score > positive_score * 1.2:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    async def _analyze_sectors(self, text: str) -> Dict[str, Any]:
        """Analyze sector-wise performance mentions"""
        sector_mentions = {}
        
        for sector, keywords in self.sector_keywords.items():
            mentions = sum(1 for keyword in keywords if keyword.lower() in text)
            if mentions > 0:
                sentiment = self._get_sector_sentiment(text, keywords)
                sector_mentions[sector] = {
                    "mentions": mentions,
                    "sentiment": sentiment,
                    "keywords_found": [kw for kw in keywords if kw.lower() in text]
                }
        
        return sector_mentions
    
    def _get_sector_sentiment(self, text: str, sector_keywords: List[str]) -> str:
        """Get sentiment for a specific sector"""
        # Look for sentiment words near sector keywords
        # This is a simplified approach
        return "NEUTRAL"  # Placeholder
    
    async def _extract_technical_signals(self, text: str) -> Dict[str, Any]:
        """Extract technical analysis signals"""
        signals = {}
        
        for signal_type, keywords in self.technical_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text]
            if found_keywords:
                signals[signal_type] = {
                    "found": True,
                    "keywords": found_keywords,
                    "context": self._get_context_for_keywords(text, found_keywords)
                }
        
        return signals
    
    def _get_context_for_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Get context sentences for found keywords"""
        contexts = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                contexts.append(sentence.strip())
        
        return contexts[:3]  # Return up to 3 context sentences
    
    async def _extract_key_themes(self, segments: List[MarketSegment]) -> List[str]:
        """Extract key themes from segments"""
        themes = []
        
        # Group segments by category
        category_counts = {}
        for segment in segments:
            category = segment.category.value
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        # Convert to themes
        for category, count in category_counts.items():
            if count >= 2:  # At least 2 mentions
                themes.append(f"{category.replace('_', ' ').title()} ({count} mentions)")
        
        # Add most important keywords
        all_keywords = []
        for segment in segments[:10]:  # Top 10 segments
            all_keywords.extend(segment.keywords)
        
        # Count keyword frequencies
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Add top keywords as themes
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for keyword, count in top_keywords:
            if count >= 2:
                themes.append(f"{keyword} (mentioned {count} times)")
        
        return themes
    
    async def _calculate_confidence(
        self, 
        recommendations: List[StockRecommendation],
        segments: List[MarketSegment],
        text: str
    ) -> float:
        """Calculate overall confidence score for the analysis"""
        confidence = 0.0
        
        # Base confidence from text length
        if len(text) > 1000:
            confidence += 0.2
        elif len(text) > 500:
            confidence += 0.1
        
        # Confidence from number of recommendations
        if len(recommendations) > 0:
            confidence += min(len(recommendations) * 0.1, 0.3)
        
        # Confidence from segment quality
        high_importance_segments = [s for s in segments if s.importance > 0.7]
        confidence += min(len(high_importance_segments) * 0.05, 0.2)
        
        # Confidence from keyword density
        total_keywords = sum(len(s.keywords) for s in segments)
        if total_keywords > 20:
            confidence += 0.2
        elif total_keywords > 10:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d')
    
    async def _save_analysis(self, analysis: AnalysisResult) -> str:
        """Save analysis results to file"""
        timestamp = self._get_current_date().replace('-', '')
        filename = f"{analysis.channel}_{timestamp}_analysis.json"
        
        output_path = self.config.settings.get('output_path', './data/analysis')
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, filename)
        
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
