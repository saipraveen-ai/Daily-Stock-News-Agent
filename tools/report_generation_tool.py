"""
Report Generation Tool for Daily Stock News Agent

This tool generates structured daily reports from analyzed content,
combining insights from multiple channels into comprehensive summaries.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import markdown

from .base_tool import BaseTool, ToolResult, ToolConfig, ToolCategory, ToolPriority

logger = logging.getLogger(__name__)


@dataclass
class DailyReport:
    """Daily stock market report structure"""
    date: str
    channels_processed: List[str]
    market_overview: Dict[str, Any]
    stock_recommendations: Dict[str, List[Dict[str, Any]]]
    sector_analysis: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    key_news_events: List[str]
    action_items: List[str]
    channel_comparison: Dict[str, Any]
    confidence_score: float
    generation_timestamp: str


class ReportGenerationTool(BaseTool):
    """
    Tool for generating structured daily reports from analyzed content.
    
    Features:
    - Combine multiple channel analyses
    - Generate structured markdown reports
    - Create HTML versions for web viewing
    - Compare channel perspectives
    - Highlight actionable insights
    - Export to multiple formats (MD, HTML, JSON, PDF)
    """
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        
        self.output_path = config.settings.get('output_path', './data/reports')
        os.makedirs(self.output_path, exist_ok=True)
        
        self.template_path = config.settings.get('template_path', './templates')
        self.include_charts = config.settings.get('include_charts', False)
        
    async def initialize(self) -> ToolResult:
        """Initialize the report generation tool"""
        try:
            # Create output directories
            os.makedirs(os.path.join(self.output_path, 'daily'), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, 'html'), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, 'json'), exist_ok=True)
            
            self._is_initialized = True
            self.logger.info("Report generation tool initialized successfully")
            
            return ToolResult(
                success=True,
                data={"message": "Report generation tool ready"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize report tool: {e}")
            return ToolResult(
                success=False,
                error_message=f"Initialization failed: {str(e)}"
            )
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute report generation.
        
        Args:
            analysis_files: List[str] - Paths to analysis JSON files
            analysis_data: List[Dict] - Direct analysis data
            report_date: str - Date for the report (YYYY-MM-DD)
            output_formats: List[str] - Output formats ['markdown', 'html', 'json', 'pdf']
            include_comparison: bool - Whether to include channel comparison
            
        Returns:
            ToolResult with generated report files
        """
        analysis_files = kwargs.get('analysis_files', [])
        analysis_data = kwargs.get('analysis_data', [])
        report_date = kwargs.get('report_date', datetime.now().strftime('%Y-%m-%d'))
        output_formats = kwargs.get('output_formats', ['markdown', 'json'])
        include_comparison = kwargs.get('include_comparison', True)
        
        try:
            # Load analysis data
            if analysis_files:
                analysis_data = await self._load_analysis_files(analysis_files)
            
            if not analysis_data:
                return ToolResult(
                    success=False,
                    error_message="No analysis data provided"
                )
            
            # Generate comprehensive report
            report = await self._generate_daily_report(
                analysis_data, report_date, include_comparison
            )
            
            # Generate output files
            output_files = await self._generate_output_files(report, output_formats)
            
            return ToolResult(
                success=True,
                data={
                    "report": report,
                    "output_files": output_files,
                    "report_date": report_date
                },
                metadata={"channels_processed": len(analysis_data)}
            )
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return ToolResult(
                success=False,
                error_message=f"Report generation failed: {str(e)}"
            )
    
    async def _load_analysis_files(self, analysis_files: List[str]) -> List[Dict[str, Any]]:
        """Load analysis data from JSON files"""
        analysis_data = []
        
        for file_path in analysis_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        analysis_data.append(data)
                else:
                    self.logger.warning(f"Analysis file not found: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to load analysis file {file_path}: {e}")
        
        return analysis_data
    
    async def _generate_daily_report(
        self, 
        analysis_data: List[Dict[str, Any]], 
        report_date: str,
        include_comparison: bool
    ) -> DailyReport:
        """Generate comprehensive daily report"""
        
        # Extract channel information
        channels_processed = [data.get('channel', 'Unknown') for data in analysis_data]
        
        # Combine market overview
        market_overview = await self._combine_market_overview(analysis_data)
        
        # Combine stock recommendations
        stock_recommendations = await self._combine_stock_recommendations(analysis_data)
        
        # Combine sector analysis
        sector_analysis = await self._combine_sector_analysis(analysis_data)
        
        # Combine technical analysis
        technical_analysis = await self._combine_technical_analysis(analysis_data)
        
        # Extract key news events
        key_news_events = await self._extract_key_news_events(analysis_data)
        
        # Generate action items
        action_items = await self._generate_action_items(analysis_data)
        
        # Channel comparison
        channel_comparison = {}
        if include_comparison and len(analysis_data) > 1:
            channel_comparison = await self._compare_channels(analysis_data)
        
        # Calculate overall confidence
        confidence_score = await self._calculate_overall_confidence(analysis_data)
        
        return DailyReport(
            date=report_date,
            channels_processed=channels_processed,
            market_overview=market_overview,
            stock_recommendations=stock_recommendations,
            sector_analysis=sector_analysis,
            technical_analysis=technical_analysis,
            key_news_events=key_news_events,
            action_items=action_items,
            channel_comparison=channel_comparison,
            confidence_score=confidence_score,
            generation_timestamp=datetime.now().isoformat()
        )
    
    async def _combine_market_overview(self, analysis_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine market overview from all channels"""
        sentiments = []
        key_themes = []
        
        for data in analysis_data:
            sentiment = data.get('market_sentiment', 'NEUTRAL')
            sentiments.append(sentiment)
            
            themes = data.get('key_themes', [])
            key_themes.extend(themes)
        
        # Determine overall sentiment
        overall_sentiment = self._determine_overall_sentiment(sentiments)
        
        # Get unique themes
        unique_themes = list(set(key_themes))
        
        return {
            "overall_sentiment": overall_sentiment,
            "channel_sentiments": dict(zip([d.get('channel') for d in analysis_data], sentiments)),
            "key_themes": unique_themes[:10],  # Top 10 themes
            "consensus_level": self._calculate_consensus_level(sentiments)
        }
    
    def _determine_overall_sentiment(self, sentiments: List[str]) -> str:
        """Determine overall market sentiment from multiple channels"""
        sentiment_weights = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
        
        total_weight = sum(sentiment_weights.get(s, 0) for s in sentiments)
        avg_weight = total_weight / len(sentiments) if sentiments else 0
        
        if avg_weight > 0.3:
            return "BULLISH"
        elif avg_weight < -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_consensus_level(self, sentiments: List[str]) -> str:
        """Calculate consensus level among channels"""
        if len(set(sentiments)) == 1:
            return "HIGH"
        elif len(set(sentiments)) == 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _combine_stock_recommendations(self, analysis_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Combine stock recommendations from all channels"""
        combined_recs = {"BUY": [], "SELL": [], "HOLD": []}
        
        for data in analysis_data:
            channel = data.get('channel', 'Unknown')
            recommendations = data.get('stock_recommendations', [])
            
            for rec in recommendations:
                rec['source_channel'] = channel
                action = rec.get('action', 'HOLD')
                if action in combined_recs:
                    combined_recs[action].append(rec)
        
        # Sort by confidence and remove duplicates
        for action in combined_recs:
            combined_recs[action] = sorted(
                combined_recs[action], 
                key=lambda x: x.get('confidence', 0), 
                reverse=True
            )
        
        return combined_recs
    
    async def _combine_sector_analysis(self, analysis_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine sector analysis from all channels"""
        sector_insights = {}
        
        for data in analysis_data:
            channel = data.get('channel', 'Unknown')
            sectors = data.get('sector_insights', {})
            
            for sector, insight in sectors.items():
                if sector not in sector_insights:
                    sector_insights[sector] = {
                        "channels": [],
                        "total_mentions": 0,
                        "sentiments": [],
                        "keywords": []
                    }
                
                sector_insights[sector]["channels"].append(channel)
                sector_insights[sector]["total_mentions"] += insight.get('mentions', 0)
                sector_insights[sector]["sentiments"].append(insight.get('sentiment', 'NEUTRAL'))
                sector_insights[sector]["keywords"].extend(insight.get('keywords_found', []))
        
        # Calculate sector rankings
        sector_rankings = sorted(
            sector_insights.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )
        
        return {
            "sector_insights": sector_insights,
            "top_sectors": [sector for sector, _ in sector_rankings[:5]],
            "sector_rankings": sector_rankings
        }
    
    async def _combine_technical_analysis(self, analysis_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine technical analysis from all channels"""
        technical_signals = {}
        
        for data in analysis_data:
            channel = data.get('channel', 'Unknown')
            signals = data.get('technical_signals', {})
            
            for signal_type, signal_data in signals.items():
                if signal_type not in technical_signals:
                    technical_signals[signal_type] = {
                        "channels": [],
                        "contexts": [],
                        "strength": 0
                    }
                
                technical_signals[signal_type]["channels"].append(channel)
                technical_signals[signal_type]["contexts"].extend(
                    signal_data.get('context', [])
                )
                technical_signals[signal_type]["strength"] += 1
        
        return technical_signals
    
    async def _extract_key_news_events(self, analysis_data: List[Dict[str, Any]]) -> List[str]:
        """Extract key news events from all channels"""
        news_events = []
        
        for data in analysis_data:
            segments = data.get('market_segments', [])
            
            # Find news-related segments
            news_segments = [
                seg for seg in segments 
                if seg.get('category') == 'news_events' and seg.get('importance', 0) > 0.6
            ]
            
            for segment in news_segments[:3]:  # Top 3 per channel
                news_events.append(segment.get('content', ''))
        
        return news_events
    
    async def _generate_action_items(self, analysis_data: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable items from analysis"""
        action_items = []
        
        # From stock recommendations
        all_recommendations = []
        for data in analysis_data:
            all_recommendations.extend(data.get('stock_recommendations', []))
        
        # High confidence recommendations
        high_conf_recs = [
            rec for rec in all_recommendations 
            if rec.get('confidence', 0) > 0.7
        ]
        
        for rec in high_conf_recs[:5]:  # Top 5
            action_items.append(
                f"{rec['action']} {rec['symbol']}: {rec.get('rationale', '')[:100]}..."
            )
        
        # From technical analysis
        for data in analysis_data:
            technical = data.get('technical_signals', {})
            for signal_type, signal_data in technical.items():
                if signal_type in ['breakout', 'breakdown'] and signal_data.get('strength', 0) > 1:
                    action_items.append(f"Monitor {signal_type} signals in market")
        
        # From sector analysis
        sector_analysis = await self._combine_sector_analysis(analysis_data)
        top_sectors = sector_analysis.get('top_sectors', [])[:3]
        if top_sectors:
            action_items.append(f"Focus on {', '.join(top_sectors)} sectors")
        
        return action_items[:10]  # Limit to 10 items
    
    async def _compare_channels(self, analysis_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare perspectives between channels"""
        if len(analysis_data) < 2:
            return {}
        
        comparison = {
            "sentiment_agreement": {},
            "recommendation_overlap": {},
            "unique_insights": {}
        }
        
        # Sentiment comparison
        sentiments = {data.get('channel'): data.get('market_sentiment') for data in analysis_data}
        comparison["sentiment_agreement"] = sentiments
        
        # Recommendation overlap
        for i, data1 in enumerate(analysis_data):
            for j, data2 in enumerate(analysis_data[i+1:], i+1):
                channel1 = data1.get('channel')
                channel2 = data2.get('channel')
                
                recs1 = {rec['symbol'] for rec in data1.get('stock_recommendations', [])}
                recs2 = {rec['symbol'] for rec in data2.get('stock_recommendations', [])}
                
                overlap = len(recs1.intersection(recs2))
                total = len(recs1.union(recs2))
                
                comparison["recommendation_overlap"][f"{channel1}_vs_{channel2}"] = {
                    "overlap_count": overlap,
                    "total_unique": total,
                    "overlap_percentage": overlap / total * 100 if total > 0 else 0
                }
        
        return comparison
    
    async def _calculate_overall_confidence(self, analysis_data: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the combined analysis"""
        confidence_scores = [data.get('confidence_score', 0.5) for data in analysis_data]
        
        if not confidence_scores:
            return 0.5
        
        # Average confidence weighted by number of recommendations
        weighted_scores = []
        for data in analysis_data:
            score = data.get('confidence_score', 0.5)
            rec_count = len(data.get('stock_recommendations', []))
            weight = max(1, rec_count)  # At least weight of 1
            weighted_scores.extend([score] * weight)
        
        return sum(weighted_scores) / len(weighted_scores)
    
    async def _generate_output_files(self, report: DailyReport, output_formats: List[str]) -> Dict[str, str]:
        """Generate output files in specified formats"""
        output_files = {}
        
        base_filename = f"daily_stock_report_{report.date.replace('-', '')}"
        
        if 'json' in output_formats:
            json_file = await self._generate_json_report(report, base_filename)
            output_files['json'] = json_file
        
        if 'markdown' in output_formats:
            md_file = await self._generate_markdown_report(report, base_filename)
            output_files['markdown'] = md_file
        
        if 'html' in output_formats:
            html_file = await self._generate_html_report(report, base_filename)
            output_files['html'] = html_file
        
        return output_files
    
    async def _generate_json_report(self, report: DailyReport, base_filename: str) -> str:
        """Generate JSON format report"""
        json_path = os.path.join(self.output_path, 'json', f"{base_filename}.json")
        
        # Convert dataclass to dict
        report_dict = {
            "date": report.date,
            "channels_processed": report.channels_processed,
            "market_overview": report.market_overview,
            "stock_recommendations": report.stock_recommendations,
            "sector_analysis": report.sector_analysis,
            "technical_analysis": report.technical_analysis,
            "key_news_events": report.key_news_events,
            "action_items": report.action_items,
            "channel_comparison": report.channel_comparison,
            "confidence_score": report.confidence_score,
            "generation_timestamp": report.generation_timestamp
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        return json_path
    
    async def _generate_markdown_report(self, report: DailyReport, base_filename: str) -> str:
        """Generate Markdown format report"""
        md_path = os.path.join(self.output_path, 'daily', f"{base_filename}.md")
        
        markdown_content = self._create_markdown_content(report)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return md_path
    
    def _create_markdown_content(self, report: DailyReport) -> str:
        """Create markdown content for the report"""
        content = f"""# Daily Stock Market Analysis - {report.date}

*Generated on {report.generation_timestamp.split('T')[0]} at {report.generation_timestamp.split('T')[1][:8]}*

## 📈 Market Overview

- **Overall Sentiment**: {report.market_overview.get('overall_sentiment', 'NEUTRAL')}
- **Consensus Level**: {report.market_overview.get('consensus_level', 'MEDIUM')}
- **Channels Processed**: {', '.join(report.channels_processed)}
- **Confidence Score**: {report.confidence_score:.2f}/1.00

### Channel Sentiments
"""
        
        channel_sentiments = report.market_overview.get('channel_sentiments', {})
        for channel, sentiment in channel_sentiments.items():
            content += f"- **{channel}**: {sentiment}\n"
        
        content += f"""
### Key Market Themes
"""
        key_themes = report.market_overview.get('key_themes', [])
        for theme in key_themes[:10]:
            content += f"- {theme}\n"
        
        content += f"""
## 🎯 Stock Recommendations

### 🟢 BUY Recommendations
"""
        buy_recs = report.stock_recommendations.get('BUY', [])
        for rec in buy_recs[:5]:
            content += f"""
**{rec['symbol']} - {rec['company_name']}**
- **Source**: {rec.get('source_channel', 'Unknown')}
- **Confidence**: {rec.get('confidence', 0):.2f}
- **Rationale**: {rec.get('rationale', 'No rationale provided')[:200]}...
- **Target Price**: {rec.get('target_price', 'Not specified')}
"""

        content += f"""
### 🔴 SELL Recommendations
"""
        sell_recs = report.stock_recommendations.get('SELL', [])
        for rec in sell_recs[:5]:
            content += f"""
**{rec['symbol']} - {rec['company_name']}**
- **Source**: {rec.get('source_channel', 'Unknown')}
- **Confidence**: {rec.get('confidence', 0):.2f}
- **Rationale**: {rec.get('rationale', 'No rationale provided')[:200]}...
"""

        content += f"""
### 🟡 HOLD Recommendations
"""
        hold_recs = report.stock_recommendations.get('HOLD', [])
        for rec in hold_recs[:3]:
            content += f"""
**{rec['symbol']} - {rec['company_name']}**
- **Source**: {rec.get('source_channel', 'Unknown')}
- **Rationale**: {rec.get('rationale', 'No rationale provided')[:150]}...
"""

        content += f"""
## 🏭 Sector Analysis

### Top Performing Sectors
"""
        top_sectors = report.sector_analysis.get('top_sectors', [])
        for sector in top_sectors:
            content += f"- **{sector}**\n"
        
        content += f"""
## 📊 Technical Analysis

### Key Signals Detected
"""
        for signal_type, signal_data in report.technical_analysis.items():
            channels = ', '.join(signal_data.get('channels', []))
            content += f"- **{signal_type.replace('_', ' ').title()}**: Mentioned by {channels}\n"
        
        content += f"""
## 📰 Key News & Events
"""
        for i, event in enumerate(report.key_news_events[:5], 1):
            content += f"{i}. {event[:200]}...\n\n"
        
        content += f"""
## 🎯 Action Items

### Today's Priority Actions
"""
        for i, action in enumerate(report.action_items, 1):
            content += f"{i}. {action}\n"
        
        if report.channel_comparison:
            content += f"""
## 🔄 Channel Comparison

### Sentiment Agreement
"""
            sentiment_agreement = report.channel_comparison.get('sentiment_agreement', {})
            for channel, sentiment in sentiment_agreement.items():
                content += f"- **{channel}**: {sentiment}\n"
            
            content += f"""
### Recommendation Overlap
"""
            rec_overlap = report.channel_comparison.get('recommendation_overlap', {})
            for comparison, data in rec_overlap.items():
                overlap_pct = data.get('overlap_percentage', 0)
                content += f"- **{comparison}**: {overlap_pct:.1f}% overlap\n"
        
        content += f"""
---
*This report was automatically generated by the Daily Stock News Agent. Always conduct your own research before making investment decisions.*
"""
        
        return content
    
    async def _generate_html_report(self, report: DailyReport, base_filename: str) -> str:
        """Generate HTML format report"""
        html_path = os.path.join(self.output_path, 'html', f"{base_filename}.html")
        
        # First generate markdown
        markdown_content = self._create_markdown_content(report)
        
        # Convert to HTML
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'toc'])
        
        # Wrap in HTML template
        full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Stock Report - {report.date}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .confidence {{ background: #e8f5e8; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .buy {{ color: #27ae60; }}
        .sell {{ color: #e74c3c; }}
        .hold {{ color: #f39c12; }}
        .bullish {{ color: #27ae60; font-weight: bold; }}
        .bearish {{ color: #e74c3c; font-weight: bold; }}
        .neutral {{ color: #7f8c8d; font-weight: bold; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        return html_path
    
    async def cleanup(self) -> ToolResult:
        """Clean up resources"""
        return ToolResult(success=True, data={"message": "Cleanup completed"})
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters"""
        analysis_files = kwargs.get('analysis_files', [])
        analysis_data = kwargs.get('analysis_data', [])
        
        if not analysis_files and not analysis_data:
            return False
        
        # Validate analysis files exist
        for file_path in analysis_files:
            if not os.path.exists(file_path):
                return False
        
        return True
