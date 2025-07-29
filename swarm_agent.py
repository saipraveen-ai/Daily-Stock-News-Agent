#!/usr/bin/env python3
"""
Complete Swarm Agent for Daily Stock News Analysis
- Downloads videos with date-based organization
- Transcribes using Whisper with metadata integration  
- Analyzes content with channel authority assessment
- Full workflow automation with enhanced metadata usage
"""

import os
import json
import yt_dlp
import whisper
from datetime import datetime
from typing import Dict, Any, List
import openai
from swarm import Swarm, Agent

class SwarmStockNewsAgent:
    def __init__(self):
        """Initialize with enhanced context tracking"""
        self.context = {
            "videos": [],
            "transcriptions": [],
            "analyses": [],
            "errors": [],
            "metadata": {}
        }
        self.client = Swarm()
        
        # Load Whisper model for transcription
        print("🔄 Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded")
        
        # Initialize specialized agents
        self.coordinator = Agent(
            name="Coordinator",
            instructions="You coordinate the daily stock news analysis workflow. You manage downloading videos, transcription, and analysis with metadata integration.",
        )
        
        self.downloader = Agent(
            name="Downloader", 
            instructions="You download YouTube videos from Telugu financial channels with the best quality for transcription. Use date-based organization.",
            functions=[self.download_videos]
        )
        
        self.transcriber = Agent(
            name="Transcriber",
            instructions="You transcribe video content using Whisper and integrate metadata for enhanced analysis.",
            functions=[self.transcribe_videos]
        )
        
        self.analyzer = Agent(
            name="Analyzer", 
            instructions="You analyze transcribed content for stock insights using metadata context for credibility assessment.",
            functions=[self.analyze_content]
        )

    def download_videos(self, channels: str = "moneypurse,daytradertelugu", target_date: str = None) -> str:
        """Download videos with consistent date-based organization matching download_best_quality.py"""
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📥 Starting download for {target_date}")
        
        # Channel configuration - exact same as working download_best_quality.py
        channel_urls = {
            'moneypurse': 'https://www.youtube.com/@MoneyPurse/videos',
            'daytradertelugu': 'https://www.youtube.com/@daytradertelugu/videos',
        }
        
        # Base yt-dlp options - exact same as working script
        ydl_opts_base = {
            'format': (
                'best[height>=1080][acodec!=none][abr>=128]/best[height>=720][acodec!=none][abr>=96]/'
                'bestvideo[height>=1080]+bestaudio[abr>=128]/bestvideo[height>=720]+bestaudio[abr>=96]/'
                'best[ext=mp4]/best'
            ),
            'writeinfojson': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': True,
            'no_warnings': False,
            'playlistend': 1,
            'playlistreverse': False,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }, {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
                'nopostoverwrites': True,
            }],
            'prefer_free_formats': False,
            'youtube_include_dash_manifest': True,
        }
        
        downloaded_videos = []
        channel_list = [ch.strip() for ch in channels.split(',')]
        
        for channel_name in channel_list:
            if channel_name not in channel_urls:
                print(f"⚠️ Unknown channel: {channel_name}")
                continue
                
            channel_url = channel_urls[channel_name]
            
            try:
                # Create date-based directory structure
                video_date_dir = f'./data/videos/{target_date}'
                os.makedirs(video_date_dir, exist_ok=True)
                
                print(f"🎯 Processing {channel_name} -> {video_date_dir}")
                
                # Channel-specific yt-dlp options with output template
                channel_ydl_opts = ydl_opts_base.copy()
                channel_ydl_opts['outtmpl'] = os.path.join(video_date_dir, f'{channel_name}.%(ext)s')
                
                with yt_dlp.YoutubeDL(channel_ydl_opts) as ydl:
                    print(f"🔍 Extracting info for {channel_name}...")
                    
                    try:
                        # Extract video info first
                        video_info = ydl.extract_info(channel_url, download=False)
                        print(f"📺 Found {len(video_info.get('entries', []))} videos")
                        
                        # Download the video
                        print(f"⬇️ Downloading {channel_name}...")
                        ydl.download([channel_url])
                        
                        # Generate metadata - exact same as download_best_quality.py
                        try:
                            if video_info and 'entries' in video_info and video_info['entries']:
                                latest_video = video_info['entries'][0]
                                metadata = self._create_metadata(channel_name, channel_url, latest_video, target_date)
                            else:
                                metadata = self._create_fallback_metadata(channel_name, channel_url, target_date)
                        except Exception as e:
                            print(f"⚠️ Could not extract detailed metadata: {e}")
                            metadata = self._create_fallback_metadata(channel_name, channel_url, target_date)
                        
                        # Save metadata
                        metadata_file = os.path.join(video_date_dir, f'{channel_name}.json')
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)
                        
                        print(f"📋 Metadata saved: {channel_name}.json")
                        
                        # Check for downloaded files
                        wav_file = os.path.join(video_date_dir, f'{channel_name}.wav')
                        mp4_file = os.path.join(video_date_dir, f'{channel_name}.mp4')
                        info_file = os.path.join(video_date_dir, f'{channel_name}.info.json')
                        
                        if os.path.exists(wav_file):
                            print(f"✅ Audio downloaded: {channel_name}.wav")
                            
                            video_data = {
                                'channel': channel_name,
                                'date': target_date,
                                'wav_file': wav_file,
                                'mp4_file': mp4_file if os.path.exists(mp4_file) else None,
                                'info_file': info_file if os.path.exists(info_file) else None,
                                'metadata_path': metadata_file,
                                'title': metadata['video_info']['title'],
                                'duration': metadata['video_info']['duration'],
                                'upload_date': metadata['video_info']['upload_date'],
                                'view_count': metadata['engagement_metrics']['view_count'],
                                'engagement_ratio': metadata['engagement_metrics']['engagement_ratio']
                            }
                            
                            downloaded_videos.append(video_data)
                            print(f"🎉 Successfully processed {channel_name}")
                        else:
                            print(f"❌ No audio file found for {channel_name}")
                    
                    except Exception as e:
                        print(f"❌ Download failed for {channel_name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Channel processing failed for {channel_name}: {e}")
                continue
        
        self.context["videos"] = downloaded_videos
        return f"✅ Downloaded {len(downloaded_videos)} videos for {target_date}"

    def _create_metadata(self, channel_name: str, channel_url: str, video_info: dict, target_date: str) -> dict:
        """Create comprehensive metadata - exact same as download_best_quality.py"""
        return {
            'channel_name': channel_name,
            'channel_url': channel_url,
            'video_info': {
                'title': video_info.get('title', 'Unknown Title'),
                'duration': video_info.get('duration', 0),
                'duration_formatted': f"{video_info.get('duration', 0)//60}:{video_info.get('duration', 0)%60:02d}",
                'description': video_info.get('description', ''),
                'upload_date': video_info.get('upload_date', ''),
                'view_count': video_info.get('view_count', 0),
                'like_count': video_info.get('like_count', 0),
                'comment_count': video_info.get('comment_count', 0),
            },
            'channel_info': {
                'follower_count': video_info.get('channel_follower_count', 0),
                'is_verified': video_info.get('channel_is_verified', False),
            },
            'engagement_metrics': {
                'view_count': video_info.get('view_count', 0),
                'like_count': video_info.get('like_count', 0),
                'comment_count': video_info.get('comment_count', 0),
                'engagement_ratio': (video_info.get('like_count', 0) / max(video_info.get('view_count', 1), 1)) * 100
            },
            'download_info': {
                'download_date': target_date,
                'quality_downloaded': 'best_available',
                'file_name': f'{channel_name}.mp4'
            }
        }

    def _create_fallback_metadata(self, channel_name: str, channel_url: str, target_date: str) -> dict:
        """Create fallback metadata when extraction fails"""
        return {
            'channel_name': channel_name,
            'channel_url': channel_url,
            'video_info': {
                'title': f'{channel_name} video {target_date}',
                'duration': 0,
                'duration_formatted': '0:00',
                'description': 'Metadata extraction failed',
                'upload_date': target_date.replace('-', ''),
                'view_count': 0,
                'like_count': 0,
                'comment_count': 0,
            },
            'channel_info': {
                'follower_count': 0,
                'is_verified': False,
            },
            'engagement_metrics': {
                'view_count': 0,
                'like_count': 0,
                'comment_count': 0,
                'engagement_ratio': 0.0
            },
            'download_info': {
                'download_date': target_date,
                'quality_downloaded': 'fallback',
                'file_name': f'{channel_name}.mp4'
            }
        }

    def transcribe_videos(self, target_date: str = None) -> str:
        """Transcribe videos with enhanced metadata integration"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            videos = self.context.get("videos", [])
            if not videos:
                return "❌ No videos to transcribe"
            
            print(f"🎙️ Transcribing {len(videos)} videos with metadata integration...")
            
            transcriptions = []
            
            for video in videos:
                try:
                    wav_file = video['wav_file']
                    channel = video['channel']
                    
                    if not os.path.exists(wav_file):
                        print(f"❌ Audio file not found: {wav_file}")
                        continue
                    
                    print(f"🔄 Transcribing {channel}...")
                    
                    # Load metadata for this video
                    metadata_file = video.get('metadata_path', '')
                    metadata = {}
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        print(f"📋 Loaded metadata for {channel}")
                    
                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(wav_file)
                    
                    # Save transcript
                    transcript_file = os.path.join(f'./data/videos/{target_date}', f'{channel}_transcript.txt')
                    
                    with open(transcript_file, 'w', encoding='utf-8') as f:
                        f.write(result["text"])
                    
                    # Enhanced transcription data with metadata
                    transcription_data = {
                        'video_info': video,
                        'transcription_result': result,
                        'transcript_file': transcript_file,
                        'transcript_text': result["text"],
                        'channel': channel,
                        'date': target_date,
                        'metadata': metadata,
                        'audio_quality': self._assess_audio_quality(result),
                        'transcript_length': len(result["text"]),
                        'confidence_segments': len([seg for seg in result.get("segments", []) if seg.get("no_speech_prob", 1) < 0.5])
                    }
                    
                    transcriptions.append(transcription_data)
                    print(f"✅ Transcribed {channel}: {len(result['text'])} characters")
                    
                except Exception as e:
                    print(f"❌ Error transcribing {video.get('channel', 'unknown')}: {e}")
                    continue
            
            self.context["transcriptions"] = transcriptions
            
            return f"✅ Transcribed {len(transcriptions)} videos with metadata integration"
            
        except Exception as e:
            error = f"❌ Transcription error: {str(e)}"
            self.context["errors"].append(error)
            return error

    def _assess_audio_quality(self, whisper_result: dict) -> str:
        """Assess audio quality from Whisper result"""
        segments = whisper_result.get("segments", [])
        if not segments:
            return "unknown"
        
        avg_confidence = sum(1 - seg.get("no_speech_prob", 1) for seg in segments) / len(segments)
        
        if avg_confidence > 0.8:
            return "high"
        elif avg_confidence > 0.6:
            return "medium"
        else:
            return "low"

    def analyze_content(self, target_date: str = None) -> str:
        """Enhanced analysis with comprehensive metadata integration"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            transcriptions = self.context.get("transcriptions", [])
            if not transcriptions:
                return "❌ No transcriptions to analyze"
            
            print(f"📊 Analyzing {len(transcriptions)} transcriptions with enhanced metadata...")
            
            analyses = []
            
            for t in transcriptions:
                try:
                    video_info = t['video_info']
                    channel = video_info['channel']
                    metadata = t.get('metadata', {})
                    transcript_text = t.get('transcript_text', '')
                    
                    if not transcript_text or len(transcript_text.strip()) < 50:
                        print(f"⚠️ Transcript too short for {channel}")
                        continue
                    
                    print(f"🔍 Analyzing {channel} with metadata context...")
                    
                    # Enhanced analysis with comprehensive metadata
                    analysis = self._analyze_with_enhanced_metadata(
                        video_info, transcript_text, metadata, t, target_date
                    )
                    
                    analyses.append(analysis)
                    print(f"✅ Analysis complete for {channel}")
                
                except Exception as e:
                    print(f"❌ Error analyzing {t.get('channel', 'unknown')}: {e}")
                    continue
            
            self.context["analyses"] = analyses
            
            if analyses:
                # Save comprehensive analysis
                analysis_file = f'./data/analysis/swarm_analysis_{target_date}.json'
                os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
                
                # Create summary analysis
                summary = self._create_analysis_summary(analyses, target_date)
                
                final_output = {
                    'summary': summary,
                    'individual_analyses': analyses,
                    'metadata': {
                        'analysis_date': target_date,
                        'total_videos': len(analyses),
                        'channels_analyzed': list(set(a['channel'] for a in analyses))
                    }
                }
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                
                return f"✅ Enhanced analysis complete: {len(analyses)} videos analyzed with metadata. Results saved to {analysis_file}"
            else:
                return "❌ No successful analyses generated"
        
        except Exception as e:
            error = f"❌ Analysis error: {str(e)}"
            self.context["errors"].append(error)
            return error
    
    def _analyze_with_enhanced_metadata(self, video_info: dict, transcript: str, metadata: dict, transcription_data: dict, target_date: str) -> dict:
        """Enhanced analysis with comprehensive metadata context"""
        try:
            # Enhanced metadata context
            engagement_ratio = metadata.get('engagement_metrics', {}).get('engagement_ratio', 0)
            channel_authority = self._assess_channel_authority(metadata)
            audio_quality = transcription_data.get('audio_quality', 'unknown')
            transcript_confidence = transcription_data.get('confidence_segments', 0)
            
            # Comprehensive metadata context for LLM
            metadata_context = f"""
VIDEO METADATA CONTEXT:
- Channel: {metadata.get('channel_name', 'Unknown')}
- Channel Authority: {channel_authority}
- View Count: {metadata.get('engagement_metrics', {}).get('view_count', 0):,}
- Engagement Ratio: {engagement_ratio:.1f}%
- Duration: {metadata.get('video_info', {}).get('duration_formatted', 'Unknown')}
- Upload Date: {metadata.get('video_info', {}).get('upload_date', 'Unknown')}
- Audio Quality: {audio_quality}
- Transcript Confidence: {transcript_confidence} reliable segments
- Transcript Length: {transcription_data.get('transcript_length', 0)} characters

CHANNEL CREDIBILITY ASSESSMENT:
- Follower Count: {metadata.get('channel_info', {}).get('follower_count', 0):,}
- Verified Status: {metadata.get('channel_info', {}).get('is_verified', False)}
- Historical Performance: {engagement_ratio:.1f}% engagement
"""
            
            # OpenAI analysis with enhanced context
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are an expert Indian stock market analyst with deep knowledge of Telugu financial content. 
Analyze the provided video transcript considering the channel's credibility metrics, engagement data, and audio quality.
Provide actionable investment insights with confidence levels based on source reliability."""},
                    {"role": "user", "content": f"""
{metadata_context}

TRANSCRIPT TO ANALYZE:
{transcript[:3000]}

Analyze this Telugu stock market video for investment insights. Consider all metadata factors when determining confidence and credibility.

Respond in JSON format with enhanced analysis:
{{
  "recommendations": ["specific actionable advice"],
  "stocks_mentioned": ["company names with sectors"],
  "market_sentiment": "bullish/bearish/neutral",
  "confidence_score": 0.0-1.0,
  "source_credibility": "high/medium/low",
  "credibility_factors": ["reasons for credibility assessment"],
  "key_insights": ["important market insights"],
  "risk_assessment": ["potential risks mentioned"],
  "target_audience": "retail/institutional/both",
  "actionability": "immediate/short-term/long-term",
  "content_quality": "excellent/good/average/poor"
}}
"""}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            ai_analysis_text = response.choices[0].message.content
            
            # Parse enhanced JSON response
            try:
                if "```json" in ai_analysis_text:
                    json_start = ai_analysis_text.find("```json") + 7
                    json_end = ai_analysis_text.find("```", json_start)
                    ai_analysis_text = ai_analysis_text[json_start:json_end]
                
                ai_analysis = json.loads(ai_analysis_text.strip())
                
                # Return comprehensive analysis
                return {
                    'channel': video_info.get('channel', 'Unknown'),
                    'video_title': video_info.get('title', metadata.get('video_info', {}).get('title', 'Unknown')),
                    'analysis_date': target_date,
                    'recommendations': ai_analysis.get('recommendations', []),
                    'stocks_mentioned': ai_analysis.get('stocks_mentioned', []),
                    'market_sentiment': ai_analysis.get('market_sentiment', 'neutral'),
                    'confidence_score': ai_analysis.get('confidence_score', 0.0),
                    'source_credibility': ai_analysis.get('source_credibility', 'unknown'),
                    'credibility_factors': ai_analysis.get('credibility_factors', []),
                    'key_insights': ai_analysis.get('key_insights', []),
                    'risk_assessment': ai_analysis.get('risk_assessment', []),
                    'target_audience': ai_analysis.get('target_audience', 'unknown'),
                    'actionability': ai_analysis.get('actionability', 'unknown'),
                    'content_quality': ai_analysis.get('content_quality', 'unknown'),
                    'metadata_used': metadata,
                    'engagement_ratio': engagement_ratio,
                    'channel_authority': channel_authority,
                    'audio_quality': audio_quality,
                    'transcript_confidence': transcript_confidence
                }
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON parsing failed: {e}")
                return self._fallback_analysis(video_info, transcript, target_date, metadata)
        
        except Exception as e:
            print(f"⚠️ OpenAI analysis failed: {e}")
            return self._fallback_analysis(video_info, transcript, target_date, metadata)
    
    def _assess_channel_authority(self, metadata: dict) -> str:
        """Enhanced channel authority assessment"""
        followers = metadata.get('channel_info', {}).get('follower_count', 0)
        is_verified = metadata.get('channel_info', {}).get('is_verified', False)
        engagement_ratio = metadata.get('engagement_metrics', {}).get('engagement_ratio', 0)
        
        # Base authority from followers
        if followers > 1000000:
            authority = "HIGH"
        elif followers > 500000:
            authority = "MEDIUM-HIGH"
        elif followers > 100000:
            authority = "MEDIUM"
        elif followers > 10000:
            authority = "MEDIUM-LOW"
        else:
            authority = "LOW"
        
        # Adjust based on engagement
        if engagement_ratio > 5.0:
            authority += " (High Engagement)"
        elif engagement_ratio > 2.0:
            authority += " (Good Engagement)"
        else:
            authority += " (Low Engagement)"
        
        if is_verified:
            authority += " (Verified)"
        
        return authority
    
    def _create_analysis_summary(self, analyses: List[dict], target_date: str) -> dict:
        """Create comprehensive analysis summary"""
        if not analyses:
            return {}
        
        # Aggregate insights
        all_stocks = []
        all_recommendations = []
        sentiments = []
        confidence_scores = []
        
        for analysis in analyses:
            all_stocks.extend(analysis.get('stocks_mentioned', []))
            all_recommendations.extend(analysis.get('recommendations', []))
            sentiments.append(analysis.get('market_sentiment', 'neutral'))
            confidence_scores.append(analysis.get('confidence_score', 0.0))
        
        # Calculate summary metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        most_common_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else 'neutral'
        unique_stocks = list(set(all_stocks))
        
        return {
            'date': target_date,
            'total_videos_analyzed': len(analyses),
            'overall_market_sentiment': most_common_sentiment,
            'average_confidence': round(avg_confidence, 2),
            'unique_stocks_mentioned': unique_stocks,
            'total_recommendations': len(all_recommendations),
            'high_confidence_analyses': len([a for a in analyses if a.get('confidence_score', 0) > 0.7]),
            'channels_covered': list(set(a['channel'] for a in analyses)),
            'top_insights': all_recommendations[:10] if all_recommendations else []
        }
    
    def _fallback_analysis(self, video_info: dict, transcript: str, target_date: str, metadata: dict = None) -> dict:
        """Enhanced fallback analysis when AI fails"""
        return {
            'channel': video_info.get('channel', 'Unknown'),
            'video_title': video_info.get('title', 'Unknown'),
            'analysis_date': target_date,
            'recommendations': ['Analysis failed - manual review required'],
            'stocks_mentioned': [],
            'market_sentiment': 'neutral',
            'confidence_score': 0.1,
            'source_credibility': 'unknown',
            'credibility_factors': ['AI analysis failed'],
            'key_insights': ['Basic analysis - AI processing failed'],
            'risk_assessment': ['Unable to assess due to processing failure'],
            'target_audience': 'unknown',
            'actionability': 'unknown',
            'content_quality': 'unknown',
            'fallback': True,
            'metadata_available': metadata is not None
        }

    def run_complete_analysis(self, channels: str = "moneypurse,daytradertelugu", target_date: str = None) -> str:
        """Run complete enhanced workflow with metadata integration"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🚀 Starting complete enhanced analysis for {target_date}")
        print(f"📺 Channels: {channels}")
        
        # Step 1: Download with metadata
        download_result = self.download_videos(channels, target_date)
        print(f"📥 Download: {download_result}")
        
        # Step 2: Transcribe with metadata integration
        transcribe_result = self.transcribe_videos(target_date)
        print(f"🎙️ Transcribe: {transcribe_result}")
        
        # Step 3: Analyze with enhanced metadata
        analysis_result = self.analyze_content(target_date)
        print(f"📊 Analysis: {analysis_result}")
        
        # Generate final summary
        summary = f"""
🎉 COMPLETE ANALYSIS FINISHED FOR {target_date}

📈 RESULTS SUMMARY:
- Videos Downloaded: {len(self.context.get('videos', []))}
- Videos Transcribed: {len(self.context.get('transcriptions', []))}
- Videos Analyzed: {len(self.context.get('analyses', []))}
- Errors Encountered: {len(self.context.get('errors', []))}

📊 Check ./data/analysis/swarm_analysis_{target_date}.json for detailed results
        """
        
        print(summary)
        return f"✅ Complete enhanced analysis finished for {target_date}"

def main():
    """Test the enhanced Swarm agent"""
    print("🤖 Initializing Enhanced Swarm Stock News Agent...")
    agent = SwarmStockNewsAgent()
    result = agent.run_complete_analysis()
    print(f"\n🏁 Final result: {result}")

if __name__ == "__main__":
    main()
