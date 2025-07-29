#!/usr/bin/env python3
"""
Complete Swarm Agent for Daily Stock News Analysis (Refactored to use tools)
- Uses centralized YouTube processing tool for downloads
- Transcribes using Whisper with metadata integration
- Analyzes content with channel authority assessment
-                 # Check if transcript already exists - use fixed channel names
                transcript_files = [
                    f"./data/transcripts/{target_date}/{channel}.json"
                ]
                
                # For daytradertelugu, also check the Telugu script filename as fallback
                if channel == 'daytradertelugu':
                    transcript_files.append(f"./data/transcripts/{target_date}/daytraderతెలుగు.json")ow automation with enhanced metadata usage
"""

import os
import sys
import json
import asyncio
import whisper
from datetime import datetime
from typing import Dict, Any, List
import openai
from swarm import Swarm, Agent

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")

# Add parent directory to path to import tools
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from tools import YouTubeProcessingTool, ToolConfig, ToolCategory, ToolPriority
from tools.speech_to_text_tool import SpeechToTextTool, TranscriptionConfig, TranscriptionProvider
from tools.content_analysis_tool import ContentAnalysisTool
from tools.report_generation_tool import ReportGenerationTool

class SwarmStockNewsAgent:
    def __init__(self):
        """Initialize with enhanced context tracking and tool integration"""
        self.context = {
            "videos": [],
            "transcriptions": [],
            "analyses": [],
            "errors": [],
            "metadata": {}
        }
        self.client = Swarm()
        
        # Initialize YouTube processing tool
        youtube_config = ToolConfig(
            name="youtube_processor",
            category=ToolCategory.YOUTUBE,
            priority=ToolPriority.HIGH,
            settings={
                'download_path': './data/videos'
            }
        )
        self.youtube_tool = YouTubeProcessingTool(youtube_config)
        
        # Initialize Speech-to-Text tool
        transcription_config = ToolConfig(
            name="speech_to_text",
            category=ToolCategory.TRANSCRIPTION,
            priority=ToolPriority.HIGH,
            settings={
                'provider': TranscriptionProvider.WHISPER,
                'model_size': 'base',
                'output_path': './data/transcripts'  # Will use date-based subdirectories
            }
        )
        self.transcription_tool = SpeechToTextTool(transcription_config)
        
        # Initialize Content Analysis tool
        analysis_config = ToolConfig(
            name="content_analyzer",
            category=ToolCategory.ANALYSIS,
            priority=ToolPriority.HIGH,
            settings={
                'output_path': './data/analyses',  # Will use date-based subdirectories
                'analysis_types': ['stock_recommendations', 'market_sentiment', 'sector_analysis'],
                'ai_provider': 'openai',  # Use OpenAI for LLM analysis
                'ai_api_key': os.getenv('OPENAI_API_KEY')  # Get API key from environment
            }
        )
        self.analysis_tool = ContentAnalysisTool(analysis_config)
        
        # Initialize Report Generation tool
        report_config = ToolConfig(
            name="report_generator", 
            category=ToolCategory.GENERATION,
            priority=ToolPriority.MEDIUM,
            settings={
                'output_path': './data/reports',  # Will use date-based subdirectories
                'formats': ['markdown', 'html', 'json']
            }
        )
        self.report_tool = ReportGenerationTool(report_config)
        
        # Initialize specialized agents
        self.coordinator = Agent(
            name="Coordinator",
            instructions="You coordinate the daily stock news analysis workflow. You manage downloading videos, transcription, and analysis with metadata integration.",
        )
        
        self.downloader = Agent(
            name="Downloader", 
            instructions="You download YouTube videos from Telugu financial channels using the centralized YouTube processing tool.",
            functions=[self.download_videos]
        )
        
        self.transcriber = Agent(
            name="Transcriber",
            instructions="You transcribe video content using the centralized speech-to-text tool with metadata integration.",
            functions=[self.transcribe_videos_sync]
        )
        
        self.analyzer = Agent(
            name="Analyzer", 
            instructions="You analyze transcribed content for stock insights using the centralized content analysis tool with metadata context.",
            functions=[self.analyze_content_sync]
        )

    async def download_videos_async(self, channels: str = "moneypurse,daytradertelugu", target_date: str = None) -> Dict[str, Any]:
        """Download videos using the centralized YouTube processing tool"""
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📥 Starting download for {target_date} using YouTube processing tool")
        
        try:
            # Initialize the tool if not already done
            if not hasattr(self.youtube_tool, '_is_initialized') or not self.youtube_tool._is_initialized:
                init_result = await self.youtube_tool.initialize()
                if not init_result.success:
                    return {
                        "success": False,
                        "error": f"Failed to initialize YouTube tool: {init_result.error_message}",
                        "videos": []
                    }
            
            # Use the YouTube tool to download videos
            result = await self.youtube_tool.execute(
                operation="download_latest",
                date=target_date.replace('-', '')  # Convert to YYYYMMDD format
            )
            
            if result.success:
                downloaded_videos = result.data.get("downloaded_videos", [])
                print(f"✅ Successfully downloaded {len(downloaded_videos)} videos")
                
                # Convert to format expected by rest of swarm agent
                # Use fixed channel names since we only process moneypurse and daytradertelugu
                video_data = []
                for video in downloaded_videos:
                    if hasattr(video, 'file_path'):
                        # Determine channel name from file path
                        file_path = video.file_path
                        if 'moneypurse' in file_path.lower():
                            channel_name = 'moneypurse'
                        elif 'daytrader' in file_path.lower():
                            channel_name = 'daytradertelugu'
                        else:
                            # Skip unknown channels
                            continue
                        
                        video_data.append({
                            'channel': channel_name,
                            'date': target_date,
                            'wav_file': video.file_path,
                            'title': getattr(video, 'title', 'Unknown'),
                            'duration': getattr(video, 'duration', 0),
                            'upload_date': getattr(video, 'upload_date', ''),
                            'view_count': getattr(video, 'view_count', 0),
                        })
                
                self.context["videos"].extend(video_data)
                
                return {
                    "success": True,
                    "videos": video_data,
                    "message": f"Downloaded {len(video_data)} videos for {target_date}"
                }
            else:
                error_msg = result.error_message or "Unknown error"
                print(f"❌ Download failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "videos": []
                }
                
        except Exception as e:
            error_msg = f"Exception during download: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "videos": []
            }

    def download_videos(self, channels: str = "moneypurse,daytradertelugu", target_date: str = None) -> str:
        """Synchronous wrapper for download_videos_async to work with Swarm"""
        
        # Run the async function
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            self.download_videos_async(channels, target_date)
        )
        
        if result["success"]:
            return json.dumps({
                "status": "success",
                "message": result["message"],
                "video_count": len(result["videos"]),
                "videos": [
                    {
                        "channel": v["channel"],
                        "title": v["title"],
                        "file": v["wav_file"]
                    } for v in result["videos"]
                ]
            }, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "message": result["error"],
                "video_count": 0
            }, indent=2)

    def transcribe_videos_sync(self, target_date: str = None) -> str:
        """Sync wrapper for transcribe_videos"""
        return asyncio.run(self.transcribe_videos(target_date))

    def analyze_content_sync(self, target_date: str = None) -> str:
        """Sync wrapper for analyze_content"""
        return asyncio.run(self.analyze_content(target_date))

    async def transcribe_videos(self, target_date: str = None) -> str:
        """Transcribe videos using centralized speech-to-text tool"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🎙️ Transcribing videos for date: {target_date}")
            
        try:
            # Initialize the transcription tool if not already done
            if not hasattr(self.transcription_tool, '_is_initialized') or not self.transcription_tool._is_initialized:
                init_result = await self.transcription_tool.initialize()
                if not init_result.success:
                    return json.dumps({
                        "error": f"Failed to initialize transcription tool: {init_result.error_message}",
                        "transcribed": 0
                    })
            
            videos = self.context.get("videos", [])
            print(f"🔍 Videos in context: {len(videos)}")
            for i, video in enumerate(videos):
                print(f"   Video {i+1}: channel={video.get('channel')}, wav_file={video.get('wav_file')}")
            
            if not videos:
                print("❌ No videos to transcribe - skipping transcription phase")
                return "❌ No videos to transcribe"

            print(f"🎙️ Transcribing {len(videos)} videos using centralized tool...")
            
            transcriptions = []
            
            for video in videos:
                wav_file = video.get('wav_file')
                channel = video['channel']
                
                print(f"🔍 Processing video for channel: {channel}")
                print(f"🔍 WAV file path: {wav_file}")
                print(f"🔍 Target date: {target_date}")
                
                if not wav_file or not os.path.exists(wav_file):
                    print(f"⚠️ WAV file not found: {wav_file}")
                    continue
                
                # Extract the actual video date from the file path instead of using target_date
                # WAV file format: ./data/videos/2025-07-25/moneypurse.wav
                video_date = target_date  # default fallback
                if '/data/videos/' in wav_file:
                    try:
                        video_date = wav_file.split('/data/videos/')[1].split('/')[0]
                        print(f"🔍 Extracted video date from path: {video_date}")
                    except:
                        print(f"⚠️ Could not extract date from path, using target_date: {target_date}")
                
                # Check if transcript already exists - use the video's actual date
                transcript_files = [
                    f"./data/transcripts/{video_date}/{channel}.json"
                ]
                
                # Add alternative filenames for daytradertelugu (it might have Telugu script filename)
                if channel == 'daytradertelugu':
                    transcript_files.extend([
                        f"./data/transcripts/{video_date}/daytraderతెలుగు.json",
                        f"./data/transcripts/{video_date}/daytradertelugu.json"
                    ])
                
                print(f"🔍 Looking for transcript files:")
                transcript_dir_path = f"./data/transcripts/{video_date}"
                print(f"🔍 Transcript directory: {transcript_dir_path}")
                if os.path.exists(transcript_dir_path):
                    files_in_dir = os.listdir(transcript_dir_path)
                    print(f"🔍 Files in transcript directory: {files_in_dir}")
                else:
                    print(f"🔍 Transcript directory does not exist: {transcript_dir_path}")
                
                for tf in transcript_files:
                    exists = os.path.exists(tf)
                    print(f"   - {tf} {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
                
                existing_transcript_file = None
                for transcript_file in transcript_files:
                    if os.path.exists(transcript_file):
                        existing_transcript_file = transcript_file
                        print(f"🎯 Found existing transcript: {existing_transcript_file}")
                        break
                
                if not existing_transcript_file:
                    print(f"🔍 No existing transcript found for {channel}, will proceed with transcription")
                
                if existing_transcript_file:
                    print(f"✅ Transcript already exists for {channel}, loading from {os.path.basename(existing_transcript_file)}")
                    
                    try:
                        with open(existing_transcript_file, 'r', encoding='utf-8') as f:
                            existing_transcript = json.load(f)
                        
                        # Create transcription data from existing file
                        transcription_data = {
                            'channel': channel,
                            'date': target_date,
                            'wav_file': wav_file,
                            'title': video.get('title', 'Unknown'),
                            'duration': video.get('duration', 0),
                            'upload_date': video.get('upload_date', ''),
                            'view_count': video.get('view_count', 0),
                            'text': existing_transcript.get('original_text', ''),
                            'segments': existing_transcript.get('segments', []),
                            'language': existing_transcript.get('language_detected', 'unknown'),
                            'confidence': existing_transcript.get('confidence', 0.0),
                            'processing_time': existing_transcript.get('processing_time', 0.0),
                            'text_length': len(existing_transcript.get('original_text', '')),
                            'provider': existing_transcript.get('provider', 'existing')
                        }
                        
                        transcriptions.append(transcription_data)
                        print(f"✅ Loaded existing transcript for {channel}: {transcription_data['text_length']} characters")
                        continue
                        
                    except Exception as e:
                        print(f"⚠️ Could not load existing transcript for {channel}: {e}")
                        # Fall through to transcribe
                
                print(f"🎙️ Transcribing {channel}...")
                
                # Use centralized transcription tool - await properly in async context
                result = await self.transcription_tool.execute(
                    audio_file=wav_file,
                    provider='whisper',
                    output_format='json'
                )
                
                if result.success:
                    # Enhanced transcription with metadata context
                    transcription_obj = result.data["transcription"]
                    transcription_data = {
                        'channel': video['channel'],
                        'date': target_date,
                        'wav_file': wav_file,
                        'title': video.get('title', 'Unknown'),
                        'duration': video.get('duration', 0),
                        'upload_date': video.get('upload_date', ''),
                        'view_count': video.get('view_count', 0),
                        'text': transcription_obj.original_text,
                        'segments': transcription_obj.segments,
                        'language': transcription_obj.language_detected,
                        'confidence': transcription_obj.confidence,
                        'processing_time': transcription_obj.processing_time,
                        'text_length': len(transcription_obj.original_text),
                        'provider': transcription_obj.provider
                    }
                    
                    # The centralized tool already saves to ./data/transcripts/{date}/channelname.json
                    # So we just need to load it back for our context
                    transcriptions.append(transcription_data)
                    print(f"✅ Transcribed {video['channel']}: {len(transcription_obj.original_text)} characters")
                else:
                    print(f"⚠️ Transcription failed for {video['channel']}: {result.error_message}")
            
            self.context["transcriptions"].extend(transcriptions)
            
            return json.dumps({
                'status': 'success',
                'transcribed': len(transcriptions),
                'transcriptions': [{
                    'channel': t['channel'],
                    'title': t['title'],
                    'text_length': len(t['text']),
                    'language': t['language'],
                    'confidence': f"{t['confidence']:.2f}",
                    'transcript_file': f"./data/transcripts/{target_date}/{t['channel']}.json"
                } for t in transcriptions]
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            print(f"❌ {error_msg}")
            return json.dumps({
                'status': 'error',
                'message': error_msg,
                'transcribed': 0
            }, indent=2)

    async def analyze_content(self, target_date: str = None) -> str:
        """Analyze transcribed content using centralized content analysis tool"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"� Analyzing content for date: {target_date}")
        
        try:
            # Initialize the content analysis tool if not already done
            if not hasattr(self.analysis_tool, '_is_initialized') or not self.analysis_tool._is_initialized:
                init_result = await self.analysis_tool.initialize()
                if not init_result.success:
                    return json.dumps({
                        "error": f"Failed to initialize analysis tool: {init_result.error_message}",
                        "analyzed": 0
                    })
            
            transcriptions = self.context.get("transcriptions", [])
            
            # If no transcriptions in context, try to load from existing transcript files
            if not transcriptions:
                print(f"📂 No transcriptions in context, loading from files for {target_date}")
                transcript_dir = f"./data/transcripts/{target_date}"
                
                if os.path.exists(transcript_dir):
                    # Load existing transcripts for both channels
                    channels = ['moneypurse', 'daytradertelugu']
                    
                    for channel in channels:
                        # Check for transcript files (prefer English names)
                        transcript_files = [
                            f"{transcript_dir}/{channel}.json"
                        ]
                        
                        # For daytradertelugu, also check Telugu script filename
                        if channel == 'daytradertelugu':
                            transcript_files.append(f"{transcript_dir}/daytraderతెలుగు.json")
                        
                        existing_transcript_file = None
                        for transcript_file in transcript_files:
                            if os.path.exists(transcript_file):
                                existing_transcript_file = transcript_file
                                break
                        
                        if existing_transcript_file:
                            try:
                                print(f"📥 Loading existing transcript: {os.path.basename(existing_transcript_file)}")
                                with open(existing_transcript_file, 'r', encoding='utf-8') as f:
                                    existing_transcript = json.load(f)
                                
                                # Create transcription data from existing file
                                transcription_data = {
                                    'channel': channel,
                                    'date': target_date,
                                    'wav_file': f"./data/videos/{target_date}/{channel}.wav",  # Assumed path
                                    'title': existing_transcript.get('title', 'Unknown'),
                                    'duration': existing_transcript.get('duration', 0),
                                    'upload_date': existing_transcript.get('upload_date', ''),
                                    'view_count': existing_transcript.get('view_count', 0),
                                    'text': existing_transcript.get('original_text', ''),
                                    'segments': existing_transcript.get('segments', []),
                                    'language': existing_transcript.get('language_detected', 'unknown'),
                                    'confidence': existing_transcript.get('confidence', 0.0),
                                    'processing_time': existing_transcript.get('processing_time', 0.0),
                                    'text_length': len(existing_transcript.get('original_text', '')),
                                    'provider': existing_transcript.get('provider', 'existing')
                                }
                                
                                transcriptions.append(transcription_data)
                                print(f"✅ Loaded transcript for {channel}: {transcription_data['text_length']} characters")
                                
                            except Exception as e:
                                print(f"⚠️ Could not load transcript for {channel}: {e}")
                        else:
                            print(f"⚠️ No transcript found for {channel}")
                else:
                    print(f"⚠️ Transcript directory not found: {transcript_dir}")
            
            if not transcriptions:
                return json.dumps({
                    "error": "No transcriptions to analyze",
                    "analyzed": 0,
                    "target_date": target_date
                })

            print(f"📊 Analyzing {len(transcriptions)} transcriptions using centralized tool...")
            
            analyses = []
            
            for transcription in transcriptions:
                print(f"📊 Analyzing {transcription['channel']}...")
                
                # Load video metadata if available
                video_metadata = {}
                try:
                    # Extract the actual video date from the wav_file path if available
                    # transcription['wav_file'] format: ./data/videos/2025-07-25/moneypurse.wav
                    video_date_for_metadata = target_date  # default fallback
                    wav_file = transcription.get('wav_file', '')
                    
                    if wav_file and '/data/videos/' in wav_file:
                        try:
                            video_date_for_metadata = wav_file.split('/data/videos/')[1].split('/')[0]
                            print(f"🔍 Using video date for metadata lookup: {video_date_for_metadata}")
                        except:
                            print(f"⚠️ Could not extract date from wav_file path: {wav_file}")
                    
                    # Try to load metadata from the correct video date directory
                    metadata_sources = [
                        f"./data/videos/{video_date_for_metadata}/{transcription['channel']}.json",
                        f"./data/videos/{video_date_for_metadata}/{transcription['channel']}.info.json"
                    ]
                    
                    for metadata_path in metadata_sources:
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                video_metadata = json.load(f)
                            print(f"📋 Loaded metadata from {metadata_path}")
                            break
                    else:
                        print(f"⚠️ No metadata found for {transcription['channel']} in {video_date_for_metadata}")
                except Exception as e:
                    print(f"⚠️ Could not load metadata: {e}")
                
                # Use centralized content analysis tool - await properly in async context
                result = await self.analysis_tool.execute(
                    transcript_text=transcription['text'],
                    channel=transcription['channel'],
                    video_title=transcription['title'],
                    analysis_types=['stock_recommendations', 'market_sentiment', 'sector_analysis', 'technical_analysis'],
                    video_metadata=video_metadata,
                    target_date=target_date
                )
                
                if result.success:
                    # Enhanced analysis with metadata
                    enhanced_analysis = {
                        'channel': transcription['channel'],
                        'date': target_date,
                        'title': transcription['title'],
                        'duration': transcription['duration'],
                        'view_count': transcription['view_count'],
                        'transcript_confidence': transcription['confidence'],
                        'analysis': result.data,
                        'metadata': {
                            'upload_date': transcription['upload_date'],
                            'language': transcription['language'],
                            'text_length': len(transcription['text'])
                        }
                    }
                    
                    # The centralized tool already saves to ./data/analyses/{date}/channelname_analysis.json
                    analyses.append(enhanced_analysis)
                    print(f"✅ Analyzed {transcription['channel']}")
                else:
                    print(f"⚠️ Analysis failed for {transcription['channel']}: {result.error_message}")
                    analyses.append({
                        'channel': transcription['channel'],
                        'error': result.error_message,
                        'date': target_date
                    })
            
            # The centralized tool saves analyses in date-based directories
            self.context["analyses"].extend(analyses)
            
            return json.dumps({
                'status': 'success',
                'analyzed': len(analyses),
                'analysis_path': f'./data/analyses/{target_date}/',
                'summary': {
                    'total_channels': len(analyses),
                    'successful_analyses': len([a for a in analyses if 'error' not in a]),
                    'failed_analyses': len([a for a in analyses if 'error' in a])
                }
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return json.dumps({
                'status': 'error',
                'message': error_msg,
                'analyzed': 0
            }, indent=2)

    async def generate_reports(self, target_date: str = None) -> str:
        """Generate reports using centralized report generation tool"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"� Generating reports for date: {target_date}")
            
        try:
            # Initialize the report tool if not already done
            if not hasattr(self.report_tool, '_is_initialized') or not self.report_tool._is_initialized:
                init_result = await self.report_tool.initialize()
                if not init_result.success:
                    return json.dumps({
                        "error": f"Failed to initialize report tool: {init_result.error_message}",
                        "generated": 0
                    })
            
            analyses = self.context.get("analyses", [])
            
            # If no analyses in context, try to load from files
            if not analyses:
                analysis_dir = f"./data/analyses/{target_date}"
                if os.path.exists(analysis_dir):
                    print(f"📂 Loading analyses from {analysis_dir}")
                    for file in os.listdir(analysis_dir):
                        if file.endswith('_analysis.json'):
                            file_path = os.path.join(analysis_dir, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    analysis_data = json.load(f)
                                    analyses.append(analysis_data)
                                print(f"✅ Loaded analysis from {file}")
                            except Exception as e:
                                print(f"⚠️ Could not load {file}: {e}")
                    
                    if analyses:
                        print(f"📊 Loaded {len(analyses)} analyses from files")
                    else:
                        return json.dumps({"error": f"No analysis files found in {analysis_dir}", "generated": 0})
                else:
                    return json.dumps({"error": f"Analysis directory not found: {analysis_dir}", "generated": 0})

            # Use the passed target_date directly since it's already the correct video date
            actual_video_date = target_date
            print(f"📊 Using passed video date: {actual_video_date}")
            print(f"📊 Generating reports for {len(analyses)} analyses using centralized tool...")
            
            # Use centralized report generation tool with the actual video date
            result = await self.report_tool.execute(
                analysis_data=analyses,
                report_date=actual_video_date,  # Use actual video date instead of target_date
                output_formats=['markdown', 'json'],
                include_comparison=True
            )
            
            if result.success:
                print(f"✅ Reports generated: {result.data.get('output_files', [])}")
                
                return json.dumps({
                    'status': 'success',
                    'generated': len(result.data.get('output_files', [])),
                    'output_files': result.data.get('output_files', []),
                    'target_date': actual_video_date  # Use actual video date in response
                })
            else:
                print(f"❌ Report generation failed: {result.error_message}")
                return json.dumps({
                    'status': 'failed',
                    'error': result.error_message,
                    'generated': 0
                })
                
        except Exception as e:
            print(f"❌ Report generation error: {e}")
            return json.dumps({
                'status': 'error',
                'error': str(e),
                'generated': 0
            })

    async def run_full_workflow(self, target_date: str = None) -> Dict[str, Any]:
        """Run the complete workflow using tools"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"🚀 Starting complete workflow for {target_date}")
        
        # Determine the actual video date upfront (apply weekend logic once)
        actual_video_date = target_date
        try:
            from datetime import datetime, timedelta
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            weekday = date_obj.weekday()  # Monday=0, Sunday=6
            
            if weekday == 5:  # Saturday
                friday_date = date_obj - timedelta(days=1)
                actual_video_date = friday_date.strftime('%Y-%m-%d')
                print(f"📅 Weekend detected (Saturday), using Friday's date: {actual_video_date}")
            elif weekday == 6:  # Sunday
                friday_date = date_obj - timedelta(days=2)
                actual_video_date = friday_date.strftime('%Y-%m-%d')
                print(f"📅 Weekend detected (Sunday), using Friday's date: {actual_video_date}")
            else:
                print(f"📅 Using target date: {actual_video_date}")
        except Exception as e:
            print(f"⚠️ Error applying weekend logic: {e}")
            actual_video_date = target_date
        
        print(f"🎯 Using video date: {actual_video_date} for all operations")
        
        # Phase 1: Download videos using YouTube tool
        download_result = await self.download_videos_async(target_date=target_date)  # Still use original date for download
        if not download_result["success"]:
            return {
                "status": "failed",
                "phase": "download",
                "error": download_result["error"]
            }
        
        # Phase 2: Transcribe videos (use actual video date)
        transcription_result = await self.transcribe_videos(target_date=actual_video_date)
        
        # Phase 3: Analyze content (use actual video date)
        analysis_result = await self.analyze_content(target_date=actual_video_date)
        
        # Phase 4: Generate reports (use actual video date)
        report_result = await self.generate_reports(target_date=actual_video_date)
        
        return {
            "status": "success",
            "date": target_date,
            "video_date": actual_video_date,  # Include both dates in response
            "phases": {
                "download": download_result,
                "transcription": json.loads(transcription_result),
                "analysis": json.loads(analysis_result),
                "reports": json.loads(report_result) if isinstance(report_result, str) else report_result
            }
        }

def main():
    """Main execution function"""
    agent = SwarmStockNewsAgent()
    
    # Run with today's date
    target_date = datetime.now().strftime('%Y-%m-%d')
    print(f"🔄 Running Daily Stock News Agent for {target_date}")
    
    try:
        # Use asyncio to run the workflow
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(agent.run_full_workflow(target_date))
        
        print("\n" + "="*50)
        print("📊 WORKFLOW COMPLETE")
        print("="*50)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")

if __name__ == "__main__":
    main()
