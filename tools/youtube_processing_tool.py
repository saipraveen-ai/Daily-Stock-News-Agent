"""
YouTube Processing Tool for Daily Stock News Agent

This tool handles downloading and processing YouTube videos from specified channels.
It monitors for new uploads, downloads audio/video content, and prepares files
for transcription processing.
"""

import os
import asyncio
import yt_dlp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from .base_tool import BaseTool, ToolResult, ToolConfig, ToolCategory, ToolPriority

logger = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Configuration for a YouTube channel"""
    name: str
    url: str
    channel_id: str
    upload_time_range: str = "18:00-20:00"  # Expected upload window
    max_video_duration: int = 3600  # 1 hour in seconds
    quality_preference: str = "best[height>=1080][acodec!=none][abr>=128]"  # Ultimate quality for transcription


@dataclass
class VideoInfo:
    """Information about a downloaded video"""
    video_id: str
    title: str
    channel: str
    upload_date: str
    duration: int
    file_path: str
    thumbnail_url: str
    description: str
    view_count: int = 0
    like_count: int = 0


class YouTubeProcessingTool(BaseTool):
    """
    Tool for processing YouTube videos from stock market analysis channels.
    
    Features:
    - Monitor specified channels for new uploads
    - Download audio/video content optimized for transcription
    - Extract metadata and video information
    - Handle rate limiting and error recovery
    - Support for multiple channels simultaneously
    """
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        
        # Default channel configurations
        self.channels = {
            "moneypurse": ChannelConfig(
                name="MoneyPurse",
                url="https://www.youtube.com/@MoneyPurse",
                channel_id="@MoneyPurse",
                upload_time_range="18:00-20:00"
            ),
            "daytradertelugu": ChannelConfig(
                name="Day Trader Telugu", 
                url="https://www.youtube.com/@daytradertelugu",
                channel_id="@daytradertelugu",
                upload_time_range="18:00-20:00"
            )
        }
        
        # yt-dlp configuration optimized for best transcription quality with date-based organization
        self.base_download_path = config.settings.get('download_path', './data/videos')
        
        # Channel mapping - matches download_best_quality.py and swarm_agent_fixed.py
        self.channel_urls = {
            'moneypurse': 'https://www.youtube.com/@MoneyPurse/videos',
            'daytradertelugu': 'https://www.youtube.com/@daytradertelugu/videos',
        }
        
        self.ydl_opts_base = {
            # Ultimate quality for transcription: multi-tier format selection
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
            'playlistend': 1,  # Latest video only
            'playlistreverse': False,
            
            # Audio optimization for Whisper transcription
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }, {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
                'nopostoverwrites': True,
            }],
            
            # Additional quality controls
            'prefer_free_formats': False,
            'youtube_include_dash_manifest': True,
        }
        
        self.download_path = config.settings.get('download_path', './data/videos')
        os.makedirs(self.download_path, exist_ok=True)
        
    async def initialize(self) -> ToolResult:
        """Initialize the YouTube processing tool"""
        try:
            # Verify yt-dlp is working
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                # Test with a simple extraction (no download)
                test_opts = self.ydl_opts_base.copy()
                test_opts['skip_download'] = True
                
            self._is_initialized = True
            self.logger.info("YouTube Processing Tool initialized successfully")
            
            return ToolResult(
                success=True,
                data={"message": "YouTube processing tool ready"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YouTube tool: {e}")
            return ToolResult(
                success=False,
                error_message=f"Initialization failed: {str(e)}"
            )
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute YouTube video processing.
        
        Args:
            operation: str - Type of operation ('check_new', 'download_latest', 'download_by_date')
            date: str - Date to check for videos (YYYY-MM-DD format)
            channel: str - Specific channel to process (optional)
            video_url: str - Specific video URL to download (optional)
            
        Returns:
            ToolResult with processing results
        """
        operation = kwargs.get('operation', 'download_latest')
        
        try:
            if operation == 'check_new':
                return await self._check_new_videos(**kwargs)
            elif operation == 'download_latest':
                return await self._download_latest_videos(**kwargs)
            elif operation == 'download_by_date':
                return await self._download_videos_by_date(**kwargs)
            elif operation == 'download_url':
                return await self._download_specific_video(**kwargs)
            else:
                return ToolResult(
                    success=False,
                    error_message=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            self.logger.error(f"YouTube processing failed: {e}")
            return ToolResult(
                success=False,
                error_message=f"Processing failed: {str(e)}"
            )
    
    async def _check_new_videos(self, **kwargs) -> ToolResult:
        """Check for new videos from monitored channels"""
        date_str = kwargs.get('date', datetime.now().strftime('%Y%m%d'))
        channel_filter = kwargs.get('channel')
        
        new_videos = []
        
        channels_to_check = self.channels
        if channel_filter:
            channels_to_check = {k: v for k, v in self.channels.items() 
                               if k == channel_filter or v.name == channel_filter}
        
        for channel_key, channel_config in channels_to_check.items():
            try:
                videos = await self._get_channel_videos(channel_config, date_str)
                new_videos.extend(videos)
                
            except Exception as e:
                self.logger.error(f"Failed to check {channel_config.name}: {e}")
                continue
        
        return ToolResult(
            success=True,
            data={
                "new_videos": new_videos,
                "count": len(new_videos),
                "date": date_str
            },
            metadata={"operation": "check_new"}
        )
    
    async def _download_latest_videos(self, **kwargs) -> ToolResult:
        """Download the latest videos from all monitored channels"""
        date_str = kwargs.get('date', datetime.now().strftime('%Y%m%d'))
        
        # Handle weekends - if it's Saturday/Sunday, look for Friday's video
        target_datetime = datetime.strptime(date_str, '%Y%m%d')
        if target_datetime.weekday() >= 5:  # Saturday (5) or Sunday (6)
            # Go back to Friday
            days_back = target_datetime.weekday() - 4  # 5-4=1 for Sat, 6-4=2 for Sun
            target_datetime = target_datetime - timedelta(days=days_back)
            date_str = target_datetime.strftime('%Y%m%d')
            print(f"📅 Weekend detected, looking for Friday's videos: {date_str}")
        
        target_date = target_datetime.strftime('%Y-%m-%d')
        
        downloaded_videos = []
        errors = []
        
        for channel_key, channel_config in self.channels.items():
            try:
                # Check if files already exist for this channel and date FIRST
                date_dir = os.path.join(self.base_download_path, target_date)
                channel_name = channel_key  # Use the key as the standardized channel name
                wav_file = os.path.join(date_dir, f'{channel_name}.wav')
                metadata_file = os.path.join(date_dir, f'{channel_name}.json')
                
                if os.path.exists(wav_file) and os.path.exists(metadata_file):
                    print(f"✅ Files already exist for {channel_name}, skipping download")
                    
                    # Load existing metadata and create VideoInfo
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        video_info = VideoInfo(
                            video_id='existing',
                            title=metadata['video_info']['title'],
                            channel=channel_name,
                            upload_date=metadata['video_info']['upload_date'],
                            duration=metadata['video_info']['duration'],
                            file_path=wav_file,
                            thumbnail_url='',
                            description=metadata['video_info']['description'],
                            view_count=metadata['engagement_metrics']['view_count'],
                            like_count=metadata['engagement_metrics']['like_count']
                        )
                        
                        downloaded_videos.append(video_info)
                        continue
                        
                    except Exception as e:
                        print(f"⚠️ Could not load existing metadata for {channel_name}: {e}")
                        # Continue to download
                
                elif os.path.exists(wav_file) and not os.path.exists(metadata_file):
                    print(f"📋 WAV file exists but metadata missing for {channel_name}, generating metadata...")
                    
                    # Look for .info.json file from yt-dlp
                    info_file = os.path.join(date_dir, f'{channel_name}.info.json')
                    if os.path.exists(info_file):
                        try:
                            with open(info_file, 'r', encoding='utf-8') as f:
                                info_data = json.load(f)
                            
                            # Generate structured metadata from .info.json
                            await self._generate_channel_metadata(info_data, metadata_file)
                            print(f"✅ Generated metadata for {channel_name}")
                            
                            # Create VideoInfo from the generated metadata
                            video_info = VideoInfo(
                                video_id=info_data.get('id', 'existing'),
                                title=info_data.get('title', 'Unknown'),
                                channel=channel_name,
                                upload_date=info_data.get('upload_date', ''),
                                duration=info_data.get('duration', 0),
                                file_path=wav_file,
                                thumbnail_url=info_data.get('thumbnail', ''),
                                description=info_data.get('description', ''),
                                view_count=info_data.get('view_count', 0),
                                like_count=info_data.get('like_count', 0)
                            )
                            
                            downloaded_videos.append(video_info)
                            continue
                            
                        except Exception as e:
                            print(f"⚠️ Could not generate metadata from .info.json for {channel_name}: {e}")
                    else:
                        print(f"⚠️ No .info.json file found for {channel_name}")
                
                # Get latest video for the target date (only 1 video per channel)
                videos = await self._get_channel_videos(channel_config, date_str)
                
                if not videos:
                    print(f"ℹ️ No videos found for {channel_config.name} on {date_str}")
                    continue
                
                # Take only the FIRST (latest) video
                video_info = videos[0]
                print(f"📥 Found video for {channel_config.name}: {video_info['title']}")
                
                download_result = await self._download_video(video_info['url'], target_date)
                if download_result.success:
                    downloaded_videos.append(download_result.data)
                    print(f"✅ Downloaded {channel_config.name}")
                else:
                    errors.append(f"{channel_config.name}: {download_result.error_message}")
                    print(f"❌ Failed to download {channel_config.name}: {download_result.error_message}")
                        
            except Exception as e:
                error_msg = f"Failed to process {channel_config.name}: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        success = len(downloaded_videos) > 0
        return ToolResult(
            success=success,
            data={
                "downloaded_videos": downloaded_videos,
                "download_count": len(downloaded_videos),
                "errors": errors
            },
            metadata={"operation": "download_latest", "target_date": target_date, "original_date": kwargs.get('date')}
        )
    
    async def _download_videos_by_date(self, **kwargs) -> ToolResult:
        """Download videos from a specific date"""
        date_str = kwargs.get('date')
        if not date_str:
            return ToolResult(
                success=False,
                error_message="Date parameter is required for download_by_date operation"
            )
        
        # Convert date format if needed
        if len(date_str) == 10 and '-' in date_str:  # YYYY-MM-DD format
            date_str = date_str.replace('-', '')
        
        kwargs['date'] = date_str
        return await self._download_latest_videos(**kwargs)
    
    async def _download_specific_video(self, **kwargs) -> ToolResult:
        """Download a specific video by URL"""
        video_url = kwargs.get('video_url')
        if not video_url:
            return ToolResult(
                success=False,
                error_message="video_url parameter is required"
            )
        
        return await self._download_video(video_url)
    
    async def _get_channel_videos(self, channel_config: ChannelConfig, date_str: str) -> List[Dict[str, Any]]:
        """Get videos from a channel for a specific date - returns only the latest video"""
        videos = []
        
        # Construct search URL for the channel's recent videos
        search_url = f"{channel_config.url}/videos"
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': False,  # Need full extraction to get upload dates
            'playlistend': 5,  # Check only last 5 videos for efficiency
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(search_url, download=False)
                
                if 'entries' in playlist_info:
                    self.logger.info(f"Found {len(playlist_info['entries'])} recent videos for {channel_config.name}")
                    
                    # Look for the latest video from the target date
                    for entry in playlist_info['entries']:
                        if entry:
                            upload_date = entry.get('upload_date', '')
                            title = entry.get('title', 'Unknown')
                            
                            self.logger.debug(f"Video: {title}, Upload date: {upload_date}, Target date: {date_str}")
                            
                            # Exact date matching for the target date
                            if upload_date == date_str:
                                videos.append({
                                    'id': entry['id'],
                                    'title': title,
                                    'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                    'upload_date': upload_date,
                                    'channel': channel_config.name,
                                    'duration': entry.get('duration', 0)
                                })
                                # Return only the FIRST matching video (latest for that date)
                                break
                    
                    # If no video found for exact date, take the latest video (fallback)
                    if not videos and playlist_info['entries']:
                        latest_entry = playlist_info['entries'][0]  # First entry is the latest
                        if latest_entry:
                            print(f"⚠️ No video found for {date_str}, using latest video from {channel_config.name}")
                            videos.append({
                                'id': latest_entry['id'],
                                'title': latest_entry.get('title', 'Unknown'),
                                'url': f"https://www.youtube.com/watch?v={latest_entry['id']}",
                                'upload_date': latest_entry.get('upload_date', 'unknown'),
                                'channel': channel_config.name,
                                'duration': latest_entry.get('duration', 0)
                            })
                                
                else:
                    self.logger.warning(f"No entries found for {channel_config.name}")
                                
        except Exception as e:
            self.logger.error(f"Failed to get videos for {channel_config.name}: {e}")
            raise
        
        self.logger.info(f"Selected {len(videos)} video(s) for {channel_config.name} on {date_str}")
        return videos
    
    async def _download_video(self, video_url: str, target_date: str = None) -> ToolResult:
        """Download a specific video with best quality settings and date-based organization"""
        try:
            if target_date is None:
                target_date = datetime.now().strftime('%Y-%m-%d')
            
            # Create date-based directory
            date_dir = os.path.join(self.base_download_path, target_date)
            os.makedirs(date_dir, exist_ok=True)
            
            # Extract video info to get channel name and video details
            info = None
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as info_ydl:
                    info = info_ydl.extract_info(video_url, download=False)
            except Exception as e:
                self.logger.error(f"❌ Failed to extract video info: {e}")
                return ToolResult(
                    success=False,
                    error_message=f"Failed to extract video information: {str(e)}"
                )
            
            if not info:
                return ToolResult(
                    success=False,
                    error_message="Video info extraction returned None"
                )
            
            channel_name = self._get_channel_name(info.get('channel', ''), info.get('uploader', ''))
            
            # Check video duration
            duration = info.get('duration', 0)
            if duration > 3600:  # More than 1 hour
                return ToolResult(
                    success=False,
                    error_message=f"Video too long: {duration/60:.1f} minutes"
                )
            
            self.logger.info(f"🎥 Downloading {channel_name} to {date_dir}")
            self.logger.info(f"📺 Video: {info.get('title', 'Unknown')}")
            self.logger.info(f"📊 Available formats: {len(info.get('formats', []))}")
            
            # Download the video with best quality settings
            ydl_opts_download = self.ydl_opts_base.copy()
            ydl_opts_download['outtmpl'] = os.path.join(date_dir, f'{channel_name}.%(ext)s')
            
            with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
                ydl.download([video_url])
            
            # Generate channel-specific metadata in the same directory
            metadata_file = os.path.join(date_dir, f'{channel_name}.json')
            await self._generate_channel_metadata(info, metadata_file)
            
            # Find the downloaded WAV file (prioritize WAV for transcription)
            wav_file = os.path.join(date_dir, f'{channel_name}.wav')
            video_file = wav_file  # Default to WAV file
            
            if not os.path.exists(wav_file):
                # Check for other audio/video extensions
                for ext in ['.mp4', '.webm', '.mkv', '.m4a']:
                        alt_file = os.path.join(date_dir, f'{channel_name}{ext}')
                        if os.path.exists(alt_file):
                            video_file = alt_file
                            break
            
            # Create VideoInfo object
            video_info = VideoInfo(
                video_id=info['id'],
                title=info.get('title', 'Unknown'),
                channel=info.get('uploader', 'Unknown'),
                upload_date=info.get('upload_date', ''),
                duration=duration,
                file_path=video_file,
                thumbnail_url=info.get('thumbnail', ''),
                description=info.get('description', ''),
                view_count=info.get('view_count', 0),
                like_count=info.get('like_count', 0)
            )
            
            self.logger.info(f"✅ Downloaded: {channel_name}.mp4 in {date_dir}")
            
            return ToolResult(
                success=True,
                data=video_info,
                metadata={
                    "download_url": video_url, 
                    "high_quality": True, 
                    "metadata_generated": True,
                    "date_organized": target_date,
                    "channel_name": channel_name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download video {video_url}: {e}")
            return ToolResult(
                success=False,
                error_message=f"Download failed: {str(e)}"
            )
    
    def _get_channel_name(self, channel: str, uploader: str) -> str:
        """Extract consistent channel name for file naming"""
        # Priority mapping for known channels
        channel_mapping = {
            'moneypurse': 'moneypurse',
            'money purse': 'moneypurse',
            'daytradertelugu': 'daytradertelugu',
            'day trader telugu': 'daytradertelugu',
        }
        
        # Clean and normalize channel name
        for name in [channel, uploader]:
            if name:
                clean_name = name.lower().replace(' ', '').replace('@', '')
                for key, value in channel_mapping.items():
                    if key in clean_name:
                        return value
        
        # Fallback to cleaned channel or uploader name
        fallback = (channel or uploader or 'unknown').lower().replace(' ', '').replace('@', '')
        return fallback[:20]  # Limit length
    
    async def _generate_channel_metadata(self, video_info: Dict[str, Any], metadata_file: str) -> None:
        """Generate channel-specific metadata for LLM analysis with consistent naming"""
        try:
            # Extract channel name using consistent naming
            channel_name = self._get_channel_name(video_info.get('channel', ''), video_info.get('uploader', ''))
            
            # Extract chapters if available
            chapters = []
            if 'chapters' in video_info and video_info['chapters']:
                for chapter in video_info['chapters']:
                    chapters.append({
                        'title': chapter.get('title', ''),
                        'start_time': chapter.get('start_time', 0),
                        'end_time': chapter.get('end_time', 0),
                        'duration': chapter.get('end_time', 0) - chapter.get('start_time', 0),
                        'start_formatted': self._format_time(chapter.get('start_time', 0)),
                        'end_formatted': self._format_time(chapter.get('end_time', 0))
                    })
            
            # Create consistent metadata structure matching download_best_quality.py
            metadata = {
                'channel_name': channel_name,
                'channel_url': video_info.get('webpage_url', ''),
                'video_info': {
                    'title': video_info.get('title', 'Unknown'),
                    'duration': video_info.get('duration', 0) or 0,
                    'duration_formatted': self._format_duration(video_info.get('duration', 0)),
                    'description': video_info.get('description', ''),
                    'upload_date': video_info.get('upload_date', ''),
                    'view_count': video_info.get('view_count', 0) or 0,
                    'like_count': video_info.get('like_count', 0) or 0,
                    'comment_count': video_info.get('comment_count', 0) or 0,
                },
                'channel_info': {
                    'follower_count': video_info.get('channel_follower_count', 0) or 0,
                    'is_verified': video_info.get('channel_is_verified', False),
                },
                'engagement_metrics': {
                    'view_count': video_info.get('view_count', 0) or 0,
                    'like_count': video_info.get('like_count', 0) or 0,
                    'comment_count': video_info.get('comment_count', 0) or 0,
                    'engagement_ratio': self._calculate_engagement_ratio(
                        video_info.get('like_count', 0), 
                        video_info.get('view_count', 0)
                    )
                },
                'content_structure': {
                    'chapters': chapters,
                    'tags': video_info.get('tags', []) or [],
                    'categories': video_info.get('categories', []) or []
                },
                'download_info': {
                    'download_date': datetime.now().strftime('%Y-%m-%d'),
                    'quality_downloaded': 'best_available',
                    'file_name': f'{channel_name}.mp4'
                }
            }
            
            # Save metadata to the specified file
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📋 Generated metadata: {os.path.basename(metadata_file)}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate metadata: {e}")
            # Print more detailed error info for debugging
            import traceback
            traceback.print_exc()
            raise
    
    def _format_time(self, seconds: int) -> str:
        """Format seconds into HH:MM:SS or MM:SS"""
        if seconds is None:
            return "00:00"
        
        try:
            # Ensure we're working with an integer
            seconds = int(float(seconds))
            
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes:02d}:{secs:02d}"
        except (ValueError, TypeError):
            return "00:00"
    
    def _format_duration(self, duration) -> str:
        """Safely format duration with error handling"""
        try:
            if duration is None:
                return "00:00"
            
            # Convert to int safely
            duration_int = int(float(duration))
            minutes = duration_int // 60
            seconds = duration_int % 60
            return f"{minutes}:{seconds:02d}"
        except (ValueError, TypeError):
            return "00:00"
    
    def _calculate_engagement_ratio(self, like_count, view_count) -> float:
        """Safely calculate engagement ratio with error handling"""
        try:
            likes = like_count or 0
            views = view_count or 0
            
            if views == 0:
                return 0.0
            
            return round((likes / views) * 100, 2)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
    
    def _get_safe_channel_name(self, channel_name: str) -> str:
        """Convert channel name to safe filename format"""
        channel_mapping = {
            'Money Purse { మనీ పర్స్ }': 'Money_Purse',
            'DAY TRADER తెలుగు': 'Day_Trader_Telugu'
        }
        
        return channel_mapping.get(channel_name, channel_name.replace(' ', '_').replace('తెలుగు', 'Telugu'))
    
    async def _load_or_create_channel_metadata(self, metadata_file: str, channel_name: str) -> Dict[str, Any]:
        """Load existing channel metadata or create new structure"""
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load existing metadata: {e}")
        
        # Create new metadata structure
        return {
            "channel_info": {
                "name": channel_name,
                "processing_date": datetime.now().isoformat(),
                "total_videos": 0,
                "last_updated": datetime.now().isoformat()
            },
            "videos": []
        }
    
    async def _save_channel_metadata(self, metadata_file: str, data: Dict[str, Any]) -> None:
        """Save channel metadata to file"""
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise
    
    async def cleanup(self) -> ToolResult:
        """Clean up resources"""
        try:
            # Clean up old video files if needed
            cleanup_count = await self._cleanup_old_files()
            
            return ToolResult(
                success=True,
                data={"cleaned_files": cleanup_count}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Cleanup failed: {str(e)}"
            )
    
    async def _cleanup_old_files(self, days_to_keep: int = 7) -> int:
        """Clean up video files older than specified days"""
        if not os.path.exists(self.download_path):
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        for filename in os.listdir(self.download_path):
            file_path = os.path.join(self.download_path, filename)
            
            if os.path.isfile(file_path):
                file_date = datetime.fromtimestamp(os.path.getctime(file_path))
                
                if file_date < cutoff_date:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        self.logger.info(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove {filename}: {e}")
        
        return cleaned_count
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters"""
        operation = kwargs.get('operation', 'download_latest')
        
        if operation == 'download_by_date':
            date = kwargs.get('date')
            if not date:
                return False
            # Validate date format
            try:
                if len(date) == 8:  # YYYYMMDD
                    datetime.strptime(date, '%Y%m%d')
                elif len(date) == 10:  # YYYY-MM-DD
                    datetime.strptime(date, '%Y-%m-%d')
                else:
                    return False
            except ValueError:
                return False
                
        elif operation == 'download_url':
            video_url = kwargs.get('video_url')
            if not video_url or 'youtube.com' not in video_url:
                return False
        
        return True
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get information about monitored channels"""
        return {
            "channels": {
                key: {
                    "name": config.name,
                    "url": config.url,
                    "upload_window": config.upload_time_range,
                    "max_duration": config.max_video_duration
                }
                for key, config in self.channels.items()
            },
            "download_path": self.download_path
        }
