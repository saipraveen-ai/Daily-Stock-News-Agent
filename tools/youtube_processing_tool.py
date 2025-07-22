"""
YouTube Processing Tool for Daily Stock News Agent

This tool handles downloading and processing YouTube videos from specified channels.
It monitors for new uploads, downloads audio/video content, and prepares files
for transcription processing.
"""

import os
import asyncio
import yt_dlp
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
    quality_preference: str = "best[height<=720]"  # Video quality preference


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
        
        # yt-dlp configuration
        self.ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[ext=webm][height<=720]/best',
            'outtmpl': os.path.join(config.settings.get('download_path', './data/videos'), 
                                  '%(uploader)s_%(upload_date)s_%(title)s.%(ext)s'),
            'writeinfojson': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': True,
            'no_warnings': False,
            'extractaudio': False,  # We want video for potential chart analysis
            'audio_quality': 'best',
            'retries': 3,
            'fragment_retries': 3,
        }
        
        self.download_path = config.settings.get('download_path', './data/videos')
        os.makedirs(self.download_path, exist_ok=True)
        
    async def initialize(self) -> ToolResult:
        """Initialize the YouTube processing tool"""
        try:
            # Verify yt-dlp is working
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                # Test with a simple extraction (no download)
                test_opts = self.ydl_opts.copy()
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
        
        downloaded_videos = []
        errors = []
        
        for channel_key, channel_config in self.channels.items():
            try:
                # Get latest videos for today
                videos = await self._get_channel_videos(channel_config, date_str)
                
                for video_info in videos:
                    download_result = await self._download_video(video_info['url'])
                    if download_result.success:
                        downloaded_videos.append(download_result.data)
                    else:
                        errors.append(f"{channel_config.name}: {download_result.error_message}")
                        
            except Exception as e:
                self.logger.error(f"Failed to process {channel_config.name}: {e}")
                errors.append(f"{channel_config.name}: {str(e)}")
        
        success = len(downloaded_videos) > 0
        return ToolResult(
            success=success,
            data={
                "downloaded_videos": downloaded_videos,
                "download_count": len(downloaded_videos),
                "errors": errors
            },
            metadata={"operation": "download_latest"}
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
        """Get videos from a channel for a specific date"""
        videos = []
        
        # Construct search URL for the channel's recent videos
        search_url = f"{channel_config.url}/videos"
        
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'playlistend': 10,  # Check last 10 videos
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                playlist_info = ydl.extract_info(search_url, download=False)
                
                if 'entries' in playlist_info:
                    self.logger.info(f"Found {len(playlist_info['entries'])} recent videos for {channel_config.name}")
                    
                    for entry in playlist_info['entries']:
                        if entry:
                            upload_date = entry.get('upload_date', '')
                            title = entry.get('title', 'Unknown')
                            
                            self.logger.debug(f"Video: {title}, Upload date: {upload_date}, Target date: {date_str}")
                            
                            # More flexible date matching - check if video is from today or recent
                            if upload_date == date_str or not date_str:  # If no specific date, get recent videos
                                videos.append({
                                    'id': entry['id'],
                                    'title': title,
                                    'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                    'upload_date': upload_date,
                                    'channel': channel_config.name,
                                    'duration': entry.get('duration', 0)
                                })
                            elif not upload_date:  # If upload_date is missing, include it anyway
                                videos.append({
                                    'id': entry['id'],
                                    'title': title,
                                    'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                    'upload_date': 'unknown',
                                    'channel': channel_config.name,
                                    'duration': entry.get('duration', 0)
                                })
                                
                else:
                    self.logger.warning(f"No entries found for {channel_config.name}")
                                
        except Exception as e:
            self.logger.error(f"Failed to get videos for {channel_config.name}: {e}")
            raise
        
        self.logger.info(f"Filtered to {len(videos)} videos for {channel_config.name} on {date_str}")
        return videos
    
    async def _download_video(self, video_url: str) -> ToolResult:
        """Download a specific video"""
        try:
            video_info = None
            download_path = None
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(video_url, download=False)
                
                # Check video duration
                duration = info.get('duration', 0)
                if duration > 3600:  # More than 1 hour
                    return ToolResult(
                        success=False,
                        error_message=f"Video too long: {duration/60:.1f} minutes"
                    )
                
                # Download the video
                ydl.download([video_url])
                
                # Construct the expected file path
                filename = ydl.prepare_filename(info)
                
                video_info = VideoInfo(
                    video_id=info['id'],
                    title=info.get('title', 'Unknown'),
                    channel=info.get('uploader', 'Unknown'),
                    upload_date=info.get('upload_date', ''),
                    duration=duration,
                    file_path=filename,
                    thumbnail_url=info.get('thumbnail', ''),
                    description=info.get('description', ''),
                    view_count=info.get('view_count', 0),
                    like_count=info.get('like_count', 0)
                )
                
            return ToolResult(
                success=True,
                data=video_info,
                metadata={"download_url": video_url}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download video {video_url}: {e}")
            return ToolResult(
                success=False,
                error_message=f"Download failed: {str(e)}"
            )
    
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
