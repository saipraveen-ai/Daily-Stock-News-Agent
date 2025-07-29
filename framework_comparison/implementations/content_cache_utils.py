"""
Shared Content Utilities for Daily Stock News Agent

This module provides common functions for checking existing content
and avoiding redundant downloads/transcriptions across all framework implementations.
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


def get_target_date(requested_date: Optional[str] = None) -> str:
    """
    Get the appropriate date for downloading videos.
    
    Logic:
    - If specific date requested, use that date
    - If today is weekday (Mon-Fri), use today
    - If today is weekend (Sat-Sun), use previous Friday
    
    Args:
        requested_date: Specific date in YYYY-MM-DD format (optional)
        
    Returns:
        Target date in YYYY-MM-DD format
    """
    if requested_date:
        return requested_date
    
    today = datetime.now()
    weekday = today.weekday()  # 0=Monday, 6=Sunday
    
    if weekday <= 4:  # Monday to Friday (0-4)
        return today.strftime('%Y-%m-%d')
    else:  # Saturday or Sunday (5-6)
        # Calculate days back to Friday
        days_back = weekday - 4  # Sat=1 day back, Sun=2 days back
        target_date = today - timedelta(days=days_back)
        return target_date.strftime('%Y-%m-%d')


class ContentCache:
    """Utility class for managing cached video and transcript content"""
    
    def __init__(self, base_data_path: str = "./data"):
        self.base_path = Path(base_data_path)
        self.videos_path = self.base_path / "videos"
        self.transcripts_path = self.base_path / "transcripts"
    
    def get_date_folder(self, date: Optional[str] = None) -> str:
        """Get the date folder name with intelligent date selection"""
        if date is None:
            date = get_target_date()
        return date
    
    def check_videos_exist(self, channels: List[str], date: Optional[str] = None) -> Tuple[bool, List[Dict]]:
        """
        Check if videos already exist for given channels and date
        With simplified naming: just channel.mp4
        
        Returns:
            Tuple[bool, List[Dict]]: (all_exist, existing_videos_info)
        """
        date_folder = self.get_date_folder(date)
        date_path = self.videos_path / date_folder
        
        if not date_path.exists():
            return False, []
        
        existing_videos = []
        channels_found = set()
        
        # Check for each requested channel with simplified naming
        for channel in channels:
            video_file = date_path / f"{channel}.mp4"
            
            if video_file.exists():
                channels_found.add(channel)
                
                # Try to load metadata from .info.json file
                info_file = date_path / f"{channel}.info.json"
                video_info = {
                    'file_path': str(video_file),
                    'channel': channel,
                    'filename': video_file.name,
                    'size_mb': video_file.stat().st_size / (1024*1024)
                }
                
                if info_file.exists():
                    try:
                        with open(info_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            video_info.update({
                                'video_id': metadata.get('id', ''),
                                'title': metadata.get('title', ''),
                                'duration': metadata.get('duration', 0),
                                'upload_date': metadata.get('upload_date', ''),
                                'view_count': metadata.get('view_count', 0)
                            })
                    except:
                        pass  # Use basic info if metadata load fails
                
                existing_videos.append(video_info)
        
        # Check if all requested channels have videos
        requested_channels = set(channels) if isinstance(channels, list) else set(channels.split(','))
        requested_channels = {ch.strip().lower() for ch in requested_channels}
        
        all_exist = requested_channels.issubset(channels_found)
        
        return all_exist, existing_videos
    
    def check_transcripts_exist(self, video_info: List[Dict], date: Optional[str] = None) -> Tuple[bool, List[Dict]]:
        """
        Check if transcripts already exist for given videos
        With simplified naming: just channel.json
        
        Returns:
            Tuple[bool, List[Dict]]: (all_exist, existing_transcripts_info)
        """
        date_folder = self.get_date_folder(date)
        date_path = self.transcripts_path / date_folder
        
        if not date_path.exists():
            return False, []
        
        existing_transcripts = []
        
        for video in video_info:
            channel = video.get('channel', '')
            
            # Look for simplified transcript file: channel.json
            transcript_file = date_path / f"{channel}.json"
            
            if transcript_file.exists():
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript_data = json.load(f)
                        
                    existing_transcripts.append({
                        'video_info': video,
                        'transcript_file': str(transcript_file),
                        'transcript_data': transcript_data
                    })
                except:
                    # Skip corrupted transcript files
                    pass
            else:
                # Missing transcript for this video
                return False, existing_transcripts
        
        return True, existing_transcripts
    
    def get_cache_summary(self, channels: List[str], date: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of cached content for given channels and date"""
        videos_exist, videos_info = self.check_videos_exist(channels, date)
        transcripts_exist, transcripts_info = self.check_transcripts_exist(videos_info, date)
        
        return {
            'date': self.get_date_folder(date),
            'videos': {
                'exist': videos_exist,
                'count': len(videos_info),
                'info': videos_info
            },
            'transcripts': {
                'exist': transcripts_exist,
                'count': len(transcripts_info),
                'info': transcripts_info
            },
            'ready_for_analysis': videos_exist and transcripts_exist
        }


def check_existing_content(channels: str, date: Optional[str] = None) -> str:
    """
    Convenience function to check existing content and return status message
    
    Args:
        channels: Comma-separated channel names
        date: Date in YYYY-MM-DD format (defaults to intelligent date selection)
    
    Returns:
        Status message about existing content
    """
    if date is None:
        date = get_target_date()
    
    cache = ContentCache()
    channel_list = [ch.strip() for ch in channels.split(',')]
    summary = cache.get_cache_summary(channel_list, date)
    
    date_str = summary['date']
    videos = summary['videos']
    transcripts = summary['transcripts']
    
    if summary['ready_for_analysis']:
        return f"✅ Content ready for {date_str}: {videos['count']} videos, {transcripts['count']} transcripts cached. Skipping download/transcription."
    
    elif videos['exist'] and not transcripts['exist']:
        return f"⚡ Videos cached for {date_str} ({videos['count']} files), but transcripts missing. Skipping download, proceeding to transcription."
    
    elif not videos['exist']:
        return f"📥 No cached content for {date_str}. Proceeding with fresh download and transcription."
    
    else:
        return f"⚠️ Partial content for {date_str}: {videos['count']} videos, {transcripts['count']} transcripts. May need refresh."


# Example usage for framework implementations
def get_or_download_videos(channels: str, date: Optional[str] = None) -> Tuple[bool, List[Dict], str]:
    """
    Check for existing videos or indicate need to download
    
    Returns:
        Tuple[bool, List[Dict], str]: (skip_download, video_info, status_message)
    """
    if date is None:
        date = get_target_date()
    
    cache = ContentCache()
    channel_list = [ch.strip() for ch in channels.split(',')]
    videos_exist, videos_info = cache.check_videos_exist(channel_list, date)
    
    if videos_exist:
        return True, videos_info, f"✅ Using cached videos for {date}: {len(videos_info)} files"
    else:
        return False, [], f"📥 No cached videos found, proceeding with download for {date}"


def get_or_transcribe_videos(video_info: List[Dict], date: Optional[str] = None) -> Tuple[bool, List[Dict], str]:
    """
    Check for existing transcripts or indicate need to transcribe
    
    Returns:
        Tuple[bool, List[Dict], str]: (skip_transcription, transcript_info, status_message)
    """
    if date is None:
        date = get_target_date()
    
    cache = ContentCache()
    transcripts_exist, transcripts_info = cache.check_transcripts_exist(video_info, date)
    
    if transcripts_exist:
        return True, transcripts_info, f"✅ Using cached transcripts for {date}: {len(transcripts_info)} files"
    else:
        return False, [], f"🎤 No cached transcripts found, proceeding with transcription for {date}"
