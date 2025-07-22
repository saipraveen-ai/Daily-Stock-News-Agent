#!/usr/bin/env python3
"""
Detailed debug script for YouTube processing tool
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/Users/saipraveen/Gen-AI/Daily-Stock-News-Agent')

from tools.youtube_processing_tool import YouTubeProcessingTool
from tools.base_tool import ToolConfig, ToolCategory, ToolPriority

# Configure logging to see debug information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_youtube_processing():
    """Test YouTube processing tool with detailed debugging"""
    
    print("🔧 Testing YouTube Processing Tool - Detailed Analysis")
    print("=" * 60)
    
    # Create tool configuration
    config = ToolConfig(
        name="youtube",
        category=ToolCategory.YOUTUBE,
        priority=ToolPriority.HIGH,
        settings={'download_path': './data/videos'}
    )
    
    # Initialize tool
    tool = YouTubeProcessingTool(config)
    
    print("\n🔌 Initializing tool...")
    init_result = await tool.initialize()
    print(f"   Initialization: {'✅ Success' if init_result.success else '❌ Failed'}")
    if not init_result.success:
        print(f"   Error: {init_result.error_message}")
        return
    
    # Test specific video URLs provided by user
    test_videos = [
        {
            "name": "MoneyPurse",
            "url": "https://www.youtube.com/watch?v=zQSC75GT7Ug"
        },
        {
            "name": "Day Trader Telugu", 
            "url": "https://www.youtube.com/watch?v=vW9Jo0-sKcE"
        }
    ]
    
    print("\n🎥 Testing specific video downloads...")
    for video in test_videos:
        print(f"\n   Testing {video['name']} video: {video['url']}")
        result = await tool.execute(operation='download_url', video_url=video['url'])
        
        print(f"   ✅ Success: {result.success}")
        if result.success:
            video_info = result.data
            print(f"   📹 Title: {video_info.title}")
            print(f"   📅 Upload Date: {video_info.upload_date}")
            print(f"   ⏱️  Duration: {video_info.duration}s ({video_info.duration/60:.1f} minutes)")
            print(f"   📁 File Path: {video_info.file_path}")
            print(f"   👁️  Views: {video_info.view_count}")
        else:
            print(f"   ❌ Error: {result.error_message}")
    
    print("\n🔍 Testing channel video discovery...")
    
    # Test getting recent videos without strict date filtering
    today = datetime.now().strftime('%Y%m%d')
    print(f"   Looking for videos on: {today}")
    
    result = await tool.execute(operation='check_new')
    print(f"   Check new videos result: {'✅ Success' if result.success else '❌ Failed'}")
    if result.success:
        print(f"   📊 Found {result.data['count']} videos for today")
        for video in result.data.get('new_videos', []):
            print(f"      - {video['title']} (Upload: {video['upload_date']}, Duration: {video.get('duration', 0)}s)")
    else:
        print(f"   ❌ Error: {result.error_message}")
    
    # Test with a broader date range - check recent videos
    print("\n📅 Testing with recent videos (no date filter)...")
    result = await tool.execute(operation='check_new', date='')
    print(f"   Recent videos result: {'✅ Success' if result.success else '❌ Failed'}")
    if result.success:
        print(f"   📊 Found {result.data['count']} recent videos")
        for video in result.data.get('new_videos', []):
            print(f"      - {video['title']} (Upload: {video['upload_date']})")
    
    # Test download_latest operation
    print("\n⬬ Testing download_latest operation...")
    result = await tool.execute(operation='download_latest')
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   📊 Downloaded {result.data['download_count']} videos")
        if result.data.get('errors'):
            print(f"   ⚠️  Errors: {len(result.data['errors'])}")
            for error in result.data['errors']:
                print(f"      - {error}")
        for video in result.data.get('downloaded_videos', []):
            print(f"   ✅ Downloaded: {video.title}")
    else:
        print(f"   Error message: {result.error_message}")
    
    # Show channel configuration
    print("\n📋 Channel Configuration:")
    channel_info = tool.get_channel_info()
    for key, channel in channel_info['channels'].items():
        print(f"   {key}: {channel['name']} - {channel['url']}")
    
    print(f"\n📁 Download path: {channel_info['download_path']}")
    
    print("\n✅ YouTube processing test completed")

if __name__ == "__main__":
    asyncio.run(test_youtube_processing())
