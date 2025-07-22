#!/usr/bin/env python3
"""
Debug script for YouTube processing issues
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.youtube_processing_tool import YouTubeProcessingTool

async def test_youtube_processing():
    """Test YouTube processing functionality"""
    
    print("🔍 Testing YouTube Processing Tool")
    print("=" * 50)
    
    try:
        # Initialize the YouTube tool with proper ToolConfig
        from tools.base_tool import ToolConfig, ToolCategory, ToolPriority
        
        config = ToolConfig(
            name="youtube",
            category=ToolCategory.YOUTUBE,
            priority=ToolPriority.HIGH,
            settings={
                'download_path': './data/videos'
            }
        )
        
        tool = YouTubeProcessingTool(config)
        
        print("✅ YouTube tool initialized successfully")
        
        # Test getting channel info
        print("\n📺 Testing channel configuration...")
        for channel_key, channel in tool.channels.items():
            print(f"   • {channel.name}: {channel.url}")
        
        # Test basic yt-dlp functionality
        print("\n🧪 Testing yt-dlp basic functionality...")
        try:
            import yt_dlp
            
            # Test with a simple video (this won't download, just get info)
            test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll as test
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(test_url, download=False)
                print(f"   ✅ yt-dlp working - got info for: {info.get('title', 'Unknown')}")
                
        except Exception as e:
            print(f"   ❌ yt-dlp error: {e}")
            return False
        
        # Test channel video fetching
        print("\n📊 Testing channel video fetching...")
        
        for channel_key, channel_config in tool.channels.items():
            print(f"\n   Testing {channel_config.name}...")
            try:
                # Get today's date
                today = datetime.now().strftime('%Y%m%d')
                
                # Test getting channel videos
                videos = await tool._get_channel_videos(channel_config, today)
                print(f"   ✅ Found {len(videos)} videos for today")
                
                # Show video titles
                for i, video in enumerate(videos[:3]):  # Show first 3
                    print(f"      {i+1}. {video.get('title', 'Unknown title')}")
                
            except Exception as e:
                print(f"   ❌ Error fetching videos from {channel_config.name}: {e}")
                logger.exception(f"Detailed error for {channel_config.name}")
        
        # Test download latest videos (dry run)
        print("\n⬬ Testing download_latest operation...")
        try:
            result = await tool.execute(operation="download_latest")
            
            print(f"   Success: {result.success}")
            if result.success:
                data = result.data or {}
                downloaded = data.get("downloaded_videos", [])
                errors = data.get("errors", [])
                
                print(f"   Downloaded videos: {len(downloaded)}")
                print(f"   Errors: {len(errors)}")
                
                if errors:
                    print("   Error details:")
                    for error in errors:
                        print(f"      • {error}")
                        
            else:
                print(f"   Error message: {result.error_message}")
                
        except Exception as e:
            print(f"   ❌ Error in download_latest: {e}")
            logger.exception("Detailed download_latest error")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize YouTube tool: {e}")
        logger.exception("Detailed initialization error")
        return False

async def main():
    """Main function"""
    success = await test_youtube_processing()
    
    if success:
        print("\n✅ YouTube processing test completed")
    else:
        print("\n❌ YouTube processing test failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
