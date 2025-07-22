#!/usr/bin/env python3
"""
Quick Setup Test for Daily Stock News Agent

This script tests the basic installation and configuration
to ensure everything is working correctly.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Setup basic logging for test
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SetupTest")

def test_python_version():
    """Test Python version compatibility"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    logger.info("✅ Python version OK")
    return True

def test_dependencies():
    """Test required dependencies"""
    required_modules = [
        'asyncio', 'logging', 'json', 'os', 're', 'time', 'datetime', 'dataclasses', 'typing', 'enum'
    ]
    
    optional_modules = {
        'yt_dlp': 'YouTube processing',
        'whisper': 'OpenAI Whisper (speech-to-text)',
        'yaml': 'Configuration files',
        'markdown': 'Report generation'
    }
    
    # Test required modules
    missing_required = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(module)
    
    if missing_required:
        logger.error(f"❌ Missing required modules: {missing_required}")
        return False
    
    logger.info("✅ Required Python modules OK")
    
    # Test optional modules
    available_optional = []
    missing_optional = []
    
    for module, description in optional_modules.items():
        try:
            __import__(module)
            available_optional.append(f"{module} ({description})")
        except ImportError:
            missing_optional.append(f"{module} ({description})")
    
    if available_optional:
        logger.info(f"✅ Available optional modules: {', '.join(available_optional)}")
    
    if missing_optional:
        logger.warning(f"⚠️  Missing optional modules: {', '.join(missing_optional)}")
        logger.info("💡 Install with: pip install yt-dlp openai-whisper pyyaml markdown")
    
    return True

def test_directories():
    """Test required directories exist"""
    required_dirs = [
        'data', 'data/videos', 'data/transcripts', 'data/reports', 'data/blogs',
        'logs', 'tools'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.error(f"❌ Missing directories: {missing_dirs}")
        logger.info("💡 Create with: mkdir -p " + " ".join(missing_dirs))
        return False
    
    logger.info("✅ Required directories exist")
    return True

def test_configuration():
    """Test configuration files"""
    config_files = {
        '.env.example': 'Environment template',
        'config.yaml': 'Main configuration', 
        'requirements.txt': 'Dependencies list'
    }
    
    missing_configs = []
    existing_configs = []
    
    for config_file, description in config_files.items():
        if os.path.exists(config_file):
            existing_configs.append(f"{config_file} ({description})")
        else:
            missing_configs.append(f"{config_file} ({description})")
    
    if existing_configs:
        logger.info(f"✅ Configuration files: {', '.join(existing_configs)}")
    
    if missing_configs:
        logger.warning(f"⚠️  Missing configuration files: {', '.join(missing_configs)}")
    
    # Check if .env exists
    if os.path.exists('.env'):
        logger.info("✅ Environment file (.env) exists")
    else:
        logger.warning("⚠️  Environment file (.env) not found")
        logger.info("💡 Copy from template: cp .env.example .env")
    
    return True

def test_tools_import():
    """Test importing custom tools"""
    try:
        from tools.base_tool import BaseTool, ToolResult, ToolConfig
        logger.info("✅ Base tool classes import OK")
        
        from tools.youtube_processing_tool import YouTubeProcessingTool
        logger.info("✅ YouTube processing tool import OK")
        
        from tools.speech_to_text_tool import SpeechToTextTool
        logger.info("✅ Speech-to-text tool import OK")
        
        from tools.content_analysis_tool import ContentAnalysisTool
        logger.info("✅ Content analysis tool import OK")
        
        from tools.report_generation_tool import ReportGenerationTool
        logger.info("✅ Report generation tool import OK")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Tool import failed: {e}")
        return False

async def test_agent_initialization():
    """Test agent initialization"""
    try:
        from autonomous_stock_news_agent import AutonomousStockNewsAgent
        
        logger.info("Testing agent initialization...")
        agent = AutonomousStockNewsAgent()
        
        # Test configuration loading
        if agent.config:
            logger.info("✅ Configuration loaded")
        else:
            logger.warning("⚠️  Configuration not loaded properly")
        
        # Test tool initialization (this might fail if dependencies are missing)
        try:
            success = await agent.initialize()
            if success:
                logger.info("✅ Agent initialization successful")
                await agent.cleanup()
                return True
            else:
                logger.warning("⚠️  Agent initialization failed (likely due to missing dependencies)")
                return False
        except Exception as e:
            logger.warning(f"⚠️  Agent initialization error: {e}")
            logger.info("💡 This is expected if optional dependencies are missing")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Agent import failed: {e}")
        return False

def test_write_permissions():
    """Test write permissions for data directories"""
    test_dirs = ['data/videos', 'data/transcripts', 'data/reports', 'logs']
    
    all_writable = True
    for test_dir in test_dirs:
        try:
            test_file = os.path.join(test_dir, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            logger.error(f"❌ Cannot write to {test_dir}: {e}")
            all_writable = False
    
    if all_writable:
        logger.info("✅ Write permissions OK")
    
    return all_writable

async def main():
    """Run all setup tests"""
    logger.info("🚀 Starting Daily Stock News Agent Setup Test")
    logger.info("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies), 
        ("Directories", test_directories),
        ("Configuration", test_configuration),
        ("Tools Import", test_tools_import),
        ("Write Permissions", test_write_permissions),
        ("Agent Initialization", test_agent_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your setup is ready.")
        logger.info("\n🚀 Next steps:")
        logger.info("1. Copy .env.example to .env and configure your API keys")
        logger.info("2. Run: python autonomous_stock_news_agent.py --request 'Test the system'")
        logger.info("3. Check the generated reports in data/reports/")
    elif passed >= total - 2:
        logger.info("✅ Setup mostly complete! Minor issues detected.")
        logger.info("💡 Check warnings above and install missing optional dependencies")
    else:
        logger.error("❌ Setup incomplete. Please fix the errors above.")
        logger.info("💡 Check SETUP_GUIDE.md for detailed instructions")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
