#!/usr/bin/env python3
"""
Interactive Daily Stock News Agent with Numerical Menu Options
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from autonomous_stock_news_agent import AutonomousStockNewsAgent

class InteractiveStockNewsAgent:
    def __init__(self):
        self.agent = None
        self.menu_options = {
            1: ("Process Today's Videos", self.process_today),
            2: ("Process Yesterday's Videos", self.process_yesterday),
            3: ("Process Specific Date", self.process_specific_date),
            4: ("Process MoneyPurse Channel Only", self.process_moneypurse),
            5: ("Process Day Trader Telugu Only", self.process_daytrader),
            6: ("Generate Weekly Summary", self.generate_weekly_summary),
            7: ("Check Agent Status", self.check_status),
            8: ("View Configuration", self.view_config),
            9: ("Test Setup", self.test_setup),
            0: ("Exit", self.exit_app)
        }
    
    async def initialize_agent(self):
        """Initialize the agent with virtual environment"""
        print("🚀 Initializing Daily Stock News Agent...")
        try:
            config_path = os.path.join(os.getcwd(), 'config.yaml')
            self.agent = AutonomousStockNewsAgent(config_path=config_path)
            
            print("🔧 Setting up tools...")
            success = await self.agent.initialize()
            
            if success:
                print("✅ Agent initialized successfully!")
            else:
                print("⚠️  Agent initialized with limited functionality (fallback mode)")
            
            return True
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            return False
    
    def display_menu(self):
        """Display the numbered menu options"""
        print("\n" + "="*60)
        print("📈 DAILY STOCK NEWS AGENT - INTERACTIVE MENU")
        print("="*60)
        
        for num, (description, _) in self.menu_options.items():
            if num == 0:
                print(f"\n{num}. {description}")
            else:
                print(f"{num}. {description}")
        
        print("="*60)
    
    def get_user_choice(self):
        """Get numerical input from user"""
        try:
            choice = int(input("\nEnter your choice (0-9): ").strip())
            if choice in self.menu_options:
                return choice
            else:
                print("❌ Invalid choice. Please enter a number between 0-9.")
                return None
        except ValueError:
            print("❌ Please enter a valid number.")
            return None
    
    async def process_today(self):
        """Process today's videos"""
        print("\n📅 Processing today's videos...")
        result = await self.agent.process_request_from_natural_language(
            "Get me today's stock market updates from both Telugu channels"
        )
        self.display_result(result)
    
    async def process_yesterday(self):
        """Process yesterday's videos"""
        print("\n📅 Processing yesterday's videos...")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = await self.agent.process_request_from_natural_language(
            f"Analyze videos from {yesterday} from both channels"
        )
        self.display_result(result)
    
    async def process_specific_date(self):
        """Process videos from a specific date"""
        date_str = input("Enter date (YYYY-MM-DD): ").strip()
        try:
            # Validate date format
            datetime.strptime(date_str, "%Y-%m-%d")
            print(f"\n📅 Processing videos from {date_str}...")
            result = await self.agent.process_request_from_natural_language(
                f"Analyze stock market videos from {date_str} from both channels"
            )
            self.display_result(result)
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD")
    
    async def process_moneypurse(self):
        """Process MoneyPurse channel only"""
        print("\n📺 Processing MoneyPurse channel...")
        result = await self.agent.process_request_from_natural_language(
            "Get me the latest videos from MoneyPurse channel only"
        )
        self.display_result(result)
    
    async def process_daytrader(self):
        """Process Day Trader Telugu channel only"""
        print("\n📺 Processing Day Trader Telugu channel...")
        result = await self.agent.process_request_from_natural_language(
            "Get me the latest videos from Day Trader Telugu channel only"
        )
        self.display_result(result)
    
    async def generate_weekly_summary(self):
        """Generate weekly summary report"""
        print("\n📊 Generating weekly summary...")
        result = await self.agent.process_request_from_natural_language(
            "Generate a comprehensive weekly summary of all stock market discussions from both channels"
        )
        self.display_result(result)
    
    async def check_status(self):
        """Check agent status"""
        print("\n🔍 Checking agent status...")
        try:
            status = await self.agent.get_processing_status()
            print(f"Agent Status: {status}")
        except Exception as e:
            print(f"❌ Error checking status: {e}")
    
    async def view_config(self):
        """View current configuration"""
        print("\n⚙️  Current Configuration:")
        print("-" * 40)
        if self.agent and self.agent.config:
            print(f"📋 Configured Channels: {len(self.agent.config.get('channels', {}))}")
            for name, channel in self.agent.config.get('channels', {}).items():
                print(f"   • {channel['name']}: {channel['url']}")
            
            automation = self.agent.config.get('automation', {})
            print(f"⏰ Daily Processing Time: {automation.get('daily_processing_time', 'Not set')}")
            print(f"📈 Report Delivery Time: {automation.get('report_delivery_time', 'Not set')}")
            print(f"🔄 Auto Processing: {'Enabled' if automation.get('enabled', False) else 'Disabled'}")
        else:
            print("❌ No configuration loaded")
    
    async def test_setup(self):
        """Test the current setup"""
        print("\n🧪 Testing setup...")
        print("Testing virtual environment...")
        print(f"✅ Python executable: {sys.executable}")
        print(f"✅ Virtual environment: {'venv' in sys.executable}")
        
        try:
            import yt_dlp
            print("✅ yt-dlp: Available")
        except ImportError:
            print("❌ yt-dlp: Not available")
        
        try:
            import whisper
            print("✅ whisper: Available")
        except ImportError:
            print("❌ whisper: Not available")
        
        try:
            import yaml
            print("✅ yaml: Available")
        except ImportError:
            print("❌ yaml: Not available")
        
        try:
            import markdown
            print("✅ markdown: Available")
        except ImportError:
            print("❌ markdown: Not available")
        
        print("✅ Setup test complete!")
    
    def display_result(self, result):
        """Display processing result"""
        print("\n📋 PROCESSING RESULT:")
        print("-" * 30)
        print(f"✅ Success: {result.success}")
        print(f"📹 Videos Processed: {len(result.processed_videos)}")
        print(f"📝 Transcriptions: {len(result.transcriptions)}")
        print(f"📊 Analyses: {len(result.analyses)}")
        print(f"📄 Reports Generated: {len(result.reports)}")
        
        if result.errors:
            print(f"⚠️  Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"   • {error}")
        
        if result.reports:
            print("\n📄 Generated Reports:")
            for report in result.reports:
                print(f"   • {report}")
        
        print(f"\n⏱️  Processing Time: {result.processing_time:.2f} seconds")
    
    async def exit_app(self):
        """Exit the application"""
        print("\n👋 Thank you for using Daily Stock News Agent!")
        if self.agent:
            print("🧹 Cleaning up...")
            await self.agent.cleanup()
        sys.exit(0)
    
    async def run(self):
        """Main application loop"""
        print("🎯 Welcome to Daily Stock News Agent!")
        print("Processing Telugu YouTube videos from MoneyPurse and Day Trader Telugu")
        
        # Initialize agent
        if not await self.initialize_agent():
            print("❌ Failed to initialize. Exiting...")
            return
        
        # Main loop
        while True:
            self.display_menu()
            choice = self.get_user_choice()
            
            if choice is not None:
                try:
                    _, func = self.menu_options[choice]
                    await func()
                except Exception as e:
                    print(f"❌ Error executing option: {e}")
            
            input("\nPress Enter to continue...")

async def main():
    """Entry point"""
    interactive_agent = InteractiveStockNewsAgent()
    await interactive_agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
