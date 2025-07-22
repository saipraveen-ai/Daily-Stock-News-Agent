#!/usr/bin/env python3
"""
Test the numerical menu interface
"""

def display_menu():
    """Display the numbered menu options"""
    print("\n" + "="*60)
    print("📈 DAILY STOCK NEWS AGENT - INTERACTIVE MENU")
    print("="*60)
    
    menu_options = {
        1: "Process Today's Videos",
        2: "Process Yesterday's Videos", 
        3: "Process Specific Date",
        4: "Process MoneyPurse Channel Only",
        5: "Process Day Trader Telugu Only",
        6: "Generate Weekly Summary",
        7: "Check Agent Status",
        8: "View Configuration",
        9: "Test Setup",
        0: "Exit"
    }
    
    for num, description in menu_options.items():
        if num == 0:
            print(f"\n{num}. {description}")
        else:
            print(f"{num}. {description}")
    
    print("="*60)

def get_user_choice():
    """Get numerical input from user"""
    try:
        choice = int(input("\nEnter your choice (0-9): ").strip())
        if 0 <= choice <= 9:
            return choice
        else:
            print("❌ Invalid choice. Please enter a number between 0-9.")
            return None
    except ValueError:
        print("❌ Please enter a valid number.")
        return None

def test_setup():
    """Test the current setup"""
    print("\n🧪 Testing setup...")
    print("Testing virtual environment...")
    import sys
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

def main():
    """Main application loop"""
    print("🎯 Welcome to Daily Stock News Agent!")
    print("Processing Telugu YouTube videos from MoneyPurse and Day Trader Telugu")
    
    while True:
        display_menu()
        choice = get_user_choice()
        
        if choice is not None:
            if choice == 0:
                print("\n👋 Thank you for using Daily Stock News Agent!")
                break
            elif choice == 9:
                test_setup()
            elif choice == 8:
                print("\n⚙️  Configuration loaded from: config.yaml")
                print("📋 Configured Channels: 2")
                print("   • MoneyPurse: https://www.youtube.com/@MoneyPurse")
                print("   • Day Trader Telugu: https://www.youtube.com/@daytradertelugu")
                print("⏰ Daily Processing Time: 19:30")
                print("📈 Report Delivery Time: 22:00")
                print("🔄 Auto Processing: Enabled")
            else:
                print(f"\n🚀 Processing option {choice}...")
                print("(This would connect to the full agent system)")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
