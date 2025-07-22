#!/usr/bin/env python3
"""
Modular Interactive Agent for Daily Stock News Agent

This interactive agent uses the modular pipeline to provide granular control
over the processing steps, allowing users to execute individual steps or
resume from where they left off.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, '/Users/saipraveen/Gen-AI/Daily-Stock-News-Agent')

from modular_pipeline import ModularPipelineManager, ProcessingStep


class ModularInteractiveAgent:
    """Interactive agent with modular pipeline support"""
    
    def __init__(self):
        # Load configuration (simplified for now)
        self.config = {
            "youtube": {
                "download_path": "./data/videos"
            },
            "speech_to_text": {
                "output_path": "./data/transcripts",
                "whisper_model": "base",
                "enable_translation": True
            },
            "content_analysis": {
                "output_path": "./data/analysis",
                "ai_provider": "local"
            },
            "report_generation": {
                "output_path": "./data/reports",
                "default_formats": ["markdown", "json"]
            }
        }
        
        self.pipeline = ModularPipelineManager(self.config)
        self.current_session = None
    
    async def initialize(self):
        """Initialize the pipeline"""
        print("🔧 Initializing Modular Pipeline...")
        success = await self.pipeline.initialize_tools()
        if success:
            print("✅ Pipeline initialized successfully")
        else:
            print("❌ Pipeline initialization failed")
        return success
    
    def display_main_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🤖 DAILY STOCK NEWS AGENT - MODULAR PIPELINE")
        print("="*60)
        
        if self.current_session:
            status = self.pipeline.get_pipeline_status(self.current_session)
            print(f"📋 Current Session: {self.current_session.session_id}")
            print(f"📅 Date: {self.current_session.date}")
            print(f"📊 Progress: {status['overall_progress']}")
            
            if status['next_steps']:
                print(f"🔄 Next Steps: {', '.join(status['next_steps'])}")
            else:
                print("✅ All steps completed!")
        else:
            print("📋 No active session")
        
        print("\n📋 SESSION MANAGEMENT:")
        print("   1. Create New Session")
        print("   2. Load Existing Session")
        print("   3. Show Session Status")
        
        print("\n🔧 INDIVIDUAL STEPS:")
        print("   4. Download Videos")
        print("   5. Transcribe Videos")
        print("   6. Analyze Content")
        print("   7. Generate Reports")
        
        print("\n⚡ PIPELINE EXECUTION:")
        print("   8. Execute Next Step")
        print("   9. Execute All Remaining Steps")
        print("   10. Execute Complete Pipeline")
        
        print("\n📊 UTILITIES:")
        print("   11. View File Counts")
        print("   12. Clean Old Files")
        print("   0. Exit")
        
        print("-" * 60)
    
    async def handle_user_choice(self, choice: str):
        """Handle user menu choice"""
        try:
            if choice == '1':
                await self.create_new_session()
            elif choice == '2':
                await self.load_existing_session()
            elif choice == '3':
                await self.show_session_status()
            elif choice == '4':
                await self.execute_single_step(ProcessingStep.DOWNLOAD)
            elif choice == '5':
                await self.execute_single_step(ProcessingStep.TRANSCRIBE)
            elif choice == '6':
                await self.execute_single_step(ProcessingStep.ANALYZE)
            elif choice == '7':
                await self.execute_single_step(ProcessingStep.GENERATE_REPORT)
            elif choice == '8':
                await self.execute_next_step()
            elif choice == '9':
                await self.execute_remaining_steps()
            elif choice == '10':
                await self.execute_complete_pipeline()
            elif choice == '11':
                await self.view_file_counts()
            elif choice == '12':
                await self.clean_old_files()
            elif choice == '0':
                print("👋 Goodbye!")
                return False
            else:
                print("❌ Invalid choice. Please try again.")
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        return True
    
    async def create_new_session(self):
        """Create a new processing session"""
        print("\n📋 Creating New Session...")
        
        # Ask for date
        date_input = input("Enter date (YYYY-MM-DD) or press Enter for today: ").strip()
        if not date_input:
            date_input = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Validate date format
            datetime.strptime(date_input, '%Y-%m-%d')
            
            self.current_session = self.pipeline.create_session(date_input)
            self.pipeline.save_state(self.current_session)
            
            print(f"✅ Created session: {self.current_session.session_id}")
            print(f"📅 Date: {self.current_session.date}")
            
            # Show initial status
            await self.show_session_status()
            
        except ValueError:
            print("❌ Invalid date format. Please use YYYY-MM-DD.")
    
    async def load_existing_session(self):
        """Load an existing session"""
        print("\n📋 Loading Existing Session...")
        
        # List available sessions
        state_files = list(self.pipeline.state_dir.glob("session_*.json"))
        
        if not state_files:
            print("❌ No existing sessions found.")
            return
        
        print("Available sessions:")
        for i, state_file in enumerate(state_files, 1):
            session_id = state_file.stem
            print(f"   {i}. {session_id}")
        
        try:
            choice = input("Enter session number: ").strip()
            session_index = int(choice) - 1
            
            if 0 <= session_index < len(state_files):
                session_id = state_files[session_index].stem
                self.current_session = self.pipeline.load_state(session_id)
                
                if self.current_session:
                    print(f"✅ Loaded session: {session_id}")
                    await self.show_session_status()
                else:
                    print("❌ Failed to load session.")
            else:
                print("❌ Invalid session number.")
                
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    async def show_session_status(self):
        """Show current session status"""
        if not self.current_session:
            print("❌ No active session. Please create or load a session first.")
            return
        
        print(f"\n📊 SESSION STATUS: {self.current_session.session_id}")
        print("=" * 60)
        
        status = self.pipeline.get_pipeline_status(self.current_session)
        
        print(f"📅 Date: {status['date']}")
        print(f"📊 Overall Progress: {status['overall_progress']}")
        
        print(f"\n📁 File Counts:")
        for file_type, count in status['file_counts'].items():
            print(f"   {file_type.capitalize()}: {count}")
        
        print(f"\n🔄 Step Status:")
        for step_name, step_info in status['steps'].items():
            status_icon = {
                'completed': '✅',
                'ready': '🔄',
                'waiting': '⏳'
            }.get(step_info['status'], '❓')
            
            print(f"   {status_icon} {step_name.upper()}: {step_info['status']}")
            
            if step_info['status'] == 'completed':
                print(f"      └─ Success: {step_info['success']}")
                print(f"      └─ Time: {step_info['processing_time']:.2f}s")
                if 'data_summary' in step_info:
                    for key, value in step_info['data_summary'].items():
                        print(f"      └─ {key}: {value}")
            elif step_info['status'] == 'waiting':
                deps = ', '.join(step_info.get('dependencies', []))
                print(f"      └─ Waiting for: {deps}")
        
        if status['next_steps']:
            print(f"\n🔄 Next Steps: {', '.join(status['next_steps'])}")
        else:
            print(f"\n🎉 All steps completed!")
    
    async def execute_single_step(self, step: ProcessingStep):
        """Execute a single processing step"""
        if not self.current_session:
            print("❌ No active session. Please create or load a session first.")
            return
        
        print(f"\n🔄 Executing Step: {step.value.upper()}")
        print("-" * 40)
        
        # Check if step can be executed
        if not self.current_session.can_execute_step(step):
            print(f"❌ Cannot execute {step.value}. Dependencies not met.")
            return
        
        # Check if step is already completed
        if self.current_session.is_step_completed(step):
            print(f"✅ Step {step.value} is already completed.")
            choice = input("Do you want to re-execute it? (y/N): ").strip().lower()
            if choice != 'y':
                return
        
        try:
            result = await self.pipeline.execute_step(self.current_session, step)
            
            if result.success:
                print(f"✅ Step {step.value} completed successfully!")
                print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
                
                # Show result summary
                if step == ProcessingStep.DOWNLOAD:
                    print(f"📹 Downloaded {result.data.get('video_count', 0)} videos")
                elif step == ProcessingStep.TRANSCRIBE:
                    print(f"📝 Transcribed {result.data.get('transcription_count', 0)} videos")
                    if result.data.get('errors'):
                        print(f"⚠️  {len(result.data['errors'])} errors occurred")
                elif step == ProcessingStep.ANALYZE:
                    print(f"🧠 Analyzed {result.data.get('analysis_count', 0)} transcripts")
                    if result.data.get('errors'):
                        print(f"⚠️  {len(result.data['errors'])} errors occurred")
                elif step == ProcessingStep.GENERATE_REPORT:
                    print(f"📄 Generated {result.data.get('report_count', 0)} reports")
                
                self.current_session.mark_step_completed(step, result)
            else:
                print(f"❌ Step {step.value} failed!")
                print(f"Error: {result.error_message}")
            
            # Save state
            self.pipeline.save_state(self.current_session)
            
        except Exception as e:
            print(f"❌ Unexpected error during {step.value}: {e}")
    
    async def execute_next_step(self):
        """Execute the next available step"""
        if not self.current_session:
            print("❌ No active session. Please create or load a session first.")
            return
        
        next_steps = self.current_session.get_next_steps()
        
        if not next_steps:
            print("🎉 All steps are already completed!")
            return
        
        next_step = next_steps[0]
        await self.execute_single_step(next_step)
    
    async def execute_remaining_steps(self):
        """Execute all remaining steps"""
        if not self.current_session:
            print("❌ No active session. Please create or load a session first.")
            return
        
        next_steps = self.current_session.get_next_steps()
        
        if not next_steps:
            print("🎉 All steps are already completed!")
            return
        
        print(f"\n🔄 Executing {len(next_steps)} remaining steps...")
        print(f"Steps: {', '.join([step.value for step in next_steps])}")
        
        for step in next_steps:
            await self.execute_single_step(step)
            
            # Check if the step failed
            if not self.current_session.is_step_completed(step):
                print(f"❌ Stopping execution due to failed step: {step.value}")
                break
    
    async def execute_complete_pipeline(self):
        """Execute the complete pipeline from start to finish"""
        if not self.current_session:
            print("❌ No active session. Please create or load a session first.")
            return
        
        print("\n🚀 Executing Complete Pipeline...")
        print("=" * 50)
        
        all_steps = [ProcessingStep.DOWNLOAD, ProcessingStep.TRANSCRIBE, 
                    ProcessingStep.ANALYZE, ProcessingStep.GENERATE_REPORT]
        
        start_time = datetime.now()
        
        for step in all_steps:
            if not self.current_session.is_step_completed(step):
                await self.execute_single_step(step)
                
                # Check if the step failed
                if not self.current_session.is_step_completed(step):
                    print(f"❌ Pipeline stopped due to failed step: {step.value}")
                    return
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n🎉 Complete pipeline execution finished!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        
        # Show final status
        await self.show_session_status()
    
    async def view_file_counts(self):
        """View file counts in all directories"""
        print("\n📁 FILE COUNTS")
        print("=" * 30)
        
        directories = [
            ("Videos", "./data/videos", "*.mp4"),
            ("Transcripts", "./data/transcripts", "*.json"),
            ("Analyses", "./data/analysis", "*.json"),
            ("Reports", "./data/reports", "*.md")
        ]
        
        for name, path, pattern in directories:
            try:
                from pathlib import Path
                dir_path = Path(path)
                if dir_path.exists():
                    files = list(dir_path.glob(pattern))
                    print(f"{name}: {len(files)} files")
                    
                    if files:
                        # Show most recent files
                        recent_files = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)[:3]
                        for file in recent_files:
                            mtime = datetime.fromtimestamp(file.stat().st_mtime)
                            print(f"  └─ {file.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
                else:
                    print(f"{name}: Directory doesn't exist")
            except Exception as e:
                print(f"{name}: Error - {e}")
    
    async def clean_old_files(self):
        """Clean old files (placeholder)"""
        print("\n🧹 File cleanup feature not implemented yet.")
        print("You can manually clean files from the data directories.")
    
    async def run(self):
        """Run the interactive agent"""
        print("🤖 Starting Modular Interactive Agent...")
        
        # Initialize
        if not await self.initialize():
            print("❌ Failed to initialize. Exiting.")
            return
        
        # Main loop
        while True:
            self.display_main_menu()
            choice = input("\nEnter your choice: ").strip()
            
            if not await self.handle_user_choice(choice):
                break
            
            input("\nPress Enter to continue...")


async def main():
    """Main entry point"""
    agent = ModularInteractiveAgent()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
