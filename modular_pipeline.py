"""
Modular Pipeline Manager for Daily Stock News Agent

This module provides a flexible pipeline that can execute individual steps,
track processing state, and resume from where it left off.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from tools import (
    BaseTool, ToolResult, ToolConfig, ToolCategory, ToolPriority,
    YouTubeProcessingTool, SpeechToTextTool, ContentAnalysisTool, ReportGenerationTool
)


class ProcessingStep(Enum):
    """Available processing steps"""
    DOWNLOAD = "download"
    TRANSCRIBE = "transcribe"
    ANALYZE = "analyze"
    GENERATE_REPORT = "generate_report"


@dataclass
class StepResult:
    """Result from a processing step"""
    step: ProcessingStep
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PipelineState:
    """State of the processing pipeline"""
    session_id: str
    date: str
    completed_steps: Set[ProcessingStep]
    step_results: Dict[ProcessingStep, StepResult]
    video_files: List[str]
    transcript_files: List[str]
    analysis_files: List[str]
    report_files: List[str]
    metadata: Dict[str, Any]
    
    def __init__(self, session_id: str, date: str):
        self.session_id = session_id
        self.date = date
        self.completed_steps = set()
        self.step_results = {}
        self.video_files = []
        self.transcript_files = []
        self.analysis_files = []
        self.report_files = []
        self.metadata = {}
    
    def is_step_completed(self, step: ProcessingStep) -> bool:
        """Check if a step is completed"""
        return step in self.completed_steps
    
    def mark_step_completed(self, step: ProcessingStep, result: StepResult):
        """Mark a step as completed"""
        self.completed_steps.add(step)
        self.step_results[step] = result
    
    def get_next_steps(self) -> List[ProcessingStep]:
        """Get the next steps that need to be executed"""
        all_steps = [ProcessingStep.DOWNLOAD, ProcessingStep.TRANSCRIBE, 
                    ProcessingStep.ANALYZE, ProcessingStep.GENERATE_REPORT]
        
        return [step for step in all_steps if not self.is_step_completed(step)]
    
    def can_execute_step(self, step: ProcessingStep) -> bool:
        """Check if a step can be executed based on dependencies"""
        if step == ProcessingStep.DOWNLOAD:
            return True
        elif step == ProcessingStep.TRANSCRIBE:
            return self.is_step_completed(ProcessingStep.DOWNLOAD) and len(self.video_files) > 0
        elif step == ProcessingStep.ANALYZE:
            return self.is_step_completed(ProcessingStep.TRANSCRIBE) and len(self.transcript_files) > 0
        elif step == ProcessingStep.GENERATE_REPORT:
            return self.is_step_completed(ProcessingStep.ANALYZE) and len(self.analysis_files) > 0
        
        return False


class ModularPipelineManager:
    """
    Manages modular execution of the stock news processing pipeline.
    
    Features:
    - Execute individual steps or complete pipeline
    - Track processing state and resume from interruptions
    - Skip already completed steps
    - Validate step dependencies
    - Save/load pipeline state
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools = {}
        self.state_dir = Path("./data/pipeline_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("ModularPipeline")
    
    async def initialize_tools(self) -> bool:
        """Initialize all tools"""
        try:
            # Initialize YouTube Processing Tool
            youtube_config = ToolConfig(
                name="youtube_processor",
                category=ToolCategory.YOUTUBE,
                priority=ToolPriority.HIGH,
                settings=self.config.get("youtube", {})
            )
            self.tools["youtube"] = YouTubeProcessingTool(youtube_config)
            
            # Initialize Speech-to-Text Tool
            stt_config = ToolConfig(
                name="speech_to_text",
                category=ToolCategory.TRANSCRIPTION,
                priority=ToolPriority.CRITICAL,
                settings=self.config.get("speech_to_text", {})
            )
            self.tools["stt"] = SpeechToTextTool(stt_config)
            
            # Initialize Content Analysis Tool
            analysis_config = ToolConfig(
                name="content_analysis",
                category=ToolCategory.ANALYSIS,
                priority=ToolPriority.HIGH,
                settings=self.config.get("content_analysis", {})
            )
            self.tools["analysis"] = ContentAnalysisTool(analysis_config)
            
            # Initialize Report Generation Tool
            report_config = ToolConfig(
                name="report_generation",
                category=ToolCategory.GENERATION,
                priority=ToolPriority.MEDIUM,
                settings=self.config.get("report_generation", {})
            )
            self.tools["report"] = ReportGenerationTool(report_config)
            
            # Initialize all tools
            initialization_tasks = [tool.initialize() for tool in self.tools.values()]
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check initialization results
            failed_tools = []
            for tool_name, result in zip(self.tools.keys(), results):
                if isinstance(result, Exception):
                    failed_tools.append(f"{tool_name}: {result}")
                elif not result.success:
                    failed_tools.append(f"{tool_name}: {result.error_message}")
            
            if failed_tools:
                self.logger.error(f"Failed to initialize tools: {failed_tools}")
                return False
            
            self.logger.info("All tools initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Tool initialization failed: {e}")
            return False
    
    def create_session(self, date: Optional[str] = None) -> PipelineState:
        """Create a new processing session"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        session_id = f"session_{date}_{datetime.now().strftime('%H%M%S')}"
        state = PipelineState(session_id, date)
        
        # Check for existing videos
        video_dir = Path(self.config.get("youtube", {}).get("download_path", "./data/videos"))
        if video_dir.exists():
            state.video_files = [str(f) for f in video_dir.glob("*.mp4")]
            if state.video_files:
                self.logger.info(f"Found {len(state.video_files)} existing videos")
                # Mark download as completed if videos exist
                download_result = StepResult(
                    step=ProcessingStep.DOWNLOAD,
                    success=True,
                    data={"video_count": len(state.video_files), "videos": state.video_files},
                    processing_time=0.0
                )
                state.mark_step_completed(ProcessingStep.DOWNLOAD, download_result)
        
        # Check for existing transcripts
        transcript_dir = Path(self.config.get("speech_to_text", {}).get("output_path", "./data/transcripts"))
        if transcript_dir.exists():
            state.transcript_files = [str(f) for f in transcript_dir.glob("*.json")]
            if state.transcript_files:
                self.logger.info(f"Found {len(state.transcript_files)} existing transcripts")
        
        # Check for existing analyses
        analysis_dir = Path(self.config.get("content_analysis", {}).get("output_path", "./data/analysis"))
        if analysis_dir.exists():
            state.analysis_files = [str(f) for f in analysis_dir.glob("*.json")]
            if state.analysis_files:
                self.logger.info(f"Found {len(state.analysis_files)} existing analyses")
        
        # Check for existing reports
        report_dir = Path(self.config.get("report_generation", {}).get("output_path", "./data/reports"))
        if report_dir.exists():
            state.report_files = [str(f) for f in report_dir.glob("*.md")]
            if state.report_files:
                self.logger.info(f"Found {len(state.report_files)} existing reports")
        
        return state
    
    def save_state(self, state: PipelineState):
        """Save pipeline state to disk"""
        state_file = self.state_dir / f"{state.session_id}.json"
        
        # Convert state to serializable format
        state_dict = {
            "session_id": state.session_id,
            "date": state.date,
            "completed_steps": [step.value for step in state.completed_steps],
            "step_results": {
                step.value: {
                    "step": result.step.value,
                    "success": result.success,
                    "data": result.data,
                    "error_message": result.error_message,
                    "processing_time": result.processing_time,
                    "timestamp": result.timestamp
                }
                for step, result in state.step_results.items()
            },
            "video_files": state.video_files,
            "transcript_files": state.transcript_files,
            "analysis_files": state.analysis_files,
            "report_files": state.report_files,
            "metadata": state.metadata
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        self.logger.debug(f"Saved state to {state_file}")
    
    def load_state(self, session_id: str) -> Optional[PipelineState]:
        """Load pipeline state from disk"""
        state_file = self.state_dir / f"{session_id}.json"
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            
            state = PipelineState(state_dict["session_id"], state_dict["date"])
            state.completed_steps = {ProcessingStep(step) for step in state_dict["completed_steps"]}
            
            # Reconstruct step results
            for step_value, result_dict in state_dict["step_results"].items():
                step = ProcessingStep(step_value)
                result = StepResult(
                    step=ProcessingStep(result_dict["step"]),
                    success=result_dict["success"],
                    data=result_dict["data"],
                    error_message=result_dict.get("error_message"),
                    processing_time=result_dict["processing_time"],
                    timestamp=result_dict["timestamp"]
                )
                state.step_results[step] = result
            
            state.video_files = state_dict["video_files"]
            state.transcript_files = state_dict["transcript_files"]
            state.analysis_files = state_dict["analysis_files"]
            state.report_files = state_dict["report_files"]
            state.metadata = state_dict["metadata"]
            
            self.logger.info(f"Loaded state for session {session_id}")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to load state {session_id}: {e}")
            return None
    
    async def execute_step(self, state: PipelineState, step: ProcessingStep) -> StepResult:
        """Execute a single processing step"""
        start_time = datetime.now()
        self.logger.info(f"Executing step: {step.value}")
        
        try:
            if step == ProcessingStep.DOWNLOAD:
                return await self._execute_download(state)
            elif step == ProcessingStep.TRANSCRIBE:
                return await self._execute_transcribe(state)
            elif step == ProcessingStep.ANALYZE:
                return await self._execute_analyze(state)
            elif step == ProcessingStep.GENERATE_REPORT:
                return await self._execute_generate_report(state)
            else:
                return StepResult(
                    step=step,
                    success=False,
                    data={},
                    error_message=f"Unknown step: {step.value}"
                )
                
        except Exception as e:
            self.logger.error(f"Step {step.value} failed: {e}")
            return StepResult(
                step=step,
                success=False,
                data={},
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _execute_download(self, state: PipelineState) -> StepResult:
        """Execute video download step"""
        start_time = datetime.now()
        
        result = await self.tools["youtube"].execute(operation='download_latest', date=state.date)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result.success:
            # Update video files list
            video_dir = Path(self.config.get("youtube", {}).get("download_path", "./data/videos"))
            state.video_files = [str(f) for f in video_dir.glob("*.mp4")]
            
            return StepResult(
                step=ProcessingStep.DOWNLOAD,
                success=True,
                data={
                    "video_count": len(state.video_files),
                    "videos": state.video_files,
                    "download_details": result.data
                },
                processing_time=processing_time
            )
        else:
            return StepResult(
                step=ProcessingStep.DOWNLOAD,
                success=False,
                data={},
                error_message=result.error_message,
                processing_time=processing_time
            )
    
    async def _execute_transcribe(self, state: PipelineState) -> StepResult:
        """Execute transcription step"""
        start_time = datetime.now()
        
        if not state.video_files:
            return StepResult(
                step=ProcessingStep.TRANSCRIBE,
                success=False,
                data={},
                error_message="No video files available for transcription"
            )
        
        transcriptions = []
        errors = []
        
        for video_file in state.video_files:
            try:
                result = await self.tools["stt"].execute(audio_file=video_file)
                
                if result.success:
                    transcriptions.append(result.data)
                    state.transcript_files.append(result.data.get("output_file", ""))
                else:
                    errors.append(f"{video_file}: {result.error_message}")
                    
            except Exception as e:
                errors.append(f"{video_file}: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        success = len(transcriptions) > 0
        return StepResult(
            step=ProcessingStep.TRANSCRIBE,
            success=success,
            data={
                "transcription_count": len(transcriptions),
                "transcriptions": transcriptions,
                "errors": errors
            },
            error_message="Some transcriptions failed" if errors and success else None,
            processing_time=processing_time
        )
    
    async def _execute_analyze(self, state: PipelineState) -> StepResult:
        """Execute content analysis step"""
        start_time = datetime.now()
        
        if not state.transcript_files:
            return StepResult(
                step=ProcessingStep.ANALYZE,
                success=False,
                data={},
                error_message="No transcript files available for analysis"
            )
        
        analyses = []
        errors = []
        
        for transcript_file in state.transcript_files:
            try:
                result = await self.tools["analysis"].execute(transcript_file=transcript_file)
                
                if result.success:
                    analyses.append(result.data)
                    state.analysis_files.append(result.data.get("output_file", ""))
                else:
                    errors.append(f"{transcript_file}: {result.error_message}")
                    
            except Exception as e:
                errors.append(f"{transcript_file}: {str(e)}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        success = len(analyses) > 0
        return StepResult(
            step=ProcessingStep.ANALYZE,
            success=success,
            data={
                "analysis_count": len(analyses),
                "analyses": analyses,
                "errors": errors
            },
            error_message="Some analyses failed" if errors and success else None,
            processing_time=processing_time
        )
    
    async def _execute_generate_report(self, state: PipelineState) -> StepResult:
        """Execute report generation step"""
        start_time = datetime.now()
        
        if not state.analysis_files:
            return StepResult(
                step=ProcessingStep.GENERATE_REPORT,
                success=False,
                data={},
                error_message="No analysis files available for report generation"
            )
        
        try:
            result = await self.tools["report"].execute(
                analysis_files=state.analysis_files,
                output_formats=self.config.get("report_generation", {}).get("default_formats", ["markdown"])
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if result.success:
                # Update report files list
                report_dir = Path(self.config.get("report_generation", {}).get("output_path", "./data/reports"))
                state.report_files = [str(f) for f in report_dir.glob("*.md")]
                
                return StepResult(
                    step=ProcessingStep.GENERATE_REPORT,
                    success=True,
                    data={
                        "report_count": len(state.report_files),
                        "reports": state.report_files,
                        "generation_details": result.data
                    },
                    processing_time=processing_time
                )
            else:
                return StepResult(
                    step=ProcessingStep.GENERATE_REPORT,
                    success=False,
                    data={},
                    error_message=result.error_message,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return StepResult(
                step=ProcessingStep.GENERATE_REPORT,
                success=False,
                data={},
                error_message=str(e),
                processing_time=processing_time
            )
    
    def get_pipeline_status(self, state: PipelineState) -> Dict[str, Any]:
        """Get current pipeline status"""
        all_steps = [ProcessingStep.DOWNLOAD, ProcessingStep.TRANSCRIBE, 
                    ProcessingStep.ANALYZE, ProcessingStep.GENERATE_REPORT]
        
        step_status = {}
        for step in all_steps:
            if state.is_step_completed(step):
                result = state.step_results[step]
                step_status[step.value] = {
                    "status": "completed",
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "timestamp": result.timestamp,
                    "data_summary": self._summarize_step_data(result)
                }
            elif state.can_execute_step(step):
                step_status[step.value] = {
                    "status": "ready",
                    "can_execute": True
                }
            else:
                step_status[step.value] = {
                    "status": "waiting",
                    "can_execute": False,
                    "dependencies": self._get_step_dependencies(step)
                }
        
        return {
            "session_id": state.session_id,
            "date": state.date,
            "overall_progress": f"{len(state.completed_steps)}/{len(all_steps)}",
            "next_steps": [step.value for step in state.get_next_steps()],
            "steps": step_status,
            "file_counts": {
                "videos": len(state.video_files),
                "transcripts": len(state.transcript_files),
                "analyses": len(state.analysis_files),
                "reports": len(state.report_files)
            }
        }
    
    def _summarize_step_data(self, result: StepResult) -> Dict[str, Any]:
        """Create a summary of step data"""
        data = result.data
        
        if result.step == ProcessingStep.DOWNLOAD:
            return {"video_count": data.get("video_count", 0)}
        elif result.step == ProcessingStep.TRANSCRIBE:
            return {
                "transcription_count": data.get("transcription_count", 0),
                "error_count": len(data.get("errors", []))
            }
        elif result.step == ProcessingStep.ANALYZE:
            return {
                "analysis_count": data.get("analysis_count", 0),
                "error_count": len(data.get("errors", []))
            }
        elif result.step == ProcessingStep.GENERATE_REPORT:
            return {"report_count": data.get("report_count", 0)}
        
        return {}
    
    def _get_step_dependencies(self, step: ProcessingStep) -> List[str]:
        """Get dependencies for a step"""
        if step == ProcessingStep.DOWNLOAD:
            return []
        elif step == ProcessingStep.TRANSCRIBE:
            return ["download"]
        elif step == ProcessingStep.ANALYZE:
            return ["download", "transcribe"]
        elif step == ProcessingStep.GENERATE_REPORT:
            return ["download", "transcribe", "analyze"]
        
        return []
