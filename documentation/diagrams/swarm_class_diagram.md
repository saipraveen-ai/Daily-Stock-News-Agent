# Swarm Agent Architecture - Class Diagram

```mermaid
classDiagram
    class SwarmStockNewsSystem {
        +client: Swarm
        +context: Dict
        +video_agent: Agent
        +transcription_agent: Agent
        +analysis_agent: Agent
        +report_agent: Agent
        +coordinator_agent: Agent
        +__init__(api_key: str)
        +_create_video_agent() Agent
        +_create_transcription_agent() Agent
        +_create_analysis_agent() Agent
        +_create_report_agent() Agent
        +_create_coordinator_agent() Agent
        +process_daily_news(channels: List, date: str) Dict
        +get_swarm_workflow_visualization() str
    }

    class Agent {
        +name: str
        +instructions: str
        +functions: List[Callable]
    }

    class VideoProcessor {
        +download_videos(channels: str, date: str) str
        +validate_video_quality() str
    }

    class TranscriptionExpert {
        +transcribe_videos() str
        +improve_translation(video_title: str) str
    }

    class StockAnalyst {
        +analyze_stock_content() str
        +calculate_portfolio_risk() str
    }

    class ReportWriter {
        +generate_comprehensive_report() str
        +create_executive_summary() str
    }

    class ProcessCoordinator {
        +handoff_to_video_processor() Agent
        +handoff_to_transcription() Agent
        +handoff_to_analysis() Agent
        +handoff_to_report_writer() Agent
        +get_processing_status() str
    }

    SwarmStockNewsSystem --> Agent : creates
    SwarmStockNewsSystem --> VideoProcessor : specializes
    SwarmStockNewsSystem --> TranscriptionExpert : specializes
    SwarmStockNewsSystem --> StockAnalyst : specializes
    SwarmStockNewsSystem --> ReportWriter : specializes
    SwarmStockNewsSystem --> ProcessCoordinator : coordinates

    VideoProcessor --|> Agent
    TranscriptionExpert --|> Agent
    StockAnalyst --|> Agent
    ReportWriter --|> Agent
    ProcessCoordinator --|> Agent

    ProcessCoordinator ..> VideoProcessor : handoff
    ProcessCoordinator ..> TranscriptionExpert : handoff
    ProcessCoordinator ..> StockAnalyst : handoff
    ProcessCoordinator ..> ReportWriter : handoff
```
