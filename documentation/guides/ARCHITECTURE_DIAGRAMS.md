# Framework Architecture Diagrams

This document contains comprehensive architectural diagrams, class diagrams, and Mermaid diagrams for all 9 AI framework implementations.

## 📊 High-Level System Architecture

```mermaid
graph TB
    subgraph "Input Sources"
        A[YouTube Telugu Channels] --> B[moneypurse Channel]
        A --> C[daytradertelugu Channel]
    end
    
    subgraph "Framework Layer"
        D[LangChain Agent]
        E[CrewAI System]
        F[AutoGen Agents]
        G[LangGraph Workflow]
        H[PydanticAI System]
        I[Swarm Agents]
        J[Semantic Kernel]
        K[Haystack Pipeline]
        L[OpenAI Assistants]
    end
    
    subgraph "Processing Pipeline"
        M[Video Download] --> N[Transcription]
        N --> O[Stock Analysis]
        O --> P[Report Generation]
    end
    
    subgraph "Output"
        Q[Investment Reports]
        R[Stock Recommendations]
        S[Market Analysis]
    end
    
    B --> D & E & F & G & H & I & J & K & L
    C --> D & E & F & G & H & I & J & K & L
    
    D --> M
    E --> M
    F --> M
    G --> M
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M
    
    P --> Q & R & S
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style K fill:#f3e5f5
    style L fill:#f3e5f5
    style Q fill:#e8f5e8
```

## 🦜 LangChain Architecture

### Class Diagram
```mermaid
classDiagram
    class LangChainStockNewsAgent {
        -ChatOpenAI llm
        -List~Tool~ tools
        -AgentExecutor agent_executor
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_tools() List~Tool~
        -_setup_agent() AgentExecutor
        -_download_videos(channels: List~str~, date: str) str
        -_transcribe_content(video_files: List~str~) str
        -_analyze_stocks(content: str) str
        -_generate_report(analysis: str, date: str) str
    }
    
    class Tool {
        +name: str
        +description: str
        +func: Callable
    }
    
    class ChatOpenAI {
        +model: str
        +temperature: float
        +openai_api_key: str
        +invoke(messages: List) str
    }
    
    class AgentExecutor {
        +agent: Agent
        +tools: List~Tool~
        +verbose: bool
        +invoke(input: Dict) Dict
    }
    
    LangChainStockNewsAgent --> ChatOpenAI
    LangChainStockNewsAgent --> Tool
    LangChainStockNewsAgent --> AgentExecutor
    AgentExecutor --> Tool
```

### Workflow Diagram
```mermaid
flowchart TD
    A[Start LangChain Agent] --> B[Initialize ChatOpenAI]
    B --> C[Create Tools]
    C --> D[Setup Agent Executor]
    D --> E[Receive Input]
    E --> F{Tool Selection}
    
    F -->|Video Download| G[download_youtube_videos]
    F -->|Transcription| H[transcribe_video_content]
    F -->|Analysis| I[analyze_stock_content]
    F -->|Report| J[generate_investment_report]
    
    G --> K[Tool Execution]
    H --> K
    I --> K
    J --> K
    
    K --> L[LLM Processing]
    L --> M{More Tools Needed?}
    M -->|Yes| F
    M -->|No| N[Generate Final Output]
    N --> O[Save Report]
    O --> P[End]
    
    style A fill:#4caf50
    style P fill:#f44336
    style F fill:#ff9800
    style M fill:#ff9800
```

## 🤝 CrewAI Architecture

### Class Diagram
```mermaid
classDiagram
    class CrewAIStockNewsAgent {
        -ChatOpenAI llm
        -List~Agent~ agents
        -List~Task~ tasks
        -Crew crew
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_agents() List~Agent~
        -_create_tasks() List~Task~
        -_create_crew() Crew
    }
    
    class Agent {
        +role: str
        +goal: str
        +backstory: str
        +llm: ChatOpenAI
        +tools: List~Tool~
        +verbose: bool
    }
    
    class Task {
        +description: str
        +agent: Agent
        +expected_output: str
        +tools: List~Tool~
    }
    
    class Crew {
        +agents: List~Agent~
        +tasks: List~Task~
        +verbose: bool
        +process: str
        +kickoff() Dict
    }
    
    CrewAIStockNewsAgent --> Agent
    CrewAIStockNewsAgent --> Task
    CrewAIStockNewsAgent --> Crew
    Crew --> Agent
    Crew --> Task
    Task --> Agent
```

### Multi-Agent Workflow
```mermaid
graph TD
    A[Crew Manager] --> B[Video Processor Agent]
    A --> C[Transcription Expert Agent]
    A --> D[Stock Analyst Agent]
    A --> E[Report Writer Agent]
    
    B --> F[Download Videos Task]
    F --> G[Video Processing Complete]
    G --> H[Handoff to Transcription]
    
    C --> I[Transcribe Content Task]
    H --> I
    I --> J[Transcription Complete]
    J --> K[Handoff to Analysis]
    
    D --> L[Analyze Stocks Task]
    K --> L
    L --> M[Analysis Complete]
    M --> N[Handoff to Report]
    
    E --> O[Generate Report Task]
    N --> O
    O --> P[Final Report]
    
    subgraph "Agent Roles"
        B1[Role: Video Specialist<br/>Goal: Download & Process<br/>Backstory: YouTube Expert]
        C1[Role: Transcription Expert<br/>Goal: Telugu → English<br/>Backstory: Language Specialist]
        D1[Role: Senior Analyst<br/>Goal: Stock Analysis<br/>Backstory: Market Expert]
        E1[Role: Report Writer<br/>Goal: Professional Reports<br/>Backstory: Financial Writer]
    end
    
    B -.-> B1
    C -.-> C1
    D -.-> D1
    E -.-> E1
    
    style A fill:#2196f3
    style P fill:#4caf50
```

## 🏗️ AutoGen Architecture

### Class Diagram
```mermaid
classDiagram
    class AutoGenStockNewsSystem {
        -Dict config_list
        -AssistantAgent video_processor
        -AssistantAgent transcription_expert
        -AssistantAgent stock_analyst
        -AssistantAgent report_writer
        -UserProxyAgent user_proxy
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_agents() None
        -_register_functions() None
    }
    
    class AssistantAgent {
        +name: str
        +system_message: str
        +llm_config: Dict
        +function_map: Dict
        +generate_reply(messages: List) str
    }
    
    class UserProxyAgent {
        +name: str
        +human_input_mode: str
        +code_execution_config: Dict
        +function_map: Dict
        +initiate_chat(recipient: Agent, message: str) None
    }
    
    AutoGenStockNewsSystem --> AssistantAgent
    AutoGenStockNewsSystem --> UserProxyAgent
```

### Conversation Flow
```mermaid
sequenceDiagram
    participant UP as UserProxy
    participant VP as VideoProcessor
    participant TE as TranscriptionExpert
    participant SA as StockAnalyst
    participant RW as ReportWriter
    
    UP->>VP: Process videos from channels
    VP->>VP: Download & validate videos
    VP->>TE: Here are the processed videos
    
    TE->>TE: Transcribe Telugu content
    TE->>TE: Translate to English
    TE->>SA: Here's the transcribed content
    
    SA->>SA: Analyze for stock insights
    SA->>SA: Extract recommendations
    SA->>RW: Here's the analysis
    
    RW->>RW: Generate professional report
    RW->>UP: Final investment report ready
    
    Note over UP,RW: Multi-agent conversation with dynamic role switching
```

## 🧠 LangGraph Architecture

### State Machine Diagram
```mermaid
stateDiagram-v2
    [*] --> StartState
    StartState --> VideoProcessingState: Initialize
    
    VideoProcessingState --> TranscriptionState: Videos Downloaded
    VideoProcessingState --> ErrorState: Download Failed
    
    TranscriptionState --> AnalysisState: Content Transcribed
    TranscriptionState --> ErrorState: Transcription Failed
    
    AnalysisState --> ReportState: Analysis Complete
    AnalysisState --> ErrorState: Analysis Failed
    
    ReportState --> EndState: Report Generated
    ReportState --> ErrorState: Report Failed
    
    ErrorState --> [*]: Error Handled
    EndState --> [*]: Success
    
    note right of VideoProcessingState
        Process YouTube videos
        Validate content quality
        Organize files
    end note
    
    note right of TranscriptionState
        Transcribe Telugu audio
        Translate to English
        Preserve financial terms
    end note
    
    note right of AnalysisState
        Extract stock mentions
        Analyze sentiment
        Generate recommendations
    end note
```

### Class Diagram
```mermaid
classDiagram
    class LangGraphStockNewsAgent {
        -ChatOpenAI llm
        -StateGraph graph
        -Dict state_schema
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_graph() StateGraph
        -_video_processing_node(state: Dict) Dict
        -_transcription_node(state: Dict) Dict
        -_analysis_node(state: Dict) Dict
        -_report_node(state: Dict) Dict
        -_should_continue(state: Dict) str
    }
    
    class StateGraph {
        +add_node(name: str, func: Callable) None
        +add_edge(start: str, end: str) None
        +add_conditional_edges(start: str, condition: Callable) None
        +compile() CompiledGraph
    }
    
    class WorkflowState {
        +channels: List~str~
        +date: str
        +videos: List~str~
        +transcriptions: List~str~
        +analysis: Dict
        +report: str
        +error: str
    }
    
    LangGraphStockNewsAgent --> StateGraph
    LangGraphStockNewsAgent --> WorkflowState
```

## 🎯 PydanticAI Architecture

### Class Diagram
```mermaid
classDiagram
    class PydanticAIStockNewsSystem {
        -OpenAIModel model
        -Agent video_agent
        -Agent transcription_agent
        -Agent analysis_agent
        -Agent report_agent
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_agents() None
    }
    
    class VideoProcessingRequest {
        +channels: List[str]
        +date: str
        +min_duration: int
        +quality_threshold: str
    }
    
    class VideoProcessingResponse {
        +status: str
        +videos_processed: int
        +total_duration: int
        +file_paths: List[str]
        +metadata: Dict
    }
    
    class TranscriptionRequest {
        +video_files: List[str]
        +source_language: str
        +target_language: str
    }
    
    class TranscriptionResponse {
        +transcriptions: List[TranscriptionResult]
        +total_confidence: float
        +processing_time: int
    }
    
    class StockAnalysisRequest {
        +content: str
        +analysis_depth: str
        +confidence_threshold: float
    }
    
    class StockAnalysisResponse {
        +overall_sentiment: str
        +confidence_score: float
        +stocks_mentioned: List[StockRecommendation]
        +market_themes: List[str]
    }
    
    PydanticAIStockNewsSystem --> VideoProcessingRequest
    PydanticAIStockNewsSystem --> VideoProcessingResponse
    PydanticAIStockNewsSystem --> TranscriptionRequest
    PydanticAIStockNewsSystem --> TranscriptionResponse
    PydanticAIStockNewsSystem --> StockAnalysisRequest
    PydanticAIStockNewsSystem --> StockAnalysisResponse
```

### Type-Safe Workflow
```mermaid
flowchart TD
    A[Input Validation] --> B{Valid Types?}
    B -->|No| C[Pydantic Validation Error]
    B -->|Yes| D[Video Agent]
    
    D --> E[VideoProcessingResponse]
    E --> F[Type Validation]
    F --> G[Transcription Agent]
    
    G --> H[TranscriptionResponse]
    H --> I[Type Validation]
    I --> J[Analysis Agent]
    
    J --> K[StockAnalysisResponse]
    K --> L[Type Validation]
    L --> M[Report Agent]
    
    M --> N[Final Report]
    
    C --> O[Error Handling]
    
    style A fill:#4caf50
    style C fill:#f44336
    style F fill:#2196f3
    style I fill:#2196f3
    style L fill:#2196f3
```

## 🚀 Swarm Architecture

### Agent Handoff Diagram
```mermaid
graph TD
    A[Swarm Client] --> B[Video Processor Agent]
    
    B --> C{Processing Complete?}
    C -->|Yes| D[Handoff to Transcription Agent]
    C -->|No| E[Continue Processing]
    E --> B
    
    D --> F[Transcription Expert Agent]
    F --> G{Transcription Complete?}
    G -->|Yes| H[Handoff to Analysis Agent]
    G -->|No| I[Continue Transcription]
    I --> F
    
    H --> J[Stock Analyst Agent]
    J --> K{Analysis Complete?}
    K -->|Yes| L[Handoff to Report Agent]
    K -->|No| M[Continue Analysis]
    M --> J
    
    L --> N[Report Writer Agent]
    N --> O[Final Report]
    
    subgraph "Agent Functions"
        P[download_youtube_videos]
        Q[transcribe_video_content]
        R[analyze_stock_content]
        S[generate_investment_report]
    end
    
    B -.-> P
    F -.-> Q
    J -.-> R
    N -.-> S
    
    style A fill:#ff9800
    style O fill:#4caf50
```

### Class Diagram
```mermaid
classDiagram
    class SwarmStockNewsSystem {
        -Swarm client
        -Agent video_processor
        -Agent transcription_expert
        -Agent stock_analyst
        -Agent report_writer
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_agents() None
    }
    
    class Agent {
        +name: str
        +instructions: str
        +functions: List~Function~
        +model: str
    }
    
    class Function {
        +name: str
        +description: str
        +parameters: Dict
        +function: Callable
    }
    
    class Swarm {
        +run(agent: Agent, messages: List) Response
    }
    
    SwarmStockNewsSystem --> Agent
    SwarmStockNewsSystem --> Swarm
    Agent --> Function
```

## 🔧 Semantic Kernel Architecture

### Plugin Architecture
```mermaid
graph TB
    subgraph "Semantic Kernel Core"
        A[Kernel] --> B[AI Service]
        A --> C[Memory Store]
        A --> D[Plugin Collection]
    end
    
    subgraph "Plugins"
        E[VideoProcessingPlugin]
        F[TranscriptionPlugin]
        G[StockAnalysisPlugin]
        H[ReportGenerationPlugin]
    end
    
    subgraph "Skills & Functions"
        I[download_videos]
        J[validate_content]
        K[transcribe_audio]
        L[translate_text]
        M[analyze_sentiment]
        N[extract_stocks]
        O[generate_report]
        P[format_output]
    end
    
    D --> E & F & G & H
    E --> I & J
    F --> K & L
    G --> M & N
    H --> O & P
    
    B --> Q[OpenAI Service]
    C --> R[In-Memory Store]
    
    style A fill:#2196f3
    style E fill:#4caf50
    style F fill:#4caf50
    style G fill:#4caf50
    style H fill:#4caf50
```

### Class Diagram
```mermaid
classDiagram
    class SemanticKernelStockNewsSystem {
        -Kernel kernel
        -str api_key
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_setup_kernel() None
        -_register_plugins() None
        -_create_plan(goal: str) Plan
    }
    
    class VideoProcessingPlugin {
        +download_videos(channels: str, date: str) str
        +validate_content(video_files: str) str
    }
    
    class TranscriptionPlugin {
        +transcribe_audio(video_files: str) str
        +translate_text(telugu_text: str) str
    }
    
    class StockAnalysisPlugin {
        +analyze_sentiment(content: str) str
        +extract_stocks(content: str) str
    }
    
    class ReportGenerationPlugin {
        +generate_report(analysis: str, date: str) str
        +format_output(report: str) str
    }
    
    SemanticKernelStockNewsSystem --> VideoProcessingPlugin
    SemanticKernelStockNewsSystem --> TranscriptionPlugin
    SemanticKernelStockNewsSystem --> StockAnalysisPlugin
    SemanticKernelStockNewsSystem --> ReportGenerationPlugin
```

## 🔍 Haystack Architecture

### Pipeline Diagram
```mermaid
flowchart TD
    A[Input: YouTube Channels] --> B[VideoDownloader]
    B --> C[VideoValidator]
    C --> D[AudioExtractor]
    D --> E[WhisperTranscriber]
    E --> F[LanguageDetector]
    F --> G[TextTranslator]
    G --> H[DocumentSplitter]
    H --> I[DocumentEmbedder]
    I --> J[DocumentStore]
    
    J --> K[TextEmbedder]
    K --> L[Retriever]
    L --> M[PromptBuilder]
    M --> N[OpenAIGenerator]
    N --> O[AnswerBuilder]
    O --> P[ReportFormatter]
    P --> Q[Output: Investment Report]
    
    subgraph "Processing Pipeline"
        B & C & D & E & F & G
    end
    
    subgraph "RAG Pipeline"
        H & I & J & K & L & M & N & O
    end
    
    style A fill:#e3f2fd
    style Q fill:#e8f5e8
    style B fill:#fff3e0
    style H fill:#f3e5f5
```

### Class Diagram
```mermaid
classDiagram
    class HaystackStockNewsSystem {
        -str api_key
        -OpenAIDocumentEmbedder embedder
        -OpenAITextEmbedder text_embedder
        -InMemoryDocumentStore document_store
        -OpenAIGenerator generator
        -Pipeline processing_pipeline
        -Pipeline rag_pipeline
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_processing_pipeline() Pipeline
        -_create_rag_pipeline() Pipeline
    }
    
    class Pipeline {
        +add_component(name: str, instance: Component) None
        +connect(sender: str, receiver: str) None
        +run(data: Dict) Dict
    }
    
    class OpenAIGenerator {
        +api_key: str
        +model: str
        +generate(prompt: str) str
    }
    
    class InMemoryDocumentStore {
        +write_documents(documents: List~Document~) None
        +get_all_documents() List~Document~
    }
    
    HaystackStockNewsSystem --> Pipeline
    HaystackStockNewsSystem --> OpenAIGenerator
    HaystackStockNewsSystem --> InMemoryDocumentStore
```

## 🤖 OpenAI Assistants Architecture

### Assistant Workflow
```mermaid
sequenceDiagram
    participant C as Client
    participant T as Thread
    participant VA as Video Assistant
    participant TA as Transcription Assistant
    participant SA as Stock Assistant
    participant RA as Report Assistant
    
    C->>T: Create conversation thread
    C->>T: Add user message (process videos)
    C->>VA: Create run with Video Assistant
    VA->>VA: Execute download_youtube_videos function
    VA->>T: Add assistant response
    
    C->>T: Add user message (transcribe content)
    C->>TA: Create run with Transcription Assistant
    TA->>TA: Execute transcribe_video_content function
    TA->>T: Add assistant response
    
    C->>T: Add user message (analyze stocks)
    C->>SA: Create run with Stock Assistant
    SA->>SA: Execute analyze_stock_content function
    SA->>T: Add assistant response
    
    C->>T: Add user message (generate report)
    C->>RA: Create run with Report Assistant
    RA->>RA: Execute generate_investment_report function
    RA->>T: Add final report
    
    Note over C,RA: Persistent conversation with stateful memory
```

### Class Diagram
```mermaid
classDiagram
    class OpenAIAssistantsStockNewsSystem {
        -OpenAI client
        -Dict assistants
        -Thread thread
        +__init__(openai_api_key: str)
        +process_daily_news(channels: List~str~, date: str) Dict
        -_create_assistants() None
        -_create_thread() None
        -_wait_for_run_completion(run: Run) Run
        -_handle_function_calls(run: Run) Run
    }
    
    class Assistant {
        +id: str
        +name: str
        +instructions: str
        +model: str
        +tools: List~Tool~
    }
    
    class Thread {
        +id: str
        +created_at: int
        +metadata: Dict
    }
    
    class Run {
        +id: str
        +thread_id: str
        +assistant_id: str
        +status: str
        +required_action: Dict
    }
    
    OpenAIAssistantsStockNewsSystem --> Assistant
    OpenAIAssistantsStockNewsSystem --> Thread
    OpenAIAssistantsStockNewsSystem --> Run
```

## 📊 Framework Comparison Matrix

### Complexity vs Features
```mermaid
quadrantChart
    title Framework Complexity vs Feature Richness
    x-axis Low Complexity --> High Complexity
    y-axis Basic Features --> Rich Features
    
    quadrant-1 High Value (Rich + Simple)
    quadrant-2 Feature Rich (Complex)
    quadrant-3 Basic (Simple)
    quadrant-4 Over-engineered
    
    OpenAI Assistants: [0.2, 0.8]
    Swarm: [0.3, 0.4]
    PydanticAI: [0.4, 0.6]
    LangChain: [0.7, 0.9]
    AutoGen: [0.6, 0.7]
    CrewAI: [0.5, 0.7]
    LangGraph: [0.8, 0.8]
    Semantic Kernel: [0.7, 0.6]
    Haystack: [0.9, 0.9]
```

### Setup Time vs Production Readiness
```mermaid
scatter-chart
    title "Framework Setup Time vs Production Readiness"
    x-axis "Setup Time (minutes)" 0 --> 12
    y-axis "Production Readiness Score" 0 --> 10
    
    OpenAI Assistants : [2, 9]
    PydanticAI : [3, 8]
    LangChain : [3.5, 9]
    AutoGen : [3.5, 7]
    Swarm : [4, 6]
    LangGraph : [4.5, 8]
    CrewAI : [5, 7]
    Semantic Kernel : [7.5, 8]
    Haystack : [8.5, 9]
```

## 🔄 Data Flow Comparison

### Framework Data Flow Patterns
```mermaid
graph TD
    subgraph "Sequential (LangChain)"
        A1[Input] --> A2[Tool 1] --> A3[Tool 2] --> A4[Tool 3] --> A5[Output]
    end
    
    subgraph "Multi-Agent (CrewAI)"
        B1[Manager] --> B2[Agent 1] --> B3[Agent 2] --> B4[Agent 3] --> B5[Result]
    end
    
    subgraph "Conversational (AutoGen)"
        C1[UserProxy] <--> C2[Agent 1]
        C2 <--> C3[Agent 2]
        C3 <--> C4[Agent 3]
        C4 --> C5[Consensus]
    end
    
    subgraph "State Machine (LangGraph)"
        D1[State 1] --> D2{Condition}
        D2 -->|Yes| D3[State 2]
        D2 -->|No| D4[State 3]
        D3 --> D5[Final State]
        D4 --> D5
    end
    
    subgraph "Pipeline (Haystack)"
        E1[Component 1] --> E2[Component 2] --> E3[Component 3]
        E1 --> E4[Component 4] --> E3
        E3 --> E5[Output]
    end
```

## 🚀 Performance Characteristics

### Framework Performance Metrics
```mermaid
gitgraph
    commit id: "Setup Time"
    branch OpenAI_Assistants
    commit id: "1-3 min"
    checkout main
    
    branch PydanticAI
    commit id: "2-4 min"
    checkout main
    
    branch LangChain
    commit id: "2-5 min"
    checkout main
    
    branch AutoGen
    commit id: "2-5 min"
    checkout main
    
    branch Swarm
    commit id: "3-5 min"
    checkout main
    
    branch LangGraph
    commit id: "3-6 min"
    checkout main
    
    branch CrewAI
    commit id: "3-7 min"
    checkout main
    
    branch Semantic_Kernel
    commit id: "5-10 min"
    checkout main
    
    branch Haystack
    commit id: "5-12 min"
```

This comprehensive diagram collection provides visual representations of all 9 framework architectures, their relationships, data flows, and comparative characteristics. Each diagram type serves a different purpose:

- **System Architecture**: Overall system view
- **Class Diagrams**: Code structure and relationships  
- **Workflow Diagrams**: Process flows and state transitions
- **Sequence Diagrams**: Interaction patterns
- **Comparison Charts**: Framework characteristics and trade-offs

These diagrams can be rendered in any Mermaid-compatible tool or documentation system.
