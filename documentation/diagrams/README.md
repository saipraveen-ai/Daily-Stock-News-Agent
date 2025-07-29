# Swarm Agent Visual Documentation

This directory contains comprehensive visual documentation for the OpenAI Swarm implementation of the Daily Stock News Agent.

## 📊 Available Diagrams

### 1. **Class Diagram** - `swarm_class_diagram.md`
- Shows the class structure and relationships
- Displays inheritance hierarchy
- Illustrates agent specializations
- Maps function ownership

### 2. **Sequence Diagram** - `swarm_sequence_diagram.md`  
- Demonstrates agent interaction flow
- Shows handoff mechanisms
- Illustrates shared context usage
- Maps temporal workflow execution

### 3. **Architecture Overview** - `swarm_architecture_overview.md`
- High-level system architecture
- External integrations (YouTube, OpenAI API)
- Agent coordination patterns
- Data storage and output generation

### 4. **Data Flow Diagram** - `swarm_data_flow.md`
- Context evolution timeline
- Data transformations at each stage
- Agent processing pipeline
- Input/output relationships

## 🛠️ How to View Diagrams

### Method 1: VS Code Mermaid Preview
1. Install the **Mermaid Preview** extension
2. Open any diagram file (`.md`)
3. Press `Ctrl+Shift+P` → "Mermaid: Preview"
4. View interactive diagram

### Method 2: AppMap Integration
1. Run `python appmap_swarm_analysis.py` from project root
2. Open AppMap extension panel
3. View generated execution diagrams
4. Explore interactive dependency maps

### Method 3: Export Options
```bash
# Export to PNG/SVG using mermaid-cli
npm install -g @mermaid-js/mermaid-cli
mmdc -i swarm_class_diagram.md -o swarm_class_diagram.png
```

## 📈 Diagram Details

### Class Diagram Features
- **SwarmStockNewsSystem**: Main orchestrator class
- **5 Specialized Agents**: Each with specific responsibilities
- **Function Mapping**: Clear function ownership
- **Inheritance Structure**: Agent base class specialization

### Sequence Diagram Features
- **4-Stage Workflow**: Video → Transcription → Analysis → Reports
- **Shared Context**: Central data coordination
- **Handoff Mechanisms**: Inter-agent communication
- **Error Handling**: Graceful failure management

### Architecture Overview Features
- **External Dependencies**: YouTube API, OpenAI API
- **Data Storage**: Files and structured data
- **Output Generation**: Multiple format support
- **Agent Coordination**: Lightweight Swarm framework

### Data Flow Features
- **Context Evolution**: Step-by-step data accumulation
- **Transformations**: Data format changes
- **Pipeline Stages**: Sequential processing
- **Feedback Loops**: Quality improvement cycles

## 🎯 Use Cases for Diagrams

### For Developers
- **Understanding**: Grasp system architecture quickly
- **Debugging**: Trace execution flow
- **Extension**: Add new agents or functions
- **Testing**: Verify component interactions

### For Stakeholders  
- **Overview**: High-level system understanding
- **Planning**: Feature development roadmap
- **Documentation**: Technical specification reference
- **Compliance**: Architecture review requirements

### For DevOps
- **Deployment**: Infrastructure planning
- **Monitoring**: Performance bottleneck identification
- **Scaling**: Component resource allocation
- **Maintenance**: System health monitoring

## 🔄 Updating Diagrams

When modifying the Swarm agent implementation:

1. **Code Changes**: Update the implementation
2. **Run Analysis**: Execute `appmap_swarm_analysis.py`
3. **Update Diagrams**: Modify Mermaid syntax if needed
4. **Validate**: Ensure diagrams reflect current code
5. **Commit**: Include diagram updates with code changes

## 📝 Diagram Maintenance Checklist

- [ ] Class diagram reflects current class structure
- [ ] Sequence diagram shows correct interaction flow
- [ ] Architecture overview includes all components
- [ ] Data flow diagram matches context evolution
- [ ] All diagrams use consistent styling
- [ ] AppMap analysis runs without errors
- [ ] Export formats are up to date

---

## 🚀 Quick Commands

```bash
# View all diagrams in VS Code
code documentation/diagrams/

# Run AppMap analysis
python appmap_swarm_analysis.py

# Generate new execution trace
python testing/swarm/test_swarm_mock.py

# Export diagrams (if mermaid-cli installed)
cd documentation/diagrams/
mmdc -i *.md -o ../exports/
```

**Status**: ✅ All diagrams current as of 2025-07-22  
**Framework**: OpenAI Swarm v0.1.0  
**Agent Count**: 5 specialized agents  
**Complexity**: Low-to-Medium (lightweight coordination)
