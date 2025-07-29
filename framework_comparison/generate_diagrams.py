#!/usr/bin/env python3
"""
Framework Architecture Diagram Generator

This script generates visual diagrams for all 9 AI framework implementations
including architectural diagrams, class diagrams, and workflow visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns
from datetime import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FrameworkDiagramGenerator:
    """Generate visual diagrams for framework comparison"""
    
    def __init__(self):
        self.frameworks = {
            'OpenAI Assistants': {'setup_time': 2, 'complexity': 3, 'features': 9, 'prod_ready': 9},
            'PydanticAI': {'setup_time': 3, 'complexity': 4, 'features': 7, 'prod_ready': 8},
            'LangChain': {'setup_time': 3.5, 'complexity': 7, 'features': 10, 'prod_ready': 9},
            'AutoGen': {'setup_time': 3.5, 'complexity': 6, 'features': 8, 'prod_ready': 7},
            'Swarm': {'setup_time': 4, 'complexity': 3, 'features': 5, 'prod_ready': 6},
            'LangGraph': {'setup_time': 4.5, 'complexity': 8, 'features': 9, 'prod_ready': 8},
            'CrewAI': {'setup_time': 5, 'complexity': 5, 'features': 8, 'prod_ready': 7},
            'Semantic Kernel': {'setup_time': 7.5, 'complexity': 7, 'features': 7, 'prod_ready': 8},
            'Haystack': {'setup_time': 8.5, 'complexity': 9, 'features': 10, 'prod_ready': 9}
        }
        
        self.colors = sns.color_palette("husl", len(self.frameworks))
    
    def generate_architecture_overview(self):
        """Generate high-level architecture overview diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Define layers and components
        layers = {
            'Input': {'y': 0.8, 'components': ['YouTube Telugu Channels', 'moneypurse', 'daytradertelugu']},
            'Frameworks': {'y': 0.6, 'components': list(self.frameworks.keys())},
            'Processing': {'y': 0.4, 'components': ['Video Download', 'Transcription', 'Stock Analysis', 'Report Gen']},
            'Output': {'y': 0.2, 'components': ['Investment Reports', 'Stock Recommendations', 'Market Analysis']}
        }
        
        # Draw layers
        for layer_name, layer_info in layers.items():
            y = layer_info['y']
            components = layer_info['components']
            
            # Layer background
            rect = patches.Rectangle((0.05, y-0.08), 0.9, 0.16, 
                                   linewidth=2, edgecolor='gray', 
                                   facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
            
            # Layer title
            ax.text(0.02, y, layer_name, fontsize=14, fontweight='bold', 
                   verticalalignment='center')
            
            # Components
            x_positions = np.linspace(0.1, 0.9, len(components))
            for i, component in enumerate(components):
                x = x_positions[i]
                
                # Component box
                if layer_name == 'Frameworks':
                    color = self.colors[i % len(self.colors)]
                else:
                    color = 'lightblue'
                    
                rect = patches.Rectangle((x-0.06, y-0.03), 0.12, 0.06,
                                       linewidth=1, edgecolor='black',
                                       facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Component text
                ax.text(x, y, component, fontsize=8, ha='center', va='center',
                       wrap=True, fontweight='bold' if layer_name == 'Frameworks' else 'normal')
        
        # Draw connections
        # Input to Frameworks
        for i in range(len(layers['Input']['components'])):
            for j in range(len(layers['Frameworks']['components'])):
                ax.annotate('', xy=(0.1 + j*0.09, 0.52), xytext=(0.3 + i*0.2, 0.72),
                           arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3))
        
        # Frameworks to Processing
        for i in range(len(layers['Frameworks']['components'])):
            ax.annotate('', xy=(0.25, 0.48), xytext=(0.1 + i*0.09, 0.52),
                       arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3))
        
        # Processing to Output
        for i in range(len(layers['Output']['components'])):
            ax.annotate('', xy=(0.3 + i*0.2, 0.28), xytext=(0.65, 0.32),
                       arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Daily Stock News Agent - Framework Architecture Overview', 
                    fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('framework_architecture_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_complexity_vs_features_chart(self):
        """Generate complexity vs features scatter plot"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        x = [data['complexity'] for data in self.frameworks.values()]
        y = [data['features'] for data in self.frameworks.values()]
        names = list(self.frameworks.keys())
        
        # Create scatter plot
        scatter = ax.scatter(x, y, c=self.colors, s=200, alpha=0.7, edgecolors='black')
        
        # Add framework names
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add quadrants
        ax.axhline(y=7.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
        
        # Quadrant labels
        ax.text(3, 9, 'High Value\n(Rich + Simple)', ha='center', va='center', 
               fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.text(8, 9, 'Feature Rich\n(Complex)', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        ax.text(3, 5, 'Basic\n(Simple)', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.text(8, 5, 'Over-engineered', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        ax.set_xlabel('Complexity Score (1-10)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature Richness Score (1-10)', fontsize=12, fontweight='bold')
        ax.set_title('Framework Complexity vs Feature Richness', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 10)
        ax.set_ylim(3, 11)
        
        plt.tight_layout()
        plt.savefig('complexity_vs_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_setup_time_ranking(self):
        """Generate setup time ranking bar chart"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Sort frameworks by setup time
        sorted_frameworks = sorted(self.frameworks.items(), key=lambda x: x[1]['setup_time'])
        names = [item[0] for item in sorted_frameworks]
        times = [item[1]['setup_time'] for item in sorted_frameworks]
        
        # Create horizontal bar chart
        bars = ax.barh(names, times, color=self.colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{time:.1f} min', ha='left', va='center', fontweight='bold')
        
        # Add medal emojis for top 3
        ax.text(times[0] + 0.5, 0, '🥇', fontsize=20, ha='left', va='center')
        ax.text(times[1] + 0.5, 1, '🥈', fontsize=20, ha='left', va='center')
        ax.text(times[2] + 0.5, 2, '🥉', fontsize=20, ha='left', va='center')
        
        ax.set_xlabel('Setup Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Framework', fontsize=12, fontweight='bold')
        ax.set_title('Framework Setup Time Ranking (Easier to Harder)', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(0, max(times) + 1)
        
        plt.tight_layout()
        plt.savefig('setup_time_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_framework_radar_chart(self):
        """Generate radar chart comparing framework characteristics"""
        # Characteristics to compare
        characteristics = ['Setup Speed', 'Feature Richness', 'Production Ready', 
                         'Documentation', 'Community', 'Learning Curve']
        
        # Data for each framework (normalized 0-10)
        framework_data = {
            'OpenAI Assistants': [9, 9, 9, 8, 7, 9],
            'LangChain': [7, 10, 9, 10, 10, 6],
            'CrewAI': [6, 8, 7, 7, 8, 7],
            'AutoGen': [7, 8, 7, 8, 8, 7],
            'LangGraph': [6, 9, 8, 8, 8, 5],
            'PydanticAI': [8, 7, 8, 7, 6, 8],
            'Swarm': [7, 5, 6, 6, 5, 8],
            'Semantic Kernel': [4, 7, 8, 7, 6, 5],
            'Haystack': [3, 10, 9, 9, 8, 4]
        }
        
        # Number of characteristics
        N = len(characteristics)
        
        # Compute angle for each characteristic
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create subplots for radar charts
        fig, axes = plt.subplots(3, 3, figsize=(18, 18), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        for i, (framework, values) in enumerate(framework_data.items()):
            ax = axes[i]
            
            # Add values for closing the plot
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=framework, color=self.colors[i])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(characteristics, fontsize=10)
            ax.set_ylim(0, 10)
            ax.set_yticks(range(0, 11, 2))
            ax.set_yticklabels(range(0, 11, 2), fontsize=8)
            ax.set_title(framework, fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
        
        plt.suptitle('Framework Characteristics Comparison (Radar Charts)', 
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig('framework_radar_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_data_flow_diagram(self):
        """Generate data flow comparison diagram"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        flow_patterns = {
            'Sequential (LangChain)': ['Input', 'Tool 1', 'Tool 2', 'Tool 3', 'Output'],
            'Multi-Agent (CrewAI)': ['Manager', 'Agent 1', 'Agent 2', 'Agent 3', 'Result'],
            'Conversational (AutoGen)': ['UserProxy', 'Agent 1', 'Agent 2', 'Agent 3', 'Consensus'],
            'State Machine (LangGraph)': ['State 1', 'Condition', 'State 2/3', 'Final State'],
            'Pipeline (Haystack)': ['Input', 'Process 1', 'Process 2', 'Process 3', 'Output'],
            'Assistant (OpenAI)': ['Thread', 'Assistant 1', 'Assistant 2', 'Assistant 3', 'Report']
        }
        
        for i, (pattern_name, steps) in enumerate(flow_patterns.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create flow diagram
            y_pos = 0.5
            x_positions = np.linspace(0.1, 0.9, len(steps))
            
            for j, (x, step) in enumerate(zip(x_positions, steps)):
                # Draw box
                rect = patches.Rectangle((x-0.08, y_pos-0.1), 0.16, 0.2,
                                       linewidth=2, edgecolor='black',
                                       facecolor=self.colors[i], alpha=0.7)
                ax.add_patch(rect)
                
                # Add text
                ax.text(x, y_pos, step, ha='center', va='center', 
                       fontsize=8, fontweight='bold', wrap=True)
                
                # Draw arrow to next step
                if j < len(steps) - 1:
                    ax.annotate('', xy=(x_positions[j+1]-0.08, y_pos), 
                               xytext=(x+0.08, y_pos),
                               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(pattern_name, fontsize=12, fontweight='bold')
        
        # Hide unused subplot
        if len(flow_patterns) < len(axes):
            axes[-1].axis('off')
        
        plt.suptitle('Framework Data Flow Patterns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data_flow_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_cost_analysis_chart(self):
        """Generate cost analysis comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Cost data (estimated daily development costs in USD)
        cost_data = {
            'OpenAI Assistants': {'demo': 0.15, 'full': 2.25, 'daily': 11.5},
            'PydanticAI': {'demo': 0.15, 'full': 2.25, 'daily': 11.5},
            'Swarm': {'demo': 0.15, 'full': 2.25, 'daily': 11.5},
            'LangChain': {'demo': 0.20, 'full': 3.00, 'daily': 14.0},
            'LangGraph': {'demo': 0.20, 'full': 3.00, 'daily': 14.0},
            'Haystack': {'demo': 0.20, 'full': 3.00, 'daily': 14.0},
            'Semantic Kernel': {'demo': 0.25, 'full': 3.75, 'daily': 16.0},
            'CrewAI': {'demo': 0.30, 'full': 4.50, 'daily': 20.0},
            'AutoGen': {'demo': 0.30, 'full': 4.50, 'daily': 20.0}
        }
        
        # Chart 1: Cost comparison bar chart
        frameworks = list(cost_data.keys())
        demo_costs = [cost_data[fw]['demo'] for fw in frameworks]
        full_costs = [cost_data[fw]['full'] for fw in frameworks]
        daily_costs = [cost_data[fw]['daily'] for fw in frameworks]
        
        x = np.arange(len(frameworks))
        width = 0.25
        
        bars1 = ax1.bar(x - width, demo_costs, width, label='Demo Mode', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x, full_costs, width, label='Full Processing', alpha=0.8, color='orange')
        bars3 = ax1.bar(x + width, daily_costs, width, label='Daily Development', alpha=0.8, color='red')
        
        ax1.set_xlabel('Framework', fontweight='bold')
        ax1.set_ylabel('Cost (USD)', fontweight='bold')
        ax1.set_title('Estimated Costs by Framework', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(frameworks, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Cost efficiency scatter plot
        setup_times = [self.frameworks[fw]['setup_time'] for fw in frameworks]
        efficiency_scores = [daily_costs[i] / setup_times[i] for i in range(len(frameworks))]
        
        scatter = ax2.scatter(setup_times, daily_costs, c=efficiency_scores, 
                            s=200, alpha=0.7, cmap='RdYlGn_r', edgecolors='black')
        
        for i, fw in enumerate(frameworks):
            ax2.annotate(fw, (setup_times[i], daily_costs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Setup Time (minutes)', fontweight='bold')
        ax2.set_ylabel('Daily Development Cost (USD)', fontweight='bold')
        ax2.set_title('Cost vs Setup Time Efficiency', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Cost/Time Ratio', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cost_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_diagrams(self):
        """Generate all diagrams"""
        print("🎨 Generating Framework Architecture Diagrams...")
        print("=" * 60)
        
        diagrams = [
            ("Architecture Overview", self.generate_architecture_overview),
            ("Complexity vs Features", self.generate_complexity_vs_features_chart),
            ("Setup Time Ranking", self.generate_setup_time_ranking),
            ("Framework Radar Charts", self.generate_framework_radar_chart),
            ("Data Flow Patterns", self.generate_data_flow_diagram),
            ("Cost Analysis", self.generate_cost_analysis_chart)
        ]
        
        for name, func in diagrams:
            print(f"📊 Generating {name}...")
            try:
                func()
                print(f"✅ {name} generated successfully!")
            except Exception as e:
                print(f"❌ Error generating {name}: {e}")
            print("-" * 40)
        
        print("🎉 All diagrams generated successfully!")
        print(f"📁 Files saved in current directory:")
        print("   - framework_architecture_overview.png")
        print("   - complexity_vs_features.png")
        print("   - setup_time_ranking.png")
        print("   - framework_radar_charts.png")
        print("   - data_flow_patterns.png")
        print("   - cost_analysis.png")

def create_framework_class_diagram():
    """Create a text-based class diagram for LangChain implementation"""
    class_diagram = """
    ┌─────────────────────────────────────┐
    │        LangChainStockNewsAgent      │
    ├─────────────────────────────────────┤
    │ - llm: ChatOpenAI                   │
    │ - tools: List[Tool]                 │
    │ - agent_executor: AgentExecutor     │
    ├─────────────────────────────────────┤
    │ + __init__(api_key: str)            │
    │ + process_daily_news(...) Dict      │
    │ - _create_tools() List[Tool]        │
    │ - _setup_agent() AgentExecutor      │
    └─────────────────────────────────────┘
                      │
                      │ uses
                      ▼
    ┌─────────────────────────────────────┐
    │             ChatOpenAI              │
    ├─────────────────────────────────────┤
    │ + model: str                        │
    │ + temperature: float                │
    │ + openai_api_key: str              │
    ├─────────────────────────────────────┤
    │ + invoke(messages: List) str        │
    └─────────────────────────────────────┘
    
                      │
                      │ creates
                      ▼
    ┌─────────────────────────────────────┐
    │               Tool                  │
    ├─────────────────────────────────────┤
    │ + name: str                         │
    │ + description: str                  │
    │ + func: Callable                    │
    └─────────────────────────────────────┘
    """
    return class_diagram

def main():
    """Main function to generate all diagrams"""
    print("🚀 Framework Architecture Diagram Generator")
    print("=" * 50)
    print(f"📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if required libraries are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Required libraries available")
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("💡 Install with: pip install matplotlib seaborn")
        return
    
    # Generate diagrams
    generator = FrameworkDiagramGenerator()
    generator.generate_all_diagrams()
    
    # Print class diagram
    print("\n" + "="*60)
    print("📋 Example Class Diagram (LangChain):")
    print(create_framework_class_diagram())

if __name__ == "__main__":
    main()
