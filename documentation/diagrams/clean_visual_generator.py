#!/usr/bin/env python3
"""
Clean Visual Framework Architecture Generator

Creates clean, professional visual diagrams using matplotlib and other visualization libraries
to represent framework architectures in a clear, intuitive way without emoji dependencies.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, ConnectionPatch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import seaborn as sns

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class CleanVisualFrameworkDiagrams:
    """Create clean, professional visual diagrams for framework architectures"""
    
    def __init__(self):
        self.colors = {
            'input': '#E3F2FD',
            'framework': '#F3E5F5', 
            'processing': '#E8F5E8',
            'output': '#FFF3E0',
            'agent': '#FFEBEE',
            'tool': '#F1F8E9',
            'data': '#E0F2F1'
        }
        
        self.framework_colors = {
            'OpenAI Assistants': '#FF5722',
            'LangChain': '#2196F3', 
            'CrewAI': '#4CAF50',
            'AutoGen': '#FF9800',
            'LangGraph': '#9C27B0',
            'PydanticAI': '#00BCD4',
            'Swarm': '#795548',
            'Semantic Kernel': '#607D8B',
            'Haystack': '#E91E63'
        }

    def create_architecture_overview(self):
        """Create a clean architectural overview diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Title
        ax.text(8, 9.5, 'Daily Stock News Agent - System Architecture', 
               ha='center', va='center', fontsize=20, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        # Input Layer
        input_box = FancyBboxPatch((0.5, 7), 3, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=self.colors['input'],
                                 edgecolor='black',
                                 linewidth=2)
        ax.add_patch(input_box)
        ax.text(2, 7.75, 'INPUT SOURCES', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        ax.text(2, 7.3, '• YouTube Videos\n• Channel Data\n• Content Analysis', 
               ha='center', va='center', fontsize=10)
        
        # Framework Layer (center)
        framework_circle = Circle((8, 6), 2, facecolor=self.colors['framework'], 
                                edgecolor='black', linewidth=3, alpha=0.7)
        ax.add_patch(framework_circle)
        ax.text(8, 6.5, 'AI FRAMEWORKS', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        ax.text(8, 6, '9 Different\nImplementations', ha='center', va='center', 
               fontsize=11)
        ax.text(8, 5.5, 'OpenAI • LangChain • CrewAI\nAutoGen • LangGraph • PydanticAI\nSwarm • Semantic Kernel • Haystack', 
               ha='center', va='center', fontsize=8)
        
        # Processing Pipeline
        processes = ['Download', 'Transcribe', 'Analyze', 'Report']
        process_x = [3, 5.5, 8, 10.5]
        for i, (process, x) in enumerate(zip(processes, process_x)):
            process_box = FancyBboxPatch((x-0.7, 3.5), 1.4, 1,
                                       boxstyle="round,pad=0.1",
                                       facecolor=self.colors['processing'],
                                       edgecolor='black',
                                       linewidth=2)
            ax.add_patch(process_box)
            ax.text(x, 4, process, ha='center', va='center', 
                   fontsize=11, fontweight='bold')
            
            if i < len(processes) - 1:
                ax.annotate('', xy=(process_x[i+1]-0.7, 4), xytext=(x+0.7, 4),
                           arrowprops=dict(arrowstyle='->', lw=3, color='darkgreen'))
        
        # Output Layer
        output_box = FancyBboxPatch((12, 7), 3, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=self.colors['output'],
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(output_box)
        ax.text(13.5, 7.75, 'OUTPUTS', ha='center', va='center', 
               fontsize=14, fontweight='bold')
        ax.text(13.5, 7.3, '• Investment Insights\n• Market Analysis\n• Action Reports', 
               ha='center', va='center', fontsize=10)
        
        # Flow arrows
        arrows = [
            ((3.5, 7.75), (6, 6.8)),    # Input to Framework
            ((10, 6.8), (12, 7.75)),    # Framework to Output
            ((8, 4.5), (8, 5.2)),       # Processing to Framework
        ]
        
        for (start, end) in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=4, color='darkblue', alpha=0.8))
        
        # Labels
        ax.text(2, 9, 'DATA INPUT', ha='center', va='center', fontsize=12, 
               fontweight='bold', color='blue',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax.text(6.5, 2.5, 'PROCESSING PIPELINE', ha='center', va='center', fontsize=12, 
               fontweight='bold', color='green',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        ax.text(13.5, 9, 'RESULTS', ha='center', va='center', fontsize=12, 
               fontweight='bold', color='orange',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 16)
        ax.set_ylim(2, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('clean_system_architecture.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()

    def create_framework_comparison_matrix(self):
        """Create a clean framework comparison matrix"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        frameworks = list(self.framework_colors.keys())
        
        # Comparison criteria
        criteria = ['Setup Time', 'Complexity', 'Features', 'Multi-Agent', 'Type Safety', 'Enterprise']
        
        # Data matrix (higher = better, except setup time where lower = better)
        data = np.array([
            [9, 8, 7, 6, 3, 8],  # OpenAI Assistants
            [6, 4, 10, 7, 6, 9], # LangChain
            [7, 6, 8, 10, 5, 7], # CrewAI
            [6, 7, 8, 9, 6, 6],  # AutoGen
            [5, 3, 9, 8, 7, 8],  # LangGraph
            [8, 8, 7, 4, 10, 5], # PydanticAI
            [9, 9, 5, 8, 4, 4],  # Swarm
            [3, 4, 7, 6, 8, 10], # Semantic Kernel
            [2, 2, 10, 5, 7, 8]  # Haystack
        ])
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=10)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(criteria)))
        ax.set_yticks(np.arange(len(frameworks)))
        ax.set_xticklabels(criteria, fontsize=12, fontweight='bold')
        ax.set_yticklabels(frameworks, fontsize=11)
        
        # Add values to cells
        for i in range(len(frameworks)):
            for j in range(len(criteria)):
                text = ax.text(j, i, data[i, j], ha="center", va="center",
                             color="black" if data[i, j] > 5 else "white",
                             fontweight='bold', fontsize=11)
        
        # Color code framework names
        for i, framework in enumerate(frameworks):
            ax.get_yticklabels()[i].set_color(self.framework_colors[framework])
            ax.get_yticklabels()[i].set_fontweight('bold')
        
        ax.set_title('Framework Comparison Matrix\n(Scores: 1=Poor, 10=Excellent)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Performance Score', rotation=270, labelpad=20, fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(len(criteria)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(frameworks)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        
        plt.tight_layout()
        plt.savefig('framework_comparison_matrix.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def create_architecture_patterns(self):
        """Create visual representation of different architecture patterns"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Framework Architecture Patterns', fontsize=20, fontweight='bold')
        
        patterns = [
            ('OpenAI Assistants', 'Thread-Based Pattern'),
            ('LangChain', 'Sequential Chain Pattern'),
            ('CrewAI', 'Multi-Agent Team Pattern'),
            ('AutoGen', 'Conversation Pattern'),
            ('LangGraph', 'State Machine Pattern'),
            ('PydanticAI', 'Type-Safe Pattern'),
            ('Swarm', 'Handoff Pattern'),
            ('Semantic Kernel', 'Plugin Pattern'),
            ('Haystack', 'Pipeline Pattern')
        ]
        
        for idx, (framework, pattern) in enumerate(patterns):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            # Framework container
            ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                                      boxstyle="round,pad=0.05",
                                      facecolor=self.framework_colors[framework],
                                      alpha=0.3,
                                      edgecolor='black',
                                      linewidth=2))
            
            ax.text(0.5, 0.95, framework, ha='center', va='top', 
                   fontsize=11, fontweight='bold', transform=ax.transAxes)
            ax.text(0.5, 0.85, pattern, ha='center', va='top', 
                   fontsize=9, style='italic', transform=ax.transAxes)
            
            # Draw specific patterns
            self._draw_pattern(ax, framework.replace(' ', '_').lower())
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('architecture_patterns.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def _draw_pattern(self, ax, framework_type):
        """Draw specific pattern for each framework"""
        if 'openai' in framework_type:
            # Thread pattern
            center = Circle((0.5, 0.5), 0.1, facecolor='lightblue', edgecolor='black')
            ax.add_patch(center)
            ax.text(0.5, 0.5, 'Thread', ha='center', va='center', fontsize=8)
            
            assistants = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
            for x, y in assistants:
                assistant = Circle((x, y), 0.05, facecolor='orange', edgecolor='black')
                ax.add_patch(assistant)
                ax.plot([0.5, x], [0.5, y], 'k-', alpha=0.5)
        
        elif 'langchain' in framework_type:
            # Chain pattern
            boxes = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)]
            for i, (x, y) in enumerate(boxes):
                box = FancyBboxPatch((x-0.05, y-0.05), 0.1, 0.1,
                                   boxstyle="round,pad=0.01",
                                   facecolor='lightgreen',
                                   edgecolor='black')
                ax.add_patch(box)
                if i < len(boxes) - 1:
                    ax.annotate('', xy=(boxes[i+1][0]-0.05, y), xytext=(x+0.05, y),
                               arrowprops=dict(arrowstyle='->', lw=2))
        
        elif 'crewai' in framework_type:
            # Multi-agent team
            manager = Circle((0.5, 0.6), 0.06, facecolor='red', edgecolor='black')
            ax.add_patch(manager)
            ax.text(0.5, 0.6, 'Mgr', ha='center', va='center', fontsize=7)
            
            agents = [(0.3, 0.4), (0.7, 0.4), (0.4, 0.3), (0.6, 0.3)]
            for x, y in agents:
                agent = Circle((x, y), 0.04, facecolor='yellow', edgecolor='black')
                ax.add_patch(agent)
                ax.plot([0.5, x], [0.6, y], 'k-', alpha=0.5)
        
        elif 'autogen' in framework_type:
            # Conversation pattern
            agents = [(0.3, 0.6), (0.7, 0.6), (0.5, 0.4)]
            colors = ['lightcoral', 'lightblue', 'lightgreen']
            for (x, y), color in zip(agents, colors):
                agent = Circle((x, y), 0.06, facecolor=color, edgecolor='black')
                ax.add_patch(agent)
                ax.text(x, y, 'A', ha='center', va='center', fontsize=8)
            
            # Conversation lines
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    ax.plot([agents[i][0], agents[j][0]], 
                           [agents[i][1], agents[j][1]], 'k--', alpha=0.5)
        
        elif 'langgraph' in framework_type:
            # State machine
            states = [(0.3, 0.7), (0.7, 0.7), (0.7, 0.3), (0.3, 0.3)]
            for i, (x, y) in enumerate(states):
                state = FancyBboxPatch((x-0.05, y-0.05), 0.1, 0.1,
                                     boxstyle="round,pad=0.01",
                                     facecolor='lightpink',
                                     edgecolor='black')
                ax.add_patch(state)
                ax.text(x, y, f'S{i+1}', ha='center', va='center', fontsize=7)
                if i < len(states) - 1:
                    next_x, next_y = states[(i+1) % len(states)]
                    ax.annotate('', xy=(next_x-0.03, next_y), xytext=(x+0.03, y),
                               arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
        
        elif 'pydantic' in framework_type:
            # Type safety
            boxes = [
                (0.2, 0.6, 'Input\nType'),
                (0.5, 0.6, 'Validate'),
                (0.8, 0.6, 'Output\nType')
            ]
            colors = ['lightcyan', 'yellow', 'lightgreen']
            for (x, y, label), color in zip(boxes, colors):
                box = FancyBboxPatch((x-0.08, y-0.08), 0.16, 0.16,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color,
                                   edgecolor='black')
                ax.add_patch(box)
                ax.text(x, y, label, ha='center', va='center', fontsize=7)
            
            # Arrows
            ax.annotate('', xy=(0.42, 0.6), xytext=(0.28, 0.6),
                       arrowprops=dict(arrowstyle='->', lw=2))
            ax.annotate('', xy=(0.72, 0.6), xytext=(0.58, 0.6),
                       arrowprops=dict(arrowstyle='->', lw=2))
        
        elif 'swarm' in framework_type:
            # Handoff pattern
            agents = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5)]
            for i, (x, y) in enumerate(agents):
                agent = Circle((x, y), 0.05, facecolor='orange', edgecolor='black')
                ax.add_patch(agent)
                ax.text(x, y, f'A{i+1}', ha='center', va='center', fontsize=7)
                if i < len(agents) - 1:
                    ax.annotate('', xy=(agents[i+1][0]-0.05, y), xytext=(x+0.05, y),
                               arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
        
        elif 'semantic' in framework_type:
            # Plugin pattern
            kernel = Circle((0.5, 0.5), 0.08, facecolor='purple', edgecolor='black')
            ax.add_patch(kernel)
            ax.text(0.5, 0.5, 'Kernel', ha='center', va='center', fontsize=7, color='white')
            
            plugins = [(0.3, 0.7), (0.7, 0.7), (0.3, 0.3), (0.7, 0.3)]
            for x, y in plugins:
                plugin = FancyBboxPatch((x-0.04, y-0.03), 0.08, 0.06,
                                      boxstyle="round,pad=0.01",
                                      facecolor='lavender',
                                      edgecolor='purple')
                ax.add_patch(plugin)
                ax.plot([0.5, x], [0.5, y], 'purple', lw=1.5)
        
        elif 'haystack' in framework_type:
            # Pipeline pattern
            components = [(0.15, 0.5), (0.35, 0.5), (0.55, 0.5), (0.75, 0.5)]
            labels = ['Input', 'Process', 'Analyze', 'Output']
            for (x, y), label in zip(components, labels):
                comp = FancyBboxPatch((x-0.06, y-0.06), 0.12, 0.12,
                                    boxstyle="round,pad=0.02",
                                    facecolor='lightsteelblue',
                                    edgecolor='steelblue')
                ax.add_patch(comp)
                ax.text(x, y, label, ha='center', va='center', fontsize=6)
                if x < 0.7:
                    next_x = components[components.index((x, y)) + 1][0]
                    ax.annotate('', xy=(next_x-0.06, y), xytext=(x+0.06, y),
                               arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))

    def create_performance_dashboard(self):
        """Create a comprehensive performance dashboard"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        frameworks = list(self.framework_colors.keys())
        setup_times = [2, 3, 5, 3.5, 4.5, 3, 4, 7.5, 8.5]
        complexity_scores = [3, 7, 5, 6, 8, 4, 3, 7, 9]
        features = [9, 10, 8, 8, 9, 7, 5, 7, 10]
        
        # 1. Setup Time Ranking
        ax1 = fig.add_subplot(gs[0, :2])
        bars = ax1.barh(frameworks, setup_times, 
                       color=[self.framework_colors[fw] for fw in frameworks],
                       alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Setup Time (minutes)')
        ax1.set_title('Setup Time Ranking', fontweight='bold', fontsize=14)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add time labels
        for bar, time in zip(bars, setup_times):
            ax1.text(time + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{time} min', va='center', fontweight='bold')
        
        # 2. Complexity vs Features Scatter
        ax2 = fig.add_subplot(gs[0, 2:])
        scatter = ax2.scatter(complexity_scores, features, 
                            s=[p*30 for p in [9, 10, 7, 7, 6, 6, 5, 6, 8]],
                            c=[self.framework_colors[fw] for fw in frameworks],
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, fw in enumerate(frameworks):
            ax2.annotate(fw, (complexity_scores[i], features[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold')
        
        ax2.set_xlabel('Complexity Score')
        ax2.set_ylabel('Feature Richness')
        ax2.set_title('Complexity vs Features', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. Framework Categories
        ax3 = fig.add_subplot(gs[1, :2])
        categories = ['Simple', 'Medium', 'Complex']
        category_counts = [3, 3, 3]  # Based on complexity
        colors = ['lightgreen', 'orange', 'red']
        
        wedges, texts, autotexts = ax3.pie(category_counts, labels=categories, 
                                          colors=colors, autopct='%1.0f%%',
                                          startangle=90)
        ax3.set_title('Framework Complexity Distribution', fontweight='bold', fontsize=14)
        
        # 4. Feature Comparison Radar
        ax4 = fig.add_subplot(gs[1, 2:], projection='polar')
        
        categories_radar = ['Setup\nEase', 'Features', 'Multi-Agent', 
                           'Type Safety', 'Enterprise', 'Community']
        N = len(categories_radar)
        
        # Sample data for top 3 frameworks
        top_frameworks = ['LangChain', 'CrewAI', 'LangGraph']
        values = {
            'LangChain': [7, 10, 7, 6, 9, 10],
            'CrewAI': [6, 8, 10, 5, 7, 8],
            'LangGraph': [5, 9, 8, 7, 8, 7]
        }
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for fw in top_frameworks:
            values_fw = values[fw] + values[fw][:1]  # Complete the circle
            ax4.plot(angles, values_fw, 'o-', linewidth=2, 
                    label=fw, color=self.framework_colors[fw])
            ax4.fill(angles, values_fw, alpha=0.25, color=self.framework_colors[fw])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories_radar)
        ax4.set_ylim(0, 10)
        ax4.set_title('Top 3 Frameworks Comparison', fontweight='bold', fontsize=14, pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 5. Implementation Difficulty Timeline
        ax5 = fig.add_subplot(gs[2, :])
        
        difficulty_order = sorted(zip(frameworks, setup_times), key=lambda x: x[1])
        ordered_frameworks = [x[0] for x in difficulty_order]
        ordered_times = [x[1] for x in difficulty_order]
        
        y_pos = np.arange(len(ordered_frameworks))
        bars = ax5.barh(y_pos, ordered_times, 
                       color=[self.framework_colors[fw] for fw in ordered_frameworks],
                       alpha=0.8, edgecolor='black')
        
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(ordered_frameworks)
        ax5.set_xlabel('Implementation Time (minutes)')
        ax5.set_title('Framework Implementation Difficulty (Easiest to Hardest)', 
                     fontweight='bold', fontsize=16)
        ax5.grid(axis='x', alpha=0.3)
        
        # Add difficulty levels
        difficulty_zones = [
            (0, 3, 'Easy', 'lightgreen'),
            (3, 6, 'Medium', 'orange'), 
            (6, 10, 'Hard', 'lightcoral')
        ]
        
        for start, end, label, color in difficulty_zones:
            ax5.axvspan(start, end, alpha=0.2, color=color)
            ax5.text((start + end) / 2, len(ordered_frameworks), label,
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.suptitle('Framework Performance Dashboard', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def generate_all_clean_diagrams(self):
        """Generate all clean visual diagrams"""
        print("🎨 Generating Clean Visual Framework Diagrams...")
        print("=" * 60)
        
        diagrams = [
            ("System Architecture Overview", self.create_architecture_overview),
            ("Framework Comparison Matrix", self.create_framework_comparison_matrix),
            ("Architecture Patterns", self.create_architecture_patterns),
            ("Performance Dashboard", self.create_performance_dashboard)
        ]
        
        for name, func in diagrams:
            print(f"🖼️  Generating {name}...")
            try:
                func()
                print(f"✅ {name} generated successfully!")
            except Exception as e:
                print(f"❌ Error generating {name}: {e}")
            print("-" * 40)
        
        print("🎉 All clean visual diagrams generated successfully!")
        print(f"📁 Clean visual files created:")
        print("   - clean_system_architecture.png")
        print("   - framework_comparison_matrix.png") 
        print("   - architecture_patterns.png")
        print("   - performance_dashboard.png")

def main():
    """Generate all clean visual diagrams"""
    print("🚀 Clean Visual Framework Architecture Generator")
    print("=" * 50)
    
    # Generate clean visual diagrams
    generator = CleanVisualFrameworkDiagrams()
    generator.generate_all_clean_diagrams()

if __name__ == "__main__":
    main()
