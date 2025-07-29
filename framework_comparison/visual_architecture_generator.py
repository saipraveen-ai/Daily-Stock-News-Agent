#!/usr/bin/env python3
"""
Visual Framework Architecture Generator

Creates intuitive visual diagrams using matplotlib, networkx, and other visualization libraries
to represent framework architectures in a clear, visual way.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow, ConnectionPatch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import seaborn as sns
from matplotlib.sankey import Sankey

# Set style for better visuals
plt.style.use('default')
sns.set_palette("Set2")

class VisualFrameworkDiagrams:
    """Create intuitive visual diagrams for framework architectures"""
    
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

    def create_system_overview_visual(self):
        """Create a clear visual overview of the entire system"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Define component positions and sizes
        components = {
            # Input layer
            'youtube': {'pos': (2, 8), 'size': (2, 1), 'color': self.colors['input'], 'icon': '📺'},
            'channels': {'pos': (1, 6.5), 'size': (1.5, 0.8), 'color': self.colors['input'], 'icon': '📱'},
            'videos': {'pos': (3, 6.5), 'size': (1.5, 0.8), 'color': self.colors['input'], 'icon': '🎥'},
            
            # Framework layer (circular arrangement)
            'frameworks': {'pos': (8, 6), 'size': (6, 4), 'color': self.colors['framework'], 'icon': '🤖'},
            
            # Processing pipeline
            'download': {'pos': (2, 4), 'size': (1.8, 1), 'color': self.colors['processing'], 'icon': '⬇️'},
            'transcribe': {'pos': (5, 4), 'size': (1.8, 1), 'color': self.colors['processing'], 'icon': '🎤'},
            'analyze': {'pos': (8, 4), 'size': (1.8, 1), 'color': self.colors['processing'], 'icon': '📊'},
            'report': {'pos': (11, 4), 'size': (1.8, 1), 'color': self.colors['processing'], 'icon': '📄'},
            
            # Output layer
            'recommendations': {'pos': (2, 1.5), 'size': (2, 1), 'color': self.colors['output'], 'icon': '💡'},
            'analysis_out': {'pos': (6, 1.5), 'size': (2, 1), 'color': self.colors['output'], 'icon': '📈'},
            'reports': {'pos': (10, 1.5), 'size': (2, 1), 'color': self.colors['output'], 'icon': '📋'}
        }
        
        # Draw components
        for name, comp in components.items():
            if name == 'frameworks':
                # Special handling for frameworks circle
                circle = Circle(comp['pos'], 2.5, facecolor=comp['color'], 
                              edgecolor='black', linewidth=2, alpha=0.3)
                ax.add_patch(circle)
                ax.text(comp['pos'][0], comp['pos'][1]+0.3, comp['icon'] + ' 9 Frameworks', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
            else:
                # Regular rectangular components
                x, y = comp['pos']
                w, h = comp['size']
                
                rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                                    boxstyle="round,pad=0.1",
                                    facecolor=comp['color'],
                                    edgecolor='black',
                                    linewidth=2)
                ax.add_patch(rect)
                
                # Add icon and text
                ax.text(x, y+0.1, comp['icon'], ha='center', va='center', fontsize=20)
                ax.text(x, y-0.2, name.replace('_', ' ').title(), 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw framework icons in circle
        framework_names = list(self.framework_colors.keys())
        framework_icons = ['🤖', '🦜', '🤝', '🏗️', '🧠', '🎯', '🚀', '🔧', '🔍']
        angles = np.linspace(0, 2*np.pi, len(framework_names), endpoint=False)
        
        for i, (name, icon) in enumerate(zip(framework_names, framework_icons)):
            angle = angles[i]
            x = 8 + 2 * np.cos(angle)
            y = 6 + 2 * np.sin(angle)
            
            # Framework circle
            circle = Circle((x, y), 0.3, facecolor=self.framework_colors[name], 
                          edgecolor='black', linewidth=1, alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, icon, ha='center', va='center', fontsize=12)
        
        # Draw arrows for data flow
        arrows = [
            # Input to frameworks
            ((2, 7.5), (6, 6.5)),
            ((3.5, 6.5), (6, 6.5)),
            
            # Frameworks to processing
            ((6, 6), (2.9, 4.5)),
            ((7, 5), (5, 4.5)),
            ((9, 5), (8, 4.5)),
            ((10, 6), (11, 4.5)),
            
            # Processing pipeline
            ((3.8, 4), (4.2, 4)),
            ((6.8, 4), (7.2, 4)),
            ((9.8, 4), (10.2, 4)),
            
            # Processing to output
            ((2.5, 3.5), (2.5, 2.5)),
            ((8, 3.5), (7, 2.5)),
            ((11, 3.5), (11, 2.5))
        ]
        
        for (start, end) in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=3, color='#333333', alpha=0.7))
        
        # Add labels
        ax.text(8, 9, '📊 Daily Stock News Agent - System Architecture', 
               ha='center', va='center', fontsize=18, fontweight='bold')
        
        ax.text(2, 9, 'INPUT', ha='center', va='center', fontsize=14, 
               fontweight='bold', color='blue')
        ax.text(8, 8.5, 'FRAMEWORKS', ha='center', va='center', fontsize=14, 
               fontweight='bold', color='purple')
        ax.text(6.5, 5, 'PROCESSING PIPELINE', ha='center', va='center', fontsize=14, 
               fontweight='bold', color='green')
        ax.text(6.5, 0.5, 'OUTPUT', ha='center', va='center', fontsize=14, 
               fontweight='bold', color='orange')
        
        ax.set_xlim(-1, 15)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('system_architecture_visual.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()

    def create_framework_comparison_visual(self):
        """Create visual comparison of all frameworks"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        
        frameworks = {
            'OpenAI Assistants': {'pos': (2, 9), 'pattern': 'assistant', 'icon': '🤖'},
            'LangChain': {'pos': (6, 9), 'pattern': 'chain', 'icon': '🦜'},
            'CrewAI': {'pos': (10, 9), 'pattern': 'crew', 'icon': '🤝'},
            'AutoGen': {'pos': (14, 9), 'pattern': 'conversation', 'icon': '🏗️'},
            'LangGraph': {'pos': (2, 5.5), 'pattern': 'graph', 'icon': '🧠'},
            'PydanticAI': {'pos': (6, 5.5), 'pattern': 'typed', 'icon': '🎯'},
            'Swarm': {'pos': (10, 5.5), 'pattern': 'swarm', 'icon': '🚀'},
            'Semantic Kernel': {'pos': (14, 5.5), 'pattern': 'kernel', 'icon': '🔧'},
            'Haystack': {'pos': (8, 2), 'pattern': 'pipeline', 'icon': '🔍'}
        }
        
        patterns = {
            'assistant': self._draw_assistant_pattern,
            'chain': self._draw_chain_pattern,
            'crew': self._draw_crew_pattern,
            'conversation': self._draw_conversation_pattern,
            'graph': self._draw_graph_pattern,
            'typed': self._draw_typed_pattern,
            'swarm': self._draw_swarm_pattern,
            'kernel': self._draw_kernel_pattern,
            'pipeline': self._draw_pipeline_pattern
        }
        
        for name, info in frameworks.items():
            x, y = info['pos']
            
            # Framework container
            container = FancyBboxPatch((x-1.8, y-1.3), 3.6, 2.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor=self.framework_colors[name],
                                     edgecolor='black',
                                     linewidth=2,
                                     alpha=0.3)
            ax.add_patch(container)
            
            # Framework title
            ax.text(x, y+1, f"{info['icon']} {name}", ha='center', va='center', 
                   fontsize=12, fontweight='bold')
            
            # Draw pattern
            patterns[info['pattern']](ax, x, y)
        
        ax.set_xlim(-1, 17)
        ax.set_ylim(0, 11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('🎯 Framework Architecture Patterns Comparison', 
                    fontsize=20, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('framework_patterns_visual.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def _draw_assistant_pattern(self, ax, x, y):
        """Draw OpenAI Assistants pattern"""
        # Thread in center
        thread = Circle((x, y-0.3), 0.3, facecolor='lightblue', edgecolor='black')
        ax.add_patch(thread)
        ax.text(x, y-0.3, '🧵', ha='center', va='center', fontsize=12)
        
        # Assistants around thread
        assistants = [(x-0.7, y), (x+0.7, y), (x, y-0.8), (x, y+0.2)]
        for i, (ax_pos, ay_pos) in enumerate(assistants):
            assistant = Circle((ax_pos, ay_pos), 0.2, facecolor='orange', edgecolor='black')
            ax.add_patch(assistant)
            ax.text(ax_pos, ay_pos, '🤖', ha='center', va='center', fontsize=8)

    def _draw_chain_pattern(self, ax, x, y):
        """Draw LangChain sequential pattern"""
        # Chain of tools
        for i in range(4):
            tool_x = x - 1.2 + i * 0.8
            tool = FancyBboxPatch((tool_x-0.2, y-0.6), 0.4, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor='lightgreen',
                                edgecolor='black')
            ax.add_patch(tool)
            ax.text(tool_x, y-0.4, '🔧', ha='center', va='center', fontsize=10)
            
            if i < 3:
                ax.annotate('', xy=(tool_x+0.3, y-0.4), xytext=(tool_x+0.1, y-0.4),
                           arrowprops=dict(arrowstyle='->', lw=2))

    def _draw_crew_pattern(self, ax, x, y):
        """Draw CrewAI multi-agent pattern"""
        # Manager in center
        manager = Circle((x, y-0.1), 0.25, facecolor='red', edgecolor='black')
        ax.add_patch(manager)
        ax.text(x, y-0.1, '👨‍💼', ha='center', va='center', fontsize=10)
        
        # Agents around manager
        agents = [(x-0.8, y-0.5), (x+0.8, y-0.5), (x-0.4, y+0.3), (x+0.4, y+0.3)]
        for ax_pos, ay_pos in agents:
            agent = Circle((ax_pos, ay_pos), 0.2, facecolor='yellow', edgecolor='black')
            ax.add_patch(agent)
            ax.text(ax_pos, ay_pos, '👤', ha='center', va='center', fontsize=8)
            # Connect to manager
            ax.plot([x, ax_pos], [y-0.1, ay_pos], 'k-', lw=1)

    def _draw_conversation_pattern(self, ax, x, y):
        """Draw AutoGen conversation pattern"""
        # Agents in conversation
        agents = [(x-0.6, y), (x+0.6, y), (x, y-0.6)]
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        for i, ((ax_pos, ay_pos), color) in enumerate(zip(agents, colors)):
            agent = Circle((ax_pos, ay_pos), 0.25, facecolor=color, edgecolor='black')
            ax.add_patch(agent)
            ax.text(ax_pos, ay_pos, '💬', ha='center', va='center', fontsize=10)
        
        # Conversation lines
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                ax.plot([agents[i][0], agents[j][0]], 
                       [agents[i][1], agents[j][1]], 'k--', lw=1, alpha=0.5)

    def _draw_graph_pattern(self, ax, x, y):
        """Draw LangGraph state machine pattern"""
        # States
        states = [(x-0.8, y), (x, y+0.4), (x+0.8, y), (x, y-0.6)]
        state_names = ['📥', '⚙️', '✅', '📤']
        
        for (sx, sy), name in zip(states, state_names):
            state = FancyBboxPatch((sx-0.2, sy-0.15), 0.4, 0.3,
                                 boxstyle="round,pad=0.05",
                                 facecolor='lightpink',
                                 edgecolor='black')
            ax.add_patch(state)
            ax.text(sx, sy, name, ha='center', va='center', fontsize=12)
        
        # Transitions
        transitions = [(0, 1), (1, 2), (2, 3), (1, 0)]
        for start, end in transitions:
            sx, sy = states[start]
            ex, ey = states[end]
            ax.annotate('', xy=(ex-0.1, ey), xytext=(sx+0.1, sy),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

    def _draw_typed_pattern(self, ax, x, y):
        """Draw PydanticAI typed pattern"""
        # Input validation
        input_box = FancyBboxPatch((x-1.2, y+0.2), 0.8, 0.3,
                                 boxstyle="round,pad=0.05",
                                 facecolor='lightcyan',
                                 edgecolor='blue')
        ax.add_patch(input_box)
        ax.text(x-0.8, y+0.35, '📝 Input', ha='center', va='center', fontsize=8)
        
        # Type checker
        type_box = FancyBboxPatch((x-0.2, y-0.1), 0.4, 0.3,
                                boxstyle="round,pad=0.05",
                                facecolor='yellow',
                                edgecolor='orange')
        ax.add_patch(type_box)
        ax.text(x, y+0.05, '✓', ha='center', va='center', fontsize=12)
        
        # Output validation
        output_box = FancyBboxPatch((x+0.4, y+0.2), 0.8, 0.3,
                                  boxstyle="round,pad=0.05",
                                  facecolor='lightgreen',
                                  edgecolor='green')
        ax.add_patch(output_box)
        ax.text(x+0.8, y+0.35, '📊 Output', ha='center', va='center', fontsize=8)
        
        # Arrows
        ax.annotate('', xy=(x-0.4, y+0.05), xytext=(x-0.6, y+0.3),
                   arrowprops=dict(arrowstyle='->', lw=2))
        ax.annotate('', xy=(x+0.6, y+0.3), xytext=(x+0.4, y+0.05),
                   arrowprops=dict(arrowstyle='->', lw=2))

    def _draw_swarm_pattern(self, ax, x, y):
        """Draw Swarm handoff pattern"""
        # Agents with handoff arrows
        agents = [(x-0.8, y+0.2), (x-0.2, y+0.2), (x+0.4, y+0.2), (x+1, y+0.2)]
        
        for i, (ax_pos, ay_pos) in enumerate(agents):
            agent = Circle((ax_pos, ay_pos), 0.2, facecolor='orange', edgecolor='black')
            ax.add_patch(agent)
            ax.text(ax_pos, ay_pos, '🐝', ha='center', va='center', fontsize=10)
            
            if i < len(agents) - 1:
                ax.annotate('', xy=(agents[i+1][0]-0.2, ay_pos), 
                           xytext=(ax_pos+0.2, ay_pos),
                           arrowprops=dict(arrowstyle='->', lw=2, color='orange'))

    def _draw_kernel_pattern(self, ax, x, y):
        """Draw Semantic Kernel plugin pattern"""
        # Kernel center
        kernel = Circle((x, y), 0.3, facecolor='purple', edgecolor='black')
        ax.add_patch(kernel)
        ax.text(x, y, '🔧', ha='center', va='center', fontsize=12)
        
        # Plugins around kernel
        plugins = [(x-0.8, y+0.4), (x+0.8, y+0.4), (x-0.8, y-0.4), (x+0.8, y-0.4)]
        plugin_icons = ['📹', '🎤', '📊', '📄']
        
        for (px, py), icon in zip(plugins, plugin_icons):
            plugin = FancyBboxPatch((px-0.15, py-0.1), 0.3, 0.2,
                                  boxstyle="round,pad=0.02",
                                  facecolor='lavender',
                                  edgecolor='purple')
            ax.add_patch(plugin)
            ax.text(px, py, icon, ha='center', va='center', fontsize=10)
            ax.plot([x, px], [y, py], 'purple', lw=1)

    def _draw_pipeline_pattern(self, ax, x, y):
        """Draw Haystack pipeline pattern"""
        # Pipeline components
        components = [
            (x-1.2, y+0.3, '📹'),
            (x-0.4, y+0.3, '🎤'),
            (x+0.4, y+0.3, '🔍'),
            (x+1.2, y+0.3, '📄')
        ]
        
        for i, (cx, cy, icon) in enumerate(components):
            comp = FancyBboxPatch((cx-0.2, cy-0.15), 0.4, 0.3,
                                boxstyle="round,pad=0.05",
                                facecolor='lightsteelblue',
                                edgecolor='steelblue')
            ax.add_patch(comp)
            ax.text(cx, cy, icon, ha='center', va='center', fontsize=12)
            
            if i < len(components) - 1:
                next_cx = components[i+1][0]
                ax.annotate('', xy=(next_cx-0.2, cy), xytext=(cx+0.2, cy),
                           arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'))

    def create_data_flow_visual(self):
        """Create visual data flow diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Data flow stages
        stages = {
            'input': {'pos': (2, 7), 'items': ['📺 YouTube', '📱 Channels', '🎥 Videos']},
            'processing': {'pos': (8, 7), 'items': ['⬇️ Download', '🎤 Transcribe', '📊 Analyze', '📄 Report']},
            'frameworks': {'pos': (2, 4), 'items': ['🤖 Assistants', '🦜 LangChain', '🤝 CrewAI']},
            'output': {'pos': (14, 7), 'items': ['💡 Recommendations', '📈 Analysis', '📋 Reports']}
        }
        
        # Draw stages
        for stage_name, stage_info in stages.items():
            x, y = stage_info['pos']
            items = stage_info['items']
            
            if stage_name == 'processing':
                # Horizontal pipeline
                for i, item in enumerate(items):
                    item_x = x - 2 + i * 1.3
                    
                    rect = FancyBboxPatch((item_x-0.5, y-0.3), 1, 0.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor=self.colors['processing'],
                                        edgecolor='black',
                                        linewidth=2)
                    ax.add_patch(rect)
                    ax.text(item_x, y, item, ha='center', va='center', 
                           fontsize=10, fontweight='bold')
                    
                    if i < len(items) - 1:
                        ax.annotate('', xy=(item_x+0.6, y), xytext=(item_x+0.4, y),
                                   arrowprops=dict(arrowstyle='->', lw=3, color='green'))
            
            elif stage_name == 'frameworks':
                # Vertical stack
                for i, item in enumerate(items):
                    item_y = y + 1 - i * 0.8
                    
                    rect = FancyBboxPatch((x-0.8, item_y-0.25), 1.6, 0.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor=self.colors['framework'],
                                        edgecolor='black',
                                        linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x, item_y, item, ha='center', va='center', 
                           fontsize=10, fontweight='bold')
            
            else:
                # Input and output - vertical stack
                for i, item in enumerate(items):
                    item_y = y + 1 - i * 0.7
                    color = self.colors['input'] if stage_name == 'input' else self.colors['output']
                    
                    rect = FancyBboxPatch((x-0.8, item_y-0.25), 1.6, 0.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color,
                                        edgecolor='black',
                                        linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x, item_y, item, ha='center', va='center', 
                           fontsize=10, fontweight='bold')
        
        # Draw main flow arrows
        main_flows = [
            ((3.6, 7), (5.5, 7)),  # Input to processing
            ((11.5, 7), (12.4, 7)),  # Processing to output
            ((2, 5.5), (5.5, 7.5)),  # Frameworks to processing
        ]
        
        for (start, end) in main_flows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=4, color='darkblue', alpha=0.7))
        
        # Add labels
        ax.text(8, 9, '🔄 Data Flow Through System', ha='center', va='center', 
               fontsize=18, fontweight='bold')
        
        ax.text(2, 8.5, 'INPUT SOURCES', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='blue')
        ax.text(8, 8.5, 'PROCESSING PIPELINE', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='green')
        ax.text(2, 2.5, 'FRAMEWORKS\n(9 Different Approaches)', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='purple')
        ax.text(14, 8.5, 'OUTPUTS', ha='center', va='center', 
               fontsize=12, fontweight='bold', color='orange')
        
        ax.set_xlim(0, 16)
        ax.set_ylim(1, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('data_flow_visual.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def create_setup_complexity_visual(self):
        """Create visual setup complexity comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Setup time visualization
        frameworks = list(self.framework_colors.keys())
        setup_times = [2, 3, 5, 3.5, 4.5, 3, 4, 7.5, 8.5]  # minutes
        complexity_scores = [3, 7, 5, 6, 8, 4, 3, 7, 9]  # 1-10 scale
        
        # Chart 1: Setup time with visual bars
        y_pos = np.arange(len(frameworks))
        bars = ax1.barh(y_pos, setup_times, 
                       color=[self.framework_colors[fw] for fw in frameworks],
                       alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add medals and time labels
        medals = ['🥇', '🥈', '🥉'] + [''] * 6
        sorted_indices = np.argsort(setup_times)
        
        for i, (bar, time, idx) in enumerate(zip(bars, setup_times, y_pos)):
            # Time label
            ax1.text(time + 0.2, idx, f'{time} min', 
                    va='center', fontweight='bold', fontsize=11)
            
            # Medal for top 3
            rank = list(sorted_indices).index(i)
            if rank < 3:
                ax1.text(time + 1, idx, medals[rank], 
                        va='center', fontsize=16)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(frameworks, fontweight='bold')
        ax1.set_xlabel('Setup Time (minutes)', fontweight='bold', fontsize=14)
        ax1.set_title('⏱️ Framework Setup Time Ranking', fontweight='bold', fontsize=16)
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_xlim(0, max(setup_times) + 2)
        
        # Chart 2: Complexity vs Features bubble chart
        features = [9, 10, 8, 8, 9, 7, 5, 7, 10]  # feature richness
        
        # Create bubble sizes based on popularity/adoption
        popularity = [9, 10, 7, 7, 6, 6, 5, 6, 8]
        bubble_sizes = [(p * 50) for p in popularity]
        
        scatter = ax2.scatter(complexity_scores, features, 
                            s=bubble_sizes,
                            c=[self.framework_colors[fw] for fw in frameworks],
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add framework labels
        for i, fw in enumerate(frameworks):
            ax2.annotate(fw, (complexity_scores[i], features[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', alpha=0.8))
        
        # Add quadrant lines
        ax2.axhline(y=7.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax2.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        # Quadrant labels
        quadrants = [
            (3, 9, 'Sweet Spot\n(Simple + Rich)', 'lightgreen'),
            (8, 9, 'Feature Rich\n(Complex)', 'orange'),
            (3, 5, 'Basic\n(Simple)', 'lightblue'),
            (8, 5, 'Over-engineered\n(Complex + Limited)', 'lightcoral')
        ]
        
        for x, y, label, color in quadrants:
            ax2.text(x, y, label, ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", 
                            facecolor=color, alpha=0.7))
        
        ax2.set_xlabel('Complexity Score (1-10)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Feature Richness (1-10)', fontweight='bold', fontsize=14)
        ax2.set_title('🎯 Complexity vs Features (Bubble = Popularity)', 
                     fontweight='bold', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, 10)
        ax2.set_ylim(3, 11)
        
        plt.tight_layout()
        plt.savefig('setup_complexity_visual.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def create_architecture_network_graph(self):
        """Create network graph showing framework relationships"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (frameworks and their characteristics)
        frameworks = {
            'OpenAI Assistants': {'type': 'simple', 'category': 'official'},
            'LangChain': {'type': 'complex', 'category': 'chains'},
            'CrewAI': {'type': 'medium', 'category': 'multi-agent'},
            'AutoGen': {'type': 'medium', 'category': 'multi-agent'},
            'LangGraph': {'type': 'complex', 'category': 'state'},
            'PydanticAI': {'type': 'simple', 'category': 'typed'},
            'Swarm': {'type': 'simple', 'category': 'multi-agent'},
            'Semantic Kernel': {'type': 'complex', 'category': 'enterprise'},
            'Haystack': {'type': 'complex', 'category': 'nlp'}
        }
        
        # Add category nodes
        categories = {
            'Multi-Agent': ['CrewAI', 'AutoGen', 'Swarm'],
            'Sequential': ['LangChain', 'Haystack'],
            'State-Based': ['LangGraph'],
            'Type-Safe': ['PydanticAI'],
            'Enterprise': ['Semantic Kernel'],
            'Official': ['OpenAI Assistants']
        }
        
        # Add all nodes
        for fw in frameworks.keys():
            G.add_node(fw, node_type='framework')
        
        for category in categories.keys():
            G.add_node(category, node_type='category')
        
        # Add edges based on relationships
        for category, framework_list in categories.items():
            for fw in framework_list:
                if fw in frameworks:
                    G.add_edge(category, fw)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw category nodes (larger, different color)
        category_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'category']
        nx.draw_networkx_nodes(G, pos, nodelist=category_nodes,
                             node_color='lightcoral', node_size=2000,
                             alpha=0.8, edgecolors='black', linewidths=2)
        
        # Draw framework nodes
        framework_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'framework']
        node_colors = [self.framework_colors[n] for n in framework_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=framework_nodes,
                             node_color=node_colors, node_size=1500,
                             alpha=0.9, edgecolors='black', linewidths=2)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.6, width=2, edge_color='gray')
        
        # Add labels
        labels = {}
        for node in G.nodes():
            if node in framework_nodes:
                # Add emoji to framework names
                emoji_map = {
                    'OpenAI Assistants': '🤖',
                    'LangChain': '🦜',
                    'CrewAI': '🤝',
                    'AutoGen': '🏗️',
                    'LangGraph': '🧠',
                    'PydanticAI': '🎯',
                    'Swarm': '🚀',
                    'Semantic Kernel': '🔧',
                    'Haystack': '🔍'
                }
                labels[node] = f"{emoji_map.get(node, '')} {node}"
            else:
                labels[node] = f"📁 {node}"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        ax.set_title('🕸️ Framework Relationship Network', fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='lightcoral', s=200, alpha=0.8, edgecolors='black', 
                       label='Categories'),
            plt.scatter([], [], c='lightblue', s=150, alpha=0.9, edgecolors='black', 
                       label='Frameworks')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('architecture_network_graph.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()

    def generate_all_visual_diagrams(self):
        """Generate all visual diagrams"""
        print("🎨 Generating Visual Framework Diagrams...")
        print("=" * 60)
        
        diagrams = [
            ("System Architecture Overview", self.create_system_overview_visual),
            ("Framework Patterns Comparison", self.create_framework_comparison_visual),
            ("Data Flow Visualization", self.create_data_flow_visual),
            ("Setup & Complexity Analysis", self.create_setup_complexity_visual),
            ("Architecture Network Graph", self.create_architecture_network_graph)
        ]
        
        for name, func in diagrams:
            print(f"🖼️  Generating {name}...")
            try:
                func()
                print(f"✅ {name} generated successfully!")
            except Exception as e:
                print(f"❌ Error generating {name}: {e}")
            print("-" * 40)
        
        print("🎉 All visual diagrams generated successfully!")
        print(f"📁 Visual files created:")
        print("   - system_architecture_visual.png")
        print("   - framework_patterns_visual.png") 
        print("   - data_flow_visual.png")
        print("   - setup_complexity_visual.png")
        print("   - architecture_network_graph.png")

def main():
    """Generate all visual diagrams"""
    print("🚀 Visual Framework Architecture Generator")
    print("=" * 50)
    
    # Check dependencies
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        print("✅ All required libraries available")
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("💡 Install with: pip install matplotlib networkx seaborn")
        return
    
    # Generate visual diagrams
    generator = VisualFrameworkDiagrams()
    generator.generate_all_visual_diagrams()

if __name__ == "__main__":
    main()
