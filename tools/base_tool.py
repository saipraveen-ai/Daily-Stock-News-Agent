"""
Base Tool Interface for Daily Stock News Agent

This module provides the foundation for all tools in the modular tool ecosystem.
Similar to the Gen-AI project's tool architecture, this ensures consistency
and extensibility across all processing tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools available in the system"""
    YOUTUBE = "youtube"
    TRANSCRIPTION = "transcription"
    ANALYSIS = "analysis" 
    GENERATION = "generation"
    STORAGE = "storage"
    NOTIFICATION = "notification"


class ToolPriority(Enum):
    """Priority levels for tool execution"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class ToolResult:
    """Standardized result format for all tools"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.metadata['timestamp'] = datetime.now().isoformat()


@dataclass 
class ToolConfig:
    """Configuration settings for tools"""
    name: str
    category: ToolCategory
    priority: ToolPriority
    enabled: bool = True
    timeout: int = 300  # 5 minutes default
    retry_count: int = 3
    dependencies: List[str] = None
    settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.settings is None:
            self.settings = {}


class BaseTool(ABC):
    """
    Abstract base class for all tools in the Daily Stock News Agent system.
    
    This provides a consistent interface and common functionality that all
    tools must implement, ensuring modularity and extensibility.
    """
    
    def __init__(self, config: ToolConfig):
        """
        Initialize the tool with configuration.
        
        Args:
            config: ToolConfig object containing tool settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._is_initialized = False
        
    @property
    def name(self) -> str:
        """Get the tool name"""
        return self.config.name
        
    @property
    def category(self) -> ToolCategory:
        """Get the tool category"""
        return self.config.category
        
    @property 
    def priority(self) -> ToolPriority:
        """Get the tool priority"""
        return self.config.priority
        
    @property
    def is_enabled(self) -> bool:
        """Check if tool is enabled"""
        return self.config.enabled
        
    @abstractmethod
    async def initialize(self) -> ToolResult:
        """
        Initialize the tool with any required setup.
        
        Returns:
            ToolResult indicating success/failure of initialization
        """
        pass
        
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the main tool functionality.
        
        Args:
            **kwargs: Tool-specific arguments
            
        Returns:
            ToolResult containing execution results
        """
        pass
        
    @abstractmethod
    async def cleanup(self) -> ToolResult:
        """
        Clean up any resources used by the tool.
        
        Returns:
            ToolResult indicating success/failure of cleanup
        """
        pass
        
    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate input parameters for the tool.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            bool: True if inputs are valid, False otherwise
        """
        pass
        
    async def health_check(self) -> ToolResult:
        """
        Perform a health check on the tool.
        
        Returns:
            ToolResult indicating tool health status
        """
        try:
            # Basic health check - can be overridden by specific tools
            if not self._is_initialized:
                return ToolResult(
                    success=False,
                    error_message=f"Tool {self.name} is not initialized"
                )
                
            return ToolResult(
                success=True,
                data={"status": "healthy", "tool": self.name}
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed for {self.name}: {e}")
            return ToolResult(
                success=False,
                error_message=f"Health check failed: {str(e)}"
            )
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the tool.
        
        Returns:
            Dict containing tool information
        """
        return {
            "name": self.name,
            "category": self.category.value,
            "priority": self.priority.value,
            "enabled": self.is_enabled,
            "initialized": self._is_initialized,
            "dependencies": self.config.dependencies,
            "timeout": self.config.timeout,
            "retry_count": self.config.retry_count
        }


class ToolRegistry:
    """
    Registry for managing all available tools in the system.
    
    This provides a centralized way to register, discover, and manage tools,
    similar to the Gen-AI project's tool management system.
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[ToolCategory, List[BaseTool]] = {
            category: [] for category in ToolCategory
        }
        self.logger = logging.getLogger("ToolRegistry")
        
    def register(self, tool: BaseTool) -> bool:
        """
        Register a tool in the registry.
        
        Args:
            tool: BaseTool instance to register
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            if tool.name in self._tools:
                self.logger.warning(f"Tool {tool.name} is already registered. Overwriting.")
                
            self._tools[tool.name] = tool
            self._categories[tool.category].append(tool)
            
            self.logger.info(f"Successfully registered tool: {tool.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool.name}: {e}")
            return False
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            BaseTool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: ToolCategory to filter by
            
        Returns:
            List of BaseTool instances in the category
        """
        return self._categories.get(category, [])
    
    def get_enabled_tools(self) -> List[BaseTool]:
        """
        Get all enabled tools.
        
        Returns:
            List of enabled BaseTool instances
        """
        return [tool for tool in self._tools.values() if tool.is_enabled]
    
    def get_tools_by_priority(self, priority: ToolPriority) -> List[BaseTool]:
        """
        Get all tools with a specific priority.
        
        Args:
            priority: ToolPriority to filter by
            
        Returns:
            List of BaseTool instances with the specified priority
        """
        return [tool for tool in self._tools.values() if tool.priority == priority]
    
    def list_tools(self) -> List[str]:
        """
        Get a list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the tool registry.
        
        Returns:
            Dict containing registry statistics and tool information
        """
        total_tools = len(self._tools)
        enabled_tools = len(self.get_enabled_tools())
        
        category_counts = {
            category.value: len(tools) 
            for category, tools in self._categories.items()
        }
        
        priority_counts = {
            priority.value: len(self.get_tools_by_priority(priority))
            for priority in ToolPriority
        }
        
        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": total_tools - enabled_tools,
            "category_distribution": category_counts,
            "priority_distribution": priority_counts,
            "tools": [tool.get_info() for tool in self._tools.values()]
        }


# Global tool registry instance
tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> bool:
    """
    Convenience function to register a tool with the global registry.
    
    Args:
        tool: BaseTool instance to register
        
    Returns:
        bool: True if registration successful, False otherwise
    """
    return tool_registry.register(tool)


def get_tool(name: str) -> Optional[BaseTool]:
    """
    Convenience function to get a tool from the global registry.
    
    Args:
        name: Name of the tool to retrieve
        
    Returns:
        BaseTool instance or None if not found
    """
    return tool_registry.get_tool(name)


def get_tools_by_category(category: ToolCategory) -> List[BaseTool]:
    """
    Convenience function to get tools by category from the global registry.
    
    Args:
        category: ToolCategory to filter by
        
    Returns:
        List of BaseTool instances in the category
    """
    return tool_registry.get_tools_by_category(category)
