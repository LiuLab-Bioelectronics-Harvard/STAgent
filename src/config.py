"""
Centralized configuration management for STAgent.
This module contains application-level constants and settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

# Get the base directory (src folder)
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Data paths
    data_path: str = str(DATA_DIR / "pancreas_processed_full.h5ad")
    plot_dir: str = str(BASE_DIR / "tmp" / "plots")
    db_path: str = str(DB_DIR / "chroma_squidpy_db")
    
    # History directories
    history_dirs: Dict[str, str] = None
    
    # Model configurations
    models: Dict[str, Dict] = None
    
    # Plot settings
    plot_width: int = 10
    plot_height: int = 6
    plot_dpi: int = 100
    plot_format: str = "png"
    
    # Processing limits
    max_recursion_steps: int = 200
    max_conversation_length: int = 200
    summary_trigger_interval: int = 10
    summary_min_messages: int = 5
    
    # File cleanup settings
    max_plot_files: int = 50
    plot_retention_days: int = 7
    
    def __post_init__(self):
        """Initialize computed fields after creation."""
        if self.history_dirs is None:
            self.history_dirs = {
                "openai": "conversation_histories/gpt",
                "anthropic": "conversation_histories/anthropic",
                "gemini": "conversation_histories/gemini",
            }
        
        if self.models is None:
            self.models = {
                "openai": {
                    "gpt-3.5-turbo": {
                        "temperature": 0,
                        "parallel_tool_calls": False
                    },
                    "gpt-4o": {
                        "temperature": 0,
                        "parallel_tool_calls": False
                    },
                    "gpt-5": {
                        "temperature": 1,
                        "parallel_tool_calls": False
                    }, 
                    "gpt-5.1-2025-11-13": {
                        "temperature": 1,
                        "parallel_tool_calls": False
                    }, 
                    "gpt-5.2": {
                        "temperature": 1,
                        "parallel_tool_calls": False
                    }, 
                    
                },
                "anthropic": {
                    "claude-sonnet-4-20250514": {
                        "temperature": 0,
                        "max_tokens": 8000
                    },
                    "claude_3_7_sonnet_20250219": {
                        "temperature": 0,
                        "max_tokens": 8000
                    },
                    "claude_3_5_sonnet_20241022": {
                        "temperature": 0,
                        "max_tokens": 8000
                    }
                },
                "gemini": {
                    "gemini-2.5-pro-exp-03-25": {
                        "temperature": 0,
                        "convert_system_message_to_human": True
                    },
                    "gemini-1.5-pro-latest": {
                        "temperature": 0,
                        "convert_system_message_to_human": True
                    }
                }
            }
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.plot_dir,
            self.db_path
        ]
        
        # Add history directories
        for hist_dir in self.history_dirs.values():
            directories.append(hist_dir)
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_history_dir(self, provider: str) -> str:
        """Get history directory for a specific provider."""
        return self.history_dirs.get(provider.lower(), "conversation_histories/default")
    
    def get_model_config(self, provider: str, model_name: str) -> Dict:
        """Get configuration for a specific model."""
        provider_models = self.models.get(provider.lower(), {})
        return provider_models.get(model_name, {})
    
    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a provider."""
        provider_models = self.models.get(provider.lower(), {})
        return list(provider_models.keys())


@dataclass 
class UIConfig:
    """UI-specific configuration."""
    
    # Streamlit settings
    page_title: str = "🤖 Spatial Transcriptomics Agent"
    page_icon: str = "🤖"
    layout: str = "wide"
    
    # Theme colors
    primary_color: str = "#FF5722"
    secondary_color: str = "#2196F3"
    background_color: str = "#FFFFFF"
    text_color: str = "#000000"
    
    # Provider display configs
    provider_configs: Dict[str, Dict[str, str]] = None
    
    # Chat styling
    user_message_bg: str = "#DCF8C6"
    ai_message_bg: str = "#E8F5E9"
    tool_message_bg: str = "#FFF3E0"
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.provider_configs is None:
            self.provider_configs = {
                "Anthropic": {
                    "icon": "🟣",
                    "color": "#9C27B0",
                    "hover_color": "#7B1FA2"
                },
                "OpenAI": {
                    "icon": "🟢",
                    "color": "#4CAF50",
                    "hover_color": "#388E3C"
                },
                "Gemini": {
                    "icon": "🔵",
                    "color": "#1E88E5",
                    "hover_color": "#1565C0"
                }
            }


@dataclass
class ToolConfig:
    """Tool-specific configuration."""
    
    # Search settings
    serp_api_key: Optional[str] = None
    google_search_results: int = 40
    search_language: str = "en"
    
    # Python REPL settings
    repl_timeout: int = 30
    max_output_length: int = 10000
    
    # Visualization settings
    default_figure_size: tuple = (10, 6)
    default_dpi: int = 100
    save_format: str = "png"
    bbox_inches: str = "tight"
    
    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    
    def __post_init__(self):
        """Initialize computed fields."""
        # Get API keys from environment
        self.serp_api_key = os.getenv("SERP_API_KEY")


# Global configuration instances
app_config = AppConfig()
ui_config = UIConfig()
tool_config = ToolConfig()


def get_config() -> AppConfig:
    """Get the global app configuration."""
    return app_config


def get_ui_config() -> UIConfig:
    """Get the global UI configuration."""
    return ui_config


def get_tool_config() -> ToolConfig:
    """Get the global tool configuration."""
    return tool_config


def update_config(**kwargs):
    """Update configuration values dynamically."""
    global app_config
    for key, value in kwargs.items():
        if hasattr(app_config, key):
            setattr(app_config, key, value)


def get_data_path() -> str:
    """Get the main data file path."""
    return app_config.data_path


def get_plot_dir() -> str:
    """Get the plot directory path."""
    return app_config.plot_dir


def get_db_path() -> str:
    """Get the database path."""
    return app_config.db_path


# Environment-specific overrides
def load_environment_config():
    """Load configuration from environment variables."""
    # Override data path if specified
    env_data_path = os.getenv("STAGENT_DATA_PATH")
    if env_data_path:
        app_config.data_path = env_data_path
    
    # Override plot directory if specified
    env_plot_dir = os.getenv("STAGENT_PLOT_DIR")
    if env_plot_dir:
        app_config.plot_dir = env_plot_dir
    
    # Override database path if specified
    env_db_path = os.getenv("STAGENT_DB_PATH")
    if env_db_path:
        app_config.db_path = env_db_path
    
    # Override recursion limit if specified
    env_recursion_limit = os.getenv("STAGENT_RECURSION_LIMIT")
    if env_recursion_limit:
        try:
            app_config.max_recursion_steps = int(env_recursion_limit)
        except ValueError:
            pass


# Load environment config on import
load_environment_config()
