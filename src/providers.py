"""
Provider abstraction layer for LLM providers.
This module eliminates code duplication across the supported model providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.base import BaseLanguageModel
import streamlit as st
import os
from config import get_config, get_ui_config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self._create_model()
        
    @abstractmethod
    def _create_model(self) -> BaseLanguageModel:
        """Create the specific LLM model instance."""
        pass
    
    @abstractmethod
    def get_history_dir(self) -> str:
        """Get the conversation history directory for this provider."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name for display purposes."""
        pass
    
    @abstractmethod
    def get_display_config(self) -> Dict[str, str]:
        """Get display configuration (icon, colors) for UI."""
        pass
    
    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate that the required API key is available."""
        pass
    
    @abstractmethod
    def setup_api_key(self) -> Optional[str]:
        """Setup API key through Streamlit interface if needed."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def _create_model(self) -> BaseLanguageModel:
        config = get_config()
        model_config = config.get_model_config("openai", self.model_name)
        return ChatOpenAI(
            model=self.model_name,
            temperature=model_config.get("temperature", 0)
        )
    
    def get_history_dir(self) -> str:
        return get_config().get_history_dir("openai")
    
    def get_available_models(self) -> List[str]:
        return get_config().get_available_models("openai")
    
    def get_provider_name(self) -> str:
        return "OpenAI"
    
    def get_display_config(self) -> Dict[str, str]:
        ui_config = get_ui_config()
        return ui_config.provider_configs.get("OpenAI", {
            "icon": "🟢",
            "color": "#2196F3",
            "hover_color": "#1976D2"
        })
    
    def validate_api_key(self) -> bool:
        return bool(os.getenv('OPENAI_API_KEY'))
    
    def setup_api_key(self) -> Optional[str]:
        if not self.validate_api_key():
            st.sidebar.markdown("""
                <div class="api-key-setup">
                    <h3>🔑 OpenAI API Key Setup</h3>
                </div>
            """, unsafe_allow_html=True)
            api_key = st.sidebar.text_input(
                label="OpenAI API Key", 
                type="password", 
                label_visibility="collapsed"
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                return api_key
            else:
                st.info("Please enter your OpenAI API Key in the sidebar.")
                st.stop()
        return os.getenv('OPENAI_API_KEY')


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""
    
    def _create_model(self) -> BaseLanguageModel:
        config = get_config()
        model_config = config.get_model_config("anthropic", self.model_name)
        return ChatAnthropic(
            model=self.model_name,
            temperature=model_config.get("temperature", 0),
            max_tokens=model_config.get("max_tokens", 8000)
        )
    
    def get_history_dir(self) -> str:
        return get_config().get_history_dir("anthropic")
    
    def get_available_models(self) -> List[str]:
        return get_config().get_available_models("anthropic")
    
    def get_provider_name(self) -> str:
        return "Anthropic"
    
    def get_display_config(self) -> Dict[str, str]:
        ui_config = get_ui_config()
        return ui_config.provider_configs.get("Anthropic", {
            "icon": "🟣(Recommended)",
            "color": "#FF5722",
            "hover_color": "#E64A19"
        })
    
    def validate_api_key(self) -> bool:
        return bool(os.getenv('ANTHROPIC_API_KEY'))
    
    def setup_api_key(self) -> Optional[str]:
        if not self.validate_api_key():
            st.sidebar.header("Anthropic API Key Setup")
            api_key = st.sidebar.text_input(
                label="Anthropic API Key", 
                type="password", 
                label_visibility="collapsed"
            )
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
                return api_key
            else:
                st.info("Please enter your Anthropic API Key in the sidebar.")
                st.stop()
        return os.getenv('ANTHROPIC_API_KEY')


class GeminiProvider(LLMProvider):
    """Google Gemini provider implementation (standard API)."""
    
    def _create_model(self) -> BaseLanguageModel:
        config = get_config()
        model_config = config.get_model_config("gemini", self.model_name)
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=model_config.get("temperature", 0),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            convert_system_message_to_human=model_config.get("convert_system_message_to_human", True)
        )
    
    def get_history_dir(self) -> str:
        return get_config().get_history_dir("gemini")
    
    def get_available_models(self) -> List[str]:
        return get_config().get_available_models("gemini")
    
    def get_provider_name(self) -> str:
        return "Gemini"
    
    def get_display_config(self) -> Dict[str, str]:
        ui_config = get_ui_config()
        return ui_config.provider_configs.get("Gemini", {
            "icon": "🔷",
            "color": "#4285F4",
            "hover_color": "#3367D6"
        })
    
    def validate_api_key(self) -> bool:
        return bool(os.getenv('GOOGLE_API_KEY'))
    
    def setup_api_key(self) -> Optional[str]:
        if not self.validate_api_key():
            st.sidebar.markdown("""
                <div class="api-key-setup">
                    <h3>🔑 Google API Key Setup</h3>
                </div>
            """, unsafe_allow_html=True)
            api_key = st.sidebar.text_input(
                label="Google API Key", 
                type="password", 
                label_visibility="collapsed"
            )
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                return api_key
            else:
                st.info("Please enter your Google API Key in the sidebar.")
                st.stop()
        return os.getenv('GOOGLE_API_KEY')


class ProviderFactory:
    """Factory class for creating LLM providers."""
    
    _providers = {
        "OpenAI": OpenAIProvider,
        "Anthropic": AnthropicProvider,
        "Gemini": GeminiProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, model_name: str) -> LLMProvider:
        """Create a provider instance."""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = cls._providers[provider_name]
        return provider_class(model_name)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider names."""
        return list(cls._providers.keys())
    
    @classmethod
    def get_provider_configs(cls) -> Dict[str, Dict[str, str]]:
        """Get display configurations for all providers."""
        from config import get_ui_config
        ui_config = get_ui_config()
        return ui_config.provider_configs


class ProviderContext:
    """Context manager for provider-specific operations."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.history_dir = provider.get_history_dir()
        
        # Ensure history directory exists
        os.makedirs(self.history_dir, exist_ok=True)
    
    def get_graph_function(self) -> Callable:
        """Get the unified graph function for all providers."""
        from graph_unified import invoke_our_graph
        return invoke_our_graph
    
    def get_utility_functions(self) -> Dict[str, Callable]:
        """Get unified utility functions for all providers."""
        from util_unified import (
            display_message, 
            render_conversation_history, 
            get_conversation_summary
        )
        
        return {
            'display_message': display_message,
            'render_conversation_history': render_conversation_history,
            'get_conversation_summary': get_conversation_summary
        }
