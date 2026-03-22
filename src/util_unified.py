# util_unified.py - Unified utility functions for all LLM providers

import os
import json
from typing import Optional

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - allow non-Streamlit/batch contexts
    st = None  # type: ignore
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Tuple, Union
from pydantic import BaseModel, Field
from config import get_config, get_ui_config

# Directory for temporary plot images using centralized config
config = get_config()
plot_dir = config.plot_dir
os.makedirs(plot_dir, exist_ok=True)


def display_message(content: Union[str, dict, list], sender: str = "assistant"):
    """
    Displays a message from the user or assistant with different styling.
    Supports displaying both text and image URLs for the user.
    Unified version that handles both OpenAI and Anthropic content formats.
    """
    # In non-Streamlit contexts (batch / benchmarks), UI rendering is a no-op.
    if st is None:
        return
    if sender == "user":
        if isinstance(content, str):
            # Display plain text message from user
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <div style="display: inline-block; background-color: #DCF8C6; color: #000; padding: 10px; border-radius: 15px; margin: 5px; max-width: 60%; text-align: left;">
                        <p style="margin: 0;">{content}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif isinstance(content, dict):
            # Check if the content has both text and image URL
            if "text" in content:
                st.markdown(
                    f"""
                    <div style="text-align: right;">
                        <div style="display: inline-block; background-color: #DCF8C6; color: #000; padding: 10px; border-radius: 15px; margin: 5px; max-width: 60%; text-align: left;">
                            <p style="margin: 0;">{content["text"]}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if "url" in content:
                st.image(content["url"], caption="User Image", use_container_width=True)
    else:
        # Display assistant's message - unified handling for both providers
        if isinstance(content, str):
            # Handle plain text content (common for both providers)
            modified_content = content.replace("\\(", "$").replace("\\)", "$")
            modified_content = modified_content.replace("\\[", "$$").replace("\\]", "$$")
            st.markdown(modified_content)
        elif isinstance(content, list):
            # Handle list content (more common with Anthropic)
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        # Process LaTeX-style text
                        modified_text = item["text"].replace("\\(", "$").replace("\\)", "$")
                        modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")
                        st.markdown(modified_text)
                    elif "url" in item:
                        st.image(item["url"], caption="Assistant Image", use_container_width=True)
                elif isinstance(item, str):
                    # Handle plain text items in the list
                    modified_content = item.replace("\\(", "$").replace("\\)", "$")
                    modified_content = modified_content.replace("\\[", "$$").replace("\\]", "$$")
                    st.markdown(modified_content)
        elif isinstance(content, dict):
            # Handle dictionary content
            if "text" in content:
                modified_text = content["text"].replace("\\(", "$").replace("\\)", "$")
                modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")
                st.markdown(modified_text)
            if "url" in content:
                st.image(content["url"], caption="Assistant Image", use_container_width=True)
        else:
            # Handle unexpected content type
            st.error(f"Unsupported content format from the assistant: {type(content)}")


def render_conversation_history(messages: List[BaseMessage]):
    """
    Renders conversation history from a list of messages, handling multiple tool calls.
    Unified version that works with both OpenAI and Anthropic messages.
    """
    # In non-Streamlit contexts (batch / benchmarks), UI rendering is a no-op.
    if st is None:
        return
    tool_input_map = {}  # Map to track tool_call_id to tool_input
    
    for entry in messages:
        # Skip if the message has name "image_assistant"
        if hasattr(entry, "name") and entry.name == "image_assistant":
            continue
        # Skip internal synthetic tool-call messages (used to satisfy provider protocols)
        if hasattr(entry, "name") and entry.name == "internal_tool_call":
            continue
            
        if isinstance(entry, HumanMessage):
            # Check if entry.content is list or string and handle appropriately
            if isinstance(entry.content, list):
                for item in entry.content:
                    if isinstance(item, dict):
                        # Display text or image URL in dictionary format
                        if item["type"] == "text":
                            display_message(item["text"], sender="user")
                        elif item["type"] == "image_url":
                            display_message({"url": item["image_url"]["url"]}, sender="user")
                    elif isinstance(item, str):
                        # Display plain text if it's a string
                        display_message(item, sender="user")
            elif isinstance(entry.content, str):
                # Display single string content
                display_message(entry.content, sender="user")

        elif isinstance(entry, AIMessage):
            display_message(entry.content, sender="assistant")
            
            # Handle tool calls in AIMessage
            if entry.tool_calls:
                tool_calls = entry.tool_calls
                for tool_call in tool_calls:
                    try:
                        arguments_json = tool_call.get('args', '{}')
                        tool_input = arguments_json
                        # Normalize tool args so the UI can show clean code (e.g., python_repl_tool/query).
                        # LangChain may provide args as dict OR JSON string.
                        if isinstance(tool_input, str):
                            try:
                                tool_input = json.loads(tool_input)
                            except Exception:
                                pass
                        tool_call_id = tool_call.get("id")
                        if tool_call_id:
                            tool_input_map[tool_call_id] = tool_input
                    except json.JSONDecodeError:
                        tool_input_map[tool_call.get("id", "unknown")] = "Error decoding tool input."

        elif isinstance(entry, ToolMessage):
            display_tool_message(entry, tool_input_map)


def display_tool_message(entry: ToolMessage, tool_input_map: dict):
    """Display a tool message with the corresponding tool input."""
    if st is None:
        return
    tool_output = entry.content
    tool_call_id = getattr(entry, "tool_call_id", None)
    tool_input = tool_input_map.get(tool_call_id, "No matching tool input found")

    artifacts = getattr(entry, "artifact", [])
    # Auto-expand tool calls when plots/artifacts exist or when this is a scholar search.
    auto_expand = bool(artifacts) or (getattr(entry, "name", "") == "google_scholar_search")
    with st.expander(f"Tool Call: {entry.name}", expanded=auto_expand):
        if isinstance(tool_input, dict) and 'query' in tool_input:
            st.code(tool_input['query'], language="python")
        else:
            st.code(tool_input or "No tool input available", language="python")
        st.write("**Tool Output:**")
        st.code(tool_output)
        
        # Handle artifacts if they exist
        if artifacts:
            st.write("**Generated Artifacts (e.g., Plots):**")
            for rel_path in artifacts:
                if rel_path.endswith(".png"):
                    # Convert relative path to absolute
                    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                    if os.path.exists(abs_path):
                        st.image(abs_path, caption="Generated Plot")
                    else:
                        st.write(f"Error: Plot file not found at {rel_path}")


# Pydantic model for structured output
class ConversationSummary(BaseModel):
    """Structure for conversation title and summary."""
    title: str = Field(description="The title of the conversation")
    summary: str = Field(description="A concise summary of the conversation's main points")


def get_conversation_summary(messages: List[BaseMessage], provider_type: str = "openai") -> Tuple[str, str]:
    """
    Get conversation title and summary using the appropriate provider.
    
    Args:
        messages: List of conversation messages
        provider_type: One of "openai", "anthropic", or "gemini"
    
    Returns:
        Tuple of (title, summary)
    """
    # Create provider-specific LLM
    normalized_provider = provider_type.lower()
    if normalized_provider == "anthropic":
        llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0)
    elif normalized_provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True,
        )
    else:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("msgs"),
        ("human", "Given the above messages between user and AI agent, return a title and concise summary of the conversation"),
    ])
    
    # Configure the structured output model
    structured_llm = llm.with_structured_output(ConversationSummary)
    summarized_chain = prompt_template | structured_llm
    
    # Invoke the chain with the messages and retrieve the response
    try:
        response = summarized_chain.invoke({"msgs": messages})
        return response.title, response.summary
    except Exception as e:
        # Fallback to default values if summarization fails
        return "Conversation Summary", f"Failed to generate summary: {str(e)}"
