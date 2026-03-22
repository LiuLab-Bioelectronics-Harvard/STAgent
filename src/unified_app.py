import base64
import json
import os
from pathlib import Path
import re
import streamlit as st
from dotenv import load_dotenv
from uuid import uuid4
from conflict_log import start_conflict_session, get_log as get_conflict_log, get_conflict_path
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from providers import ProviderFactory, ProviderContext
from graph_unified import invoke_our_graph
from util_unified import display_message, render_conversation_history, get_conversation_summary
from speech_to_text import input_from_mic, convert_text_to_speech
from datetime import datetime
from prompt import system_prompt
from typing import Any
from trace_log import set_current_trace_id

# Load environment variables
load_dotenv(Path(__file__).resolve().with_name(".env"))

def make_trace_id() -> str:
    # Timestamp-prefixed ID so conflict/history filenames are not "random-looking"
    return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:10]}"

# Initialize session state if not present
if "page" not in st.session_state:
    st.session_state["page"] = "OpenAI"

if "final_state" not in st.session_state:
    st.session_state["final_state"] = {
        "messages": [SystemMessage(content=system_prompt)]
    }
if "audio_transcription" not in st.session_state:
    st.session_state["audio_transcription"] = None
if "trace_id" not in st.session_state:
    st.session_state["trace_id"] = make_trace_id()
    set_current_trace_id(st.session_state["trace_id"])
    start_conflict_session(st.session_state["trace_id"])

# Add custom CSS with theme-aware styling
st.markdown("""
<style>
    /* Custom styling for the main title */
    .main-title {
        text-align: center;
        color: #FF5722;
        padding: 1rem 0;
        border-bottom: 2px solid #FF5722;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Provider selection styling */
    .provider-section {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        background-color: #FF5722;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .new-chat-button > button {
        background-color: #2196F3 !important;
        margin: 1rem 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.75rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .user-message {
        background-color: #FBE9E7;
        border-left: 4px solid #FF5722;
    }
    
    .ai-message {
        background-color: #E8F5E9;
        border-left: 4px solid #2196F3;
    }
    
    /* Form styling */
    .stForm {
        background-color: var(--background-color);
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }
    
    /* Image upload area styling */
    [data-testid="stFileUploader"] {
        background-color: var(--background-color);
        padding: 1.25rem;
        border-radius: 12px;
        border: 2px dashed #FF5722;
    }
    
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        .main-title {
            color: #FFAB91;
            border-bottom-color: #FFAB91;
        }
        
        .provider-section {
            background-color: #1E1E1E;
        }
        
        .user-message {
            background-color: #3E2723;
            border-left: 4px solid #FFAB91;
        }
        
        .ai-message {
            background-color: #1A237E;
            border-left: 4px solid #90CAF9;
        }
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 24px;
        border: 2px solid #FF5722;
        font-size: 16px;
    }
    
    /* Submit button hover effect */
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        background-color: #E64A19;
    }
    
    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #FBE9E7;
        transition: background-color 0.3s ease;
    }

    /* API key setup styling */
    .api-key-setup {
        background-color: var(--secondary-background-color);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        border: 1px solid #FF5722;
    }

    /* Audio instructions styling */
    .audio-instructions {
        background-color: var(--secondary-background-color);
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 14px;
        border: 1px solid #FF5722;
    }
    
    /* Main chat interface title styling */
    .chat-title {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 18px;
        background: linear-gradient(90deg, #FBE9E7, transparent);
        border-radius: 12px;
        margin-bottom: 24px;
    }
    
    .robot-icon {
        font-size: 28px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    .provider-name {
        color: #FF5722;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Set up Streamlit layout
st.markdown('<h1 class="main-title">🤖 Spatial Transcriptomics Agent</h1>', unsafe_allow_html=True)

# Navigation in sidebar with improved styling
st.sidebar.markdown('<div class="provider-section">', unsafe_allow_html=True)
st.sidebar.title("🎯 Navigation")

# Get provider configurations from factory
PROVIDER_CONFIGS = ProviderFactory.get_provider_configs()

# Update the provider selection
provider_options = [f"{PROVIDER_CONFIGS[p]['icon']} {p}" for p in ProviderFactory.get_available_providers()]
selected = st.sidebar.radio("Select LLM Provider Family", provider_options)
page = selected.split(" ")[-1]  # Extract provider name without emoji
st.session_state["page"] = page

# Create provider instance
current_provider = ProviderFactory.create_provider(page, "dummy_model")  # Model will be set later
provider_context = ProviderContext(current_provider)

# Set provider-specific variables
HISTORY_DIR = current_provider.get_history_dir()
available_models = current_provider.get_available_models()

# Add model selection with improved styling (default to gpt-5.2 when available)
default_model = "gpt-5.2"
default_index = available_models.index(default_model) if default_model in available_models else 0
selected_model = st.sidebar.selectbox(f"🔧 Select {page} Model:", available_models, index=default_index)

# Add New Chat button with custom styling
st.sidebar.markdown('<div class="new-chat-button">', unsafe_allow_html=True)
if st.sidebar.button("🔄 Start New Chat"):
    st.session_state["final_state"] = {
        "messages": [SystemMessage(content=system_prompt)]
    }
    st.session_state["trace_id"] = make_trace_id()
    set_current_trace_id(st.session_state["trace_id"])
    start_conflict_session(st.session_state["trace_id"])
    st.session_state["last_summary_point"] = 0
    st.session_state["last_summary_title"] = "Default Title"
    st.session_state["last_summary_summary"] = "This is the default summary for short conversations."
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Set up environment for API keys using provider abstraction
current_provider.setup_api_key()

os.makedirs(HISTORY_DIR, exist_ok=True)

# Helper Functions for Conversation Management
def save_history(title: str, summary: str):
    """Save the current conversation history to a file with title and summary."""
    history_data = {
        "title": title,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
        "trace_id": st.session_state.get("trace_id"),
        "messages": messages_to_dicts(st.session_state["final_state"]["messages"])
    }
    filename = f"{HISTORY_DIR}/{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(history_data, f)
    st.rerun()

def load_all_histories():
    """Load all saved conversation histories as a list of metadata for display."""
    histories = []
    for file in os.listdir(HISTORY_DIR):
        if file.endswith(".json"):
            with open(os.path.join(HISTORY_DIR, file), "r") as f:
                history = json.load(f)
                histories.append({
                    "title": history["title"],
                    "summary": history["summary"],
                    "timestamp": history["timestamp"],
                    "filename": file
                })
    return sorted(histories, key=lambda x: x["timestamp"], reverse=True)

def load_history(filename: str):
    """Load a specific conversation history file into session state."""
    try:
        with open(os.path.join(HISTORY_DIR, filename), "r") as f:
            history_data = json.load(f)
            st.session_state["final_state"]["messages"] = dicts_to_messages(history_data["messages"])
            # Restore trace_id so conflict log follows this conversation (if present)
            if history_data.get("trace_id"):
                st.session_state["trace_id"] = history_data["trace_id"]
                set_current_trace_id(st.session_state["trace_id"])
                start_conflict_session(st.session_state["trace_id"])
        st.sidebar.success(f"Conversation '{history_data['title']}' loaded successfully")
    except FileNotFoundError:
        st.sidebar.error("Conversation history not found.")

def delete_history(filename: str):
    """Delete a specific conversation history file."""
    os.remove(os.path.join(HISTORY_DIR, filename))
    st.sidebar.success("Conversation history deleted.")
    st.rerun()

# Convert messages to serializable dictionaries and vice versa
def messages_to_dicts(messages):
    return [msg.dict() for msg in messages]

def dicts_to_messages(dicts):
    reconstructed_messages = []
    for d in dicts:
        if d["type"] == "ai":
            reconstructed_messages.append(AIMessage(**d))
        elif d["type"] == "human":
            reconstructed_messages.append(HumanMessage(**d))
        elif d["type"] == "tool":
            reconstructed_messages.append(ToolMessage(**d))
    return reconstructed_messages

# Organize Sidebar with Tabs and improved styling
st.sidebar.title("⚙️ Settings")

# Debug toggle (prints richer logs to terminal)
if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = False
st.session_state["debug_logs"] = st.sidebar.checkbox("Enable debug logs", value=st.session_state["debug_logs"])

tab1, tab2, tab3, tab4 = st.sidebar.tabs(["💬 Conversation", "🎤 Voice", "🖼️ Image", "⚠️ Conflicts"])

# Initialize session state variables
if "last_summary_point" not in st.session_state:
    st.session_state["last_summary_point"] = 0
if "last_summary_title" not in st.session_state:
    st.session_state["last_summary_title"] = "Default Title"
if "last_summary_summary" not in st.session_state:
    st.session_state["last_summary_summary"] = "This is the default summary for short conversations."

# Tab 1: Conversation Management
with tab1:
    st.subheader("History")
    histories = load_all_histories()
    if histories:
        st.markdown("### Saved Histories")
        for history in histories:
            with st.expander(f"{history['title']} ({history['timestamp'][:10]})"):
                st.write(history["summary"])
                if st.button("Load", key=f"load_{history['filename']}"):
                    load_history(history["filename"])
                if st.button("Delete", key=f"delete_{history['filename']}"):
                    delete_history(history["filename"])

    # Determine title and summary based on message count and last summary point
    message_count = len(st.session_state["final_state"]["messages"])
    if message_count > 5 and (message_count - 5) % 10 == 0 and message_count != st.session_state["last_summary_point"]:
        provider_type = page.lower()
        try:
            generated_title, generated_summary = get_conversation_summary(st.session_state["final_state"]["messages"], provider_type)
            st.session_state["last_summary_title"] = generated_title
            st.session_state["last_summary_summary"] = generated_summary
        except Exception as e:
            st.session_state["last_summary_title"] = "Default Title"
            st.session_state["last_summary_summary"] = "This is the default summary for short conversations."
        st.session_state["last_summary_point"] = message_count
    elif message_count <= 5:
        st.session_state["last_summary_title"] = "Default Title"
        st.session_state["last_summary_summary"] = "This is the default summary for short conversations."

    title = st.text_input("Conversation Title", value=st.session_state["last_summary_title"])
    summary = st.text_area("Conversation Summary", value=st.session_state["last_summary_summary"])

    if st.button("Save Conversation"):
        save_history(title, summary)
        st.sidebar.success(f"Conversation saved as '{title}'")

# Tab 2: Voice Options
with tab2:
    st.subheader("Audio Options")
    use_audio_input = st.checkbox("Enable Voice Input", value=False)
    if use_audio_input:
        with st.form("audio_input_form", clear_on_submit=True):
            st.markdown("""
                <div class="audio-instructions">
                    <strong>Instructions for Recording Audio:</strong>
                    <ol style="padding-left: 20px; line-height: 1.5;">
                        <li>Click <strong>Submit Audio</strong> below to activate the audio recorder.</li>
                        <li>Once activated, click <strong>Start Recording</strong> to begin capturing audio.</li>
                        <li>When finished, click <strong>Stop</strong> to end the recording.</li>
                        <li>Finally, click <strong>Submit Audio</strong> again to use the recorded audio.</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
            submitted_audio = st.form_submit_button("Submit Audio")
            if submitted_audio:
                audio_transcript = input_from_mic()
                if audio_transcript:
                    st.session_state["audio_transcription"] = audio_transcript
                    prompt = st.session_state["audio_transcription"]
                else:
                    st.session_state["audio_transcription"] = None

    use_voice_response = st.checkbox("Enable Voice Response", value=False)
    if use_voice_response:
        st.write("If the voice response is too long, a summarized version will generate.")

# Tab 3: Image Upload
with tab3:
    st.subheader("Image")
    with st.form("image_upload_form", clear_on_submit=True):
        uploaded_images = st.file_uploader("Upload one or more images (optional)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submitted = st.form_submit_button("Submit Images")
        if submitted:
            if uploaded_images:
                st.session_state["uploaded_images_data"] = [
                    base64.b64encode(image.read()).decode("utf-8") for image in uploaded_images
                ]
            else:
                st.session_state["uploaded_images_data"] = []

# Tab 4: Conflict Checking Viewer
with tab4:
    st.subheader("Conflict Checking")
    trace_id = st.session_state.get("trace_id")
    st.write(f"**trace_id**: `{trace_id}`" if trace_id else "**trace_id**: (missing)")

    if trace_id:
        start_conflict_session(trace_id)
        conflict_path = get_conflict_path(trace_id)
        log = get_conflict_log(trace_id)
        events = log.get("events", []) if isinstance(log, dict) else []

        st.caption(f"Log file: `{conflict_path}`")
        st.write(f"**events**: {len(events)}")

        if events:
            # Aggregate conflicts across events for a quick overview
            all_conflicts = []
            for ev in events:
                res = (ev.get("result") or {})
                for c in (res.get("conflicts") or []):
                    all_conflicts.append(c)

            sev_counts = {"high": 0, "medium": 0, "low": 0}
            for c in all_conflicts:
                sev = (c.get("severity") or "").lower()
                if sev in sev_counts:
                    sev_counts[sev] += 1

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total conflicts", len(all_conflicts))
            col_b.metric("High", sev_counts["high"])
            col_c.metric("Medium", sev_counts["medium"])
            col_d.metric("Low", sev_counts["low"])

            last = events[-1]
            last_time = last.get("time", "")
            last_model = last.get("model", "")
            last_trigger = last.get("trigger_tool", "")
            last_summary = ((last.get("result") or {}).get("summary") or "").strip()
            last_conflicts = ((last.get("result") or {}).get("conflicts") or [])
            st.markdown("**Latest event**")
            st.write(f"- **time**: {last_time}")
            st.write(f"- **model**: {last_model}")
            st.write(f"- **trigger_tool**: {last_trigger}")
            if last_summary:
                st.write(f"- **summary**: {last_summary}")
            if last_conflicts:
                st.write(f"- **conflicts**: {len(last_conflicts)}")
            else:
                st.write("- **conflicts**: none ✅")

            if last_conflicts:
                st.markdown("**Latest conflicts (details)**")
                for c in last_conflicts:
                    st.write(
                        f"- **{c.get('severity','?')}** "
                        f"(conf={c.get('confidence','?')}): {c.get('claim','')}"
                    )
                    if c.get("conflict_type") or c.get("conflict_kind"):
                        st.write(
                            f"  - **type/kind**: {c.get('conflict_type','?')} / {c.get('conflict_kind','?')}"
                        )
                    evs = c.get("evidence") or []
                    if evs:
                        st.write("  - **evidence**:")
                        for e in evs[:5]:
                            st.write(f"    - {e}")
                    if c.get("suggested_resolution"):
                        st.write(f"  - **suggested_resolution**: {c.get('suggested_resolution')}")

            show_all = st.checkbox("Show all events", value=False)
            show_raw = st.checkbox("Show raw JSON", value=False)

            if show_all:
                for ev in reversed(events[-20:]):
                    with st.expander(f"{ev.get('time','')} | {ev.get('trigger_tool','')} | {ev.get('model','')}", expanded=False):
                        res = ev.get("result") or {}
                        summary = (res.get("summary") or "").strip()
                        if summary:
                            st.write(f"**summary**: {summary}")
                        conflicts = res.get("conflicts") or []
                        if conflicts:
                            st.write("**conflicts**:")
                            for c in conflicts:
                                st.write(
                                    f"- **{c.get('severity','?')}** "
                                    f"(conf={c.get('confidence','?')}): {c.get('claim','')}"
                                )
                                if c.get("suggested_resolution"):
                                    st.write(f"  - **suggested_resolution**: {c.get('suggested_resolution')}")
                        else:
                            st.write("**conflicts**: none")
                        if show_raw:
                            st.json(ev)

            st.download_button(
                "Download conflict log JSON",
                data=json.dumps(log, ensure_ascii=False, indent=2),
                file_name=f"conflicts_{trace_id}.json",
                mime="application/json",
            )
        else:
            st.info("No conflict events yet. After any assistant response > ~80 chars, an event should appear here.")
    else:
        st.info("Start a chat to initialize trace_id and conflict logging.")

# Initialize prompt variable
prompt = st.session_state.get("audio_transcription")

# Helper for debug logging
def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                try:
                    parts.append(json.dumps(item, ensure_ascii=False))
                except Exception:
                    parts.append(str(item))
        return "\n".join([p for p in parts if str(p).strip()])
    return str(content)

def _excerpt(text: str, n: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + " ...[truncated]"


_SENSITIVE_TEXT_PATTERNS = [
    (re.compile(r'(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|ACCESS_KEY|PRIVATE_KEY))\b\s*=\s*([^\s#]+)'), r'\1=<redacted>'),
    (re.compile(r'(?i)("?(?:api_key|token|secret|password|access_key|private_key)"?\s*:\s*")([^"]+)(")'), r'\1<redacted>\3'),
    (re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}'), "Bearer <redacted>"),
    (re.compile(r'(?i)(sk-ant-[A-Za-z0-9\-_]{10,}|sk-[A-Za-z0-9\-_]{10,}|AIza[0-9A-Za-z\-_]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9\-]{10,})'), "<redacted-secret>"),
    (re.compile(r'://([^/\s:@]+):([^@\s]+)@'), '://<redacted>:<redacted>@'),
]


def _redact_sensitive_text(text: str) -> str:
    redacted = text or ""
    for pattern, repl in _SENSITIVE_TEXT_PATTERNS:
        redacted = pattern.sub(repl, redacted)
    return redacted


def _safe_excerpt(text: str, n: int = 600) -> str:
    return _excerpt(_redact_sensitive_text(text), n)

# Main chat interface
st.markdown(f"""
    <div class="chat-title">
        <span class="robot-icon">🤖</span>
        <span>Chat with Spatial Transcriptomics Agent</span>
    </div>
""", unsafe_allow_html=True)

render_conversation_history(st.session_state["final_state"]["messages"][0:])

# Capture text input if no audio input
if prompt is None:
    prompt = st.chat_input()

# Process new user input if available
if prompt:
    content_list = [{"type": "text", "text": prompt}]
    if "uploaded_images_data" in st.session_state and st.session_state["uploaded_images_data"]:
        content_list.extend([
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
            for img_data in st.session_state["uploaded_images_data"]
        ])
        st.session_state["uploaded_images_data"] = []
    
    user_message = HumanMessage(content=content_list)
    st.session_state["final_state"]["messages"].append(user_message)
    render_conversation_history([user_message])

    with st.spinner(f"Agent is thinking..."):
        previous_message_count = len(st.session_state["final_state"]["messages"])
        if st.session_state.get("debug_logs"):
            print(
                f"[STAgent] invoke model={selected_model} msgs={previous_message_count} trace_id={st.session_state.get('trace_id')} at={datetime.now().isoformat()}\n"
                f"[STAgent] user_excerpt:\n{_safe_excerpt(_content_to_text(user_message.content), 800)}\n"
            )
        else:
            print(f"[STAgent] invoke model={selected_model} msgs={previous_message_count} at={datetime.now().isoformat()}")
        updated_state = invoke_our_graph(st.session_state["final_state"]["messages"], selected_model)
    
    st.session_state["final_state"] = updated_state
    new_messages = st.session_state["final_state"]["messages"][previous_message_count:]

    if st.session_state.get("debug_logs"):
        print(f"[STAgent] returned new_messages={len(new_messages)} at={datetime.now().isoformat()}")
        for i, m in enumerate(new_messages[-8:]):  # keep bounded
            m_type = m.__class__.__name__
            m_name = getattr(m, "name", None)
            tool_calls = getattr(m, "tool_calls", None)
            header = f"[STAgent] new[{i - min(len(new_messages), 8)}] type={m_type}"
            if m_name:
                header += f" name={m_name}"
            if tool_calls:
                try:
                    tool_names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict)]
                except Exception:
                    tool_names = ["(unreadable)"]
                header += f" tool_calls={tool_names}"
            print(header)
            print(_safe_excerpt(_content_to_text(getattr(m, "content", "")), 1200))
            print("---")
    
    if st.session_state.get("render_last_message", True):
        # Render everything produced by this invocation (tool outputs + plots + final assistant text)
        render_conversation_history(new_messages)
    
    if use_voice_response:
        audio_file = convert_text_to_speech(new_messages[-1].content)
        if audio_file:
            st.audio(audio_file)
    
    st.session_state["audio_transcription"] = None 

