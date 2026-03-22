"""
Unified graph implementation for the supported STAgent providers.
This eliminates code duplication across the provider-specific graph entry points.
"""

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import matplotlib.pyplot as plt
import re
from typing import Annotated, TypedDict, Literal, Tuple, List, Dict, Any, Optional
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from tools import PythonREPL
from langgraph.prebuilt import ToolNode
from prompt import system_prompt
from langgraph.types import Command
from textwrap import dedent
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - allow non-Streamlit/batch contexts
    st = None  # type: ignore
from util_unified import display_message, render_conversation_history, get_conversation_summary
from langchain_core.runnables.config import RunnableConfig
from config import get_config, get_tool_config
from pydantic import BaseModel, Field
from conflict_log import append_event as conflict_append_event, start_conflict_session
from trace_log import get_current_trace_id as get_current_trace_id_disk, set_current_trace_id
from tools import (
    explore_metadata_tool,
    quality_control,
    preprocess_stereo_seq,
    google_scholar_search,
    deeper_research,
    squidpy_rag_agent,
    get_conflict_log_tool,
    results_aggregator_tool,
    visualize_cell_cell_interaction_tool,
    ligand_receptor_compute_squidpy,
    ligand_receptor_visualize_squidpy,
    visualize_spatial_cell_type_map,
    visualize_cell_type_composition,
    visualize_umap,
    report_tool,
    cell_type_annotation_guide,
    spatial_domain_identification_staligner,
    gene_imputation_tangram,
)

# Directory Setup using centralized config
config = get_config()
plot_dir = config.plot_dir
os.makedirs(plot_dir, exist_ok=True)
load_dotenv(Path(__file__).resolve().with_name(".env"))

python_repl = PythonREPL()

_FALLBACK_SESSION_STATE: Dict[str, Any] = {}


def _parse_repl_exec_directives(query: str) -> Dict[str, Optional[str]]:
    """
    Parse optional execution directives from comment headers in code.

    Supported headers:
      # STAGENT_EXEC_MODE: external|internal
      # STAGENT_PYTHON_BIN: conda run -n STAgent_gpusub python
      # STAGENT_EXEC_CWD: /abs/or/relative/path
      # STAGENT_EXEC_TIMEOUT: 7200

    Defaults are intentionally conservative so existing behavior is unchanged.
    """
    defaults: Dict[str, Optional[str]] = {
        "mode": "internal",
        "python_bin": None,
        "cwd": None,
        "timeout": None,
    }
    text = str(query or "")
    if not text.strip():
        return defaults

    # Read only the first block of lines where directives are expected.
    header_lines = text.splitlines()[:40]
    pattern = re.compile(
        r"^\s*#\s*(STAGENT_EXEC_MODE|STAGENT_PYTHON_BIN|STAGENT_EXEC_CWD|STAGENT_EXEC_TIMEOUT)\s*:\s*(.*?)\s*$"
    )
    for line in header_lines:
        m = pattern.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if key == "STAGENT_EXEC_MODE":
            defaults["mode"] = value.lower() if value else "internal"
        elif key == "STAGENT_PYTHON_BIN":
            defaults["python_bin"] = value or None
        elif key == "STAGENT_EXEC_CWD":
            defaults["cwd"] = value or None
        elif key == "STAGENT_EXEC_TIMEOUT":
            defaults["timeout"] = value or None
    return defaults


def _in_streamlit_app() -> bool:
    """True only when running under `streamlit run` (not just `import streamlit`)."""
    try:
        if st is None:
            return False
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _get_session_state() -> Dict[str, Any]:
    """Best-effort session state that works in UI + batch contexts."""
    if st is None:
        return _FALLBACK_SESSION_STATE
    try:
        return st.session_state  # type: ignore[attr-defined]
    except Exception:
        return _FALLBACK_SESSION_STATE


def _make_session_id() -> str:
    """Timestamp-prefixed id for logs/filenames."""
    return f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:10]}"


def _resolve_session_id(config: RunnableConfig) -> str:
    """
    Resolve a session_id/trace_id for logging.

    Priority:
    1) config.configurable.session_id (explicit)
    2) env STAGENT_SESSION_ID (batch control)
    3) streamlit session_state["trace_id"] (UI)
    4) trace_log current.json (batch persistence)
    5) generate a new id (and persist to disk)
    """
    sid = None
    try:
        sid = (config or {}).get("configurable", {}).get("session_id")
    except Exception:
        sid = None
    if sid:
        return str(sid)

    env_sid = os.getenv("STAGENT_SESSION_ID")
    if env_sid:
        return str(env_sid)

    ss = _get_session_state()
    if ss.get("trace_id"):
        return str(ss.get("trace_id"))

    disk = get_current_trace_id_disk()
    if disk:
        return str(disk)

    new_sid = _make_session_id()
    try:
        set_current_trace_id(new_sid)
    except Exception:
        pass
    return new_sid


@tool(response_format="content_and_artifact")
def python_repl_tool(query: str) -> Tuple[str, List[str]]:
    """A Python shell for executing Python commands.

    Default behavior is in-process execution for backward compatibility.

    Optional external subprocess mode can be enabled by directive headers in `query`:
      - # STAGENT_EXEC_MODE: external
      - # STAGENT_PYTHON_BIN: <python command>  (optional)
      - # STAGENT_EXEC_CWD: <working directory> (optional)
      - # STAGENT_EXEC_TIMEOUT: <seconds>        (optional)

    External defaults:
      - STAGENT_GPU_PYTHON_BIN (fallback: conda run -n STAgent_gpusub python)
      - STAGENT_GPU_TOOL_CWD

    If you want to see expression values, print them with `print(...)`.
    """
    
    plot_paths = []  # List to store file paths of generated plots
    result_parts = []  # List to store different parts of the output
    
    try:
        directives = _parse_repl_exec_directives(query)
        mode = str(directives.get("mode") or "internal").lower()
        use_external = mode == "external"

        if use_external:
            python_bin = (
                directives.get("python_bin")
                or os.getenv("STAGENT_GPU_PYTHON_BIN")
                or "conda run -n STAgent_gpusub python"
            )
            cwd = directives.get("cwd") or os.getenv("STAGENT_GPU_TOOL_CWD") or None
            timeout_s = get_tool_config().repl_timeout
            timeout_raw = directives.get("timeout")
            if timeout_raw:
                try:
                    parsed = int(str(timeout_raw).strip())
                    if parsed > 0:
                        timeout_s = parsed
                except Exception:
                    pass
            if timeout_raw is None:
                env_timeout = os.getenv("STAGENT_EXTERNAL_TIMEOUT")
                if env_timeout:
                    try:
                        parsed = int(str(env_timeout).strip())
                        if parsed > 0:
                            timeout_s = parsed
                    except Exception:
                        pass
            output = python_repl.run_external(
                query,
                python_bin=python_bin,
                timeout=timeout_s,
                cwd=cwd,
            )
            if output and output.strip():
                result_parts.append(output.strip())
            # External subprocess execution is isolated, so matplotlib artifacts are not auto-captured.
            result_parts.append(
                f"Execution mode: external (subprocess, timeout={timeout_s}s). "
                "Plot artifacts are not auto-captured in this mode."
            )
        else:
            output = python_repl.run(query)
            if output and output.strip():
                result_parts.append(output.strip())
            
            figures = [plt.figure(i) for i in plt.get_fignums()]
            if figures:
                tool_config = get_tool_config()
                for fig in figures:
                    fig.set_size_inches(*tool_config.default_figure_size)
                    plot_filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{tool_config.save_format}"
                    # Create relative path
                    rel_path = os.path.join("tmp/plots", plot_filename)
                    # Convert to absolute path for saving
                    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                    
                    fig.savefig(abs_path, bbox_inches=tool_config.bbox_inches, dpi=tool_config.default_dpi)
                    plot_paths.append(rel_path)  # Store relative path
                
                plt.close("all")
                result_parts.append(f"Generated {len(plot_paths)} plot(s).")
        
        if not result_parts:  # If no output and no figures
            result_parts.append("Executed code successfully with no output. If you want to see the output of a value, you should print it out with `print(...)`.")

    except Exception as e:
        result_parts.append(f"Error executing code: {e}")
    
    # Join all parts of the result with newlines
    result_summary = "\n".join(result_parts)
    
    # Return both the summary and plot paths (if any)
    return result_summary, plot_paths


# Tools List and Node Setup
tools = [
    python_repl_tool,
    explore_metadata_tool,
    quality_control,
    preprocess_stereo_seq,
    google_scholar_search,
    deeper_research,
    get_conflict_log_tool,
    squidpy_rag_agent,
    results_aggregator_tool,
    visualize_cell_cell_interaction_tool,
    ligand_receptor_compute_squidpy,
    ligand_receptor_visualize_squidpy,
    visualize_spatial_cell_type_map,
    visualize_cell_type_composition,
    visualize_umap,
    report_tool,
    cell_type_annotation_guide, # 20260130 added
    spatial_domain_identification_staligner,
    gene_imputation_tangram,
]
tool_node = ToolNode(tools)


# Graph Setup
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_messages_len: list[int]


class ConflictItem(BaseModel):
    claim: str = Field(description="Atomic claim/conclusion derived from the assistant's interpretation.")
    conflict_type: Literal["internal_knowledge", "literature", "both"] = Field(
        description="Whether the conflict is against internal knowledge, retrieved literature, or both."
    )
    conflict_kind: Literal["hard_contradiction", "opposite_trend", "uncertain_evidence"] = Field(
        description="Type of conflict: direct contradiction, opposite trend, or weak/uncertain evidence."
    )
    evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Short quotes/snippets supporting why this is a conflict. "
            "Prefix each evidence line with 'INT:' (internal knowledge) or 'LIT:' (literature excerpt)."
        ),
    )
    severity: Literal["low", "medium", "high"] = Field(description="How serious the conflict is for downstream conclusions.")
    confidence: float = Field(description="0-1 confidence score for the conflict assessment.")
    suggested_resolution: str = Field(
        description="Action to resolve/mitigate (e.g., add caveat, rerun QC, check confounders, verify mapping)."
    )


class ConflictCheckResult(BaseModel):
    extracted_claims: List[str] = Field(default_factory=list, description="Extracted claims from the assistant message.")
    conflicts: List[ConflictItem] = Field(default_factory=list, description="Detected conflicts, if any.")
    summary: str = Field(default="", description="Short summary of overall alignment/conflicts.")


class UnifiedLLMGraph:
    """Unified LLM Graph that works with both OpenAI and Anthropic providers."""
    
    def __init__(self):
        self.graph = StateGraph(GraphsState)
        self._setup_models()
        self._setup_graph()
        
    def _setup_models(self):
        """Setup models for all providers using centralized config."""
        config = get_config()
        
        # OpenAI models
        self.openai_models = {}
        self.openai_models_plain = {}
        for model_name in config.get_available_models("openai"):
            model_config = config.get_model_config("openai", model_name)
            self.openai_models_plain[model_name] = ChatOpenAI(
                model_name=model_name,
                temperature=model_config.get("temperature", 0)
            )
            self.openai_models[model_name] = self.openai_models_plain[model_name].bind_tools(
                tools, parallel_tool_calls=model_config.get("parallel_tool_calls", False)
            )
        
        # Anthropic models
        self.anthropic_models = {}
        self.anthropic_models_plain = {}
        for model_name in config.get_available_models("anthropic"):
            model_config = config.get_model_config("anthropic", model_name)
            self.anthropic_models_plain[model_name] = ChatAnthropic(
                model_name=model_name,
                temperature=model_config.get("temperature", 0),
                max_tokens=model_config.get("max_tokens", 8000)
            )
            self.anthropic_models[model_name] = self.anthropic_models_plain[model_name].bind_tools(tools)
        
        # Gemini models (standard API)
        self.gemini_models = {}
        self.gemini_models_plain = {}
        for model_name in config.get_available_models("gemini"):
            model_config = config.get_model_config("gemini", model_name)
            self.gemini_models_plain[model_name] = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=model_config.get("temperature", 0),
                google_api_key=os.getenv('GOOGLE_API_KEY'),
                convert_system_message_to_human=model_config.get("convert_system_message_to_human", True)
            )
            self.gemini_models[model_name] = self.gemini_models_plain[model_name].bind_tools(tools)
        
        # Combined models dictionary
        self.all_models = {
            **self.openai_models, 
            **self.anthropic_models,
            **self.gemini_models,
        }

        # Plain (no-tool) models for internal checks (avoid tool recursion)
        self.all_models_plain = {
            **self.openai_models_plain,
            **self.anthropic_models_plain,
            **self.gemini_models_plain,
        }

    def _content_to_text(self, content: Any) -> str:
        """Normalize message content (str/list/dict) into a compact string."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join([p for p in parts if p.strip()])
        return str(content)

    def _latest_user_text(self, messages: List[AnyMessage]) -> str:
        """Best-effort latest human text extraction."""
        for m in reversed(messages or []):
            if isinstance(m, HumanMessage):
                return self._content_to_text(getattr(m, "content", ""))
        return ""

    def _has_explicit_lr_confirmation(self, messages: List[AnyMessage]) -> bool:
        """
        Require explicit user intent before running ligand-receptor profiling.

        We accept either:
        - a direct confirmation phrase (yes/confirm/proceed/run/go ahead), or
        - an explicit subgroup directive in user text (source_groups / target_groups).
        """
        text = (self._latest_user_text(messages) or "").lower()
        if not text.strip():
            return False
        has_confirm = bool(
            re.search(r"\b(yes|confirm|confirmed|proceed|go ahead|run it|do it|execute)\b", text)
        )
        has_explicit_groups = ("source_groups" in text) or ("target_groups" in text)
        return has_confirm or has_explicit_groups

    def _run_conflict_check(
        self,
        *,
        session_id: str,
        model_name: str,
        trigger_tool: str,
        assistant_text: str,
        literature_text: str,
        message_index: int,
    ) -> Optional[ConflictCheckResult]:
        """Run conflict checking and append results to the session conflict log."""
        assistant_text = assistant_text.strip()
        if len(assistant_text) < 80:
            return None

        # Keep token/cost bounded
        assistant_text = assistant_text[:6000]
        literature_text = (literature_text or "").strip()[:9000]

        checker = self.all_models_plain.get(model_name)
        if checker is None:
            return None

        check_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a scientific consistency checker for spatial biology. "
             "Given an assistant's interpretation and optional literature excerpts, "
             "extract the BIOLOGICAL claims (mechanisms, spatial organization, cell-type relationships, developmental/disease implications) "
             "and detect conflicts.\n\n"
             "Conflicts include:\n"
             "- hard_contradiction: directly incompatible statements\n"
             "- opposite_trend: literature suggests opposite direction/association/trend\n"
             "- uncertain_evidence: literature is weak/mixed/insufficient for the strength of the claim\n\n"
             "Compare claims against (1) internal general scientific knowledge and (2) the provided literature excerpts. "
             "If literature is absent, only check internal knowledge.\n\n"
             "Be conservative: do not invent citations; only use provided excerpts as evidence for literature conflicts.\n\n"
             "CRITICAL focus rules:\n"
             "- Ignore tool/process narration (e.g., mentions of UMAP/code/tool names) unless it changes a biological conclusion.\n"
             "- If the 'conflict' is actually a technical confounder (annotation mismatch, contamination, batch effect), "
             "rewrite the claim in biological terms (what biological conclusion would be invalidated) and put the technical issue in suggested_resolution.\n"
             "- Prefer biologically meaningful atomic claims over purely technical statements.\n\n"
             "Literature-grounding rules (when literature excerpts are provided):\n"
             "- A conflict_type of 'literature' or 'both' MUST include at least 1-2 short quoted/snippet evidence lines "
             "copied from LITERATURE_EXCERPTS, each prefixed with 'LIT:'.\n"
             "- Do NOT label something as a literature conflict if you cannot point to a provided excerpt.\n"
             "- If literature is mixed/weak, prefer conflict_kind='uncertain_evidence' and explain that the excerpt is insufficient for the claim strength.\n\n"
             "For each detected conflict, you MUST set conflict_type:\n"
             "- internal_knowledge: conflict only against general scientific knowledge (no excerpt support)\n"
             "- literature: conflict only against the provided literature excerpts (support with 'LIT:' evidence)\n"
             "- both: conflicts against both internal knowledge and the excerpts (include both 'INT:' and 'LIT:' evidence)\n\n"
             "Evidence formatting requirement:\n"
             "- Prefix each evidence line with 'INT:' or 'LIT:' so downstream code can separate them.\n"
             "Return structured output."),
            ("human",
             "TRIGGER_TOOL: {trigger_tool}\n"
             "ASSISTANT_INTERPRETATION:\n{assistant_text}\n\n"
             "LITERATURE_EXCERPTS (may be empty):\n{literature_text}\n\n"
             "Return:\n"
             "- extracted_claims: up to 10 concise claims\n"
             "- conflicts: only claims with detected conflicts (may be empty)\n"
             "- summary: 1-3 sentences"),
        ])

        structured = checker.with_structured_output(ConflictCheckResult)
        chain = check_prompt | structured
        try:
            result: ConflictCheckResult = chain.invoke({
                "trigger_tool": trigger_tool,
                "assistant_text": assistant_text,
                "literature_text": literature_text,
            })
        except Exception:
            return None

        # Safety normalization: keep conflict_type consistent with evidence + lit availability.
        lit_present = bool((literature_text or "").strip())
        if result and result.conflicts:
            for c in result.conflicts:
                ev = c.evidence or []
                has_lit_ev = any(str(x).lstrip().upper().startswith("LIT:") for x in ev)
                has_int_ev = any(str(x).lstrip().upper().startswith("INT:") for x in ev)

                if not lit_present:
                    # No excerpts were provided, so literature-based conflicts are impossible.
                    c.conflict_type = "internal_knowledge"
                    # Normalize evidence prefixes if any were mistakenly marked as literature.
                    if has_lit_ev and not has_int_ev:
                        c.evidence = [re.sub(r"(?i)^\\s*LIT\\s*:", "INT:", str(x), count=1) for x in ev]
                    continue

                # If literature is present, align conflict_type with evidence.
                if has_lit_ev and has_int_ev:
                    c.conflict_type = "both"
                elif has_lit_ev and not has_int_ev:
                    c.conflict_type = "literature"
                elif has_int_ev and not has_lit_ev:
                    c.conflict_type = "internal_knowledge"
                else:
                    # No prefixed evidence; trust model label if valid, otherwise default to "both".
                    if c.conflict_type not in ("internal_knowledge", "literature", "both"):
                        c.conflict_type = "both"

        start_conflict_session(session_id)
        event = {
            "event_id": f"evt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}",
            "time": datetime.utcnow().isoformat(),
            "model": model_name,
            "trigger_tool": trigger_tool,
            "message_index": message_index,
            "assistant_excerpt": assistant_text[:400],
            "literature_excerpt": literature_text[:400] if literature_text else "",
            "result": result.model_dump(),
        }
        conflict_append_event(session_id, event)
        return result
    
    def _get_provider_type(self, model_name: str) -> str:
        """Determine provider type from model name."""
        if model_name in self.openai_models:
            return "openai"
        elif model_name in self.anthropic_models:
            return "anthropic"
        elif model_name in self.gemini_models:
            return "gemini"
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _get_excluded_model_check(self, model_name: str) -> str:
        """Get the model name to exclude from image processing."""
        provider_type = self._get_provider_type(model_name)
        if provider_type == "openai":
            return "gpt-3.5-turbo"  # Exclude basic models from image processing
        elif provider_type == "anthropic":
            return "claude_3_5_haiku"  # Exclude basic Anthropic models
        elif provider_type == "gemini":
            return "gemini-1.0-pro"  # Exclude basic Gemini models
        else:
            return ""  # Default: don't exclude
    
    def _call_model(self, state: GraphsState, config: RunnableConfig) -> Command[Literal["tools", "conflictNode", "__end__"]]:
        """Unified model calling function that works with both providers."""
        ss = _get_session_state()
        # Keep UI state in sync when available, but do not require Streamlit to run.
        try:
            if isinstance(ss.get("final_state"), dict) and "messages" in ss["final_state"]:
                ss["final_state"]["messages"] = state["messages"]
        except Exception:
            pass
        
        # Get model configuration
        model_name = config["configurable"].get("model", "gpt-4o")
        llm = self.all_models[model_name]
        
        # Track message lengths
        previous_message_count = len(state["messages"])
        state["input_messages_len"].append(previous_message_count)
        
        # Render conversation history
        try:
            render_conversation_history(
                state["messages"][state["input_messages_len"][-2]:state["input_messages_len"][-1]]
            )
        except Exception:
            pass
        
        # Check recursion limit
        cur_messages_len = len(state["messages"]) - state["input_messages_len"][0]
        max_steps = get_config().max_recursion_steps
        if cur_messages_len > max_steps:
            if st is not None and _in_streamlit_app():
                st.markdown(
                    f"""
                    <p style="color:blue; font-size:16px;">
                        Current recursion step is {cur_messages_len}. Terminated because you exceeded the limit of {max_steps}.
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            try:
                ss["render_last_message"] = False
            except Exception:
                pass
            return Command(
                update={"messages": []},
                goto="__end__",
            )
        
        # Handle post-tool analysis and auto literature search
        last_message = state["messages"][-1]
        excluded_model = self._get_excluded_model_check(model_name)
        artifact_count = 0
        
        last_idx = len(state["messages"]) - 1
        last_tool_name = getattr(last_message, "name", "") if isinstance(last_message, ToolMessage) else ""
        has_artifacts = bool(getattr(last_message, "artifact", None)) if isinstance(last_message, ToolMessage) else False
        is_plot_tool = last_tool_name in {
            "python_repl_tool",
            "visualize_umap",
            "visualize_cell_type_composition",
            "visualize_spatial_cell_type_map",
            "visualize_cell_cell_interaction_tool",
        }
        should_post_tool_flow = isinstance(last_message, ToolMessage) and (has_artifacts or is_plot_tool)

        if has_artifacts:
            try:
                artifact_count = len(last_message.artifact) if isinstance(last_message.artifact, list) else 1
            except Exception:
                artifact_count = 1
            
            # Prepare content list with initial text (XML-structured, no web-search directive).
            # This is a hard enforcement point: after plots are generated, ALWAYS provide analysis/explanation.
            content_list = [{
                "type": "text",
                "text": """
<image_analysis>
  <instructions>
    <task>
      You MUST produce an analysis message now.
      Examine each visualization (if present) and provide:
      - A concise description of what is shown
      - Key observations and patterns
      - Biological interpretation grounded in the plots and the tool output
    </task>
    <constraints>
      - Do not assume external web search is required unless the user asks.
      - Keep language consistent with the user.
      - If plots are unclear or missing, interpret based on the tool output and suggest minimal code fixes (do not call plt.close()).
      - DO NOT call any tools in this response. Output only your explanation/analysis.
    </constraints>
    <next_step>
      End with an explicit "Next step" recommendation (one action).
    </next_step>
  </instructions>
</image_analysis>
                """
            }]
            
            # Add all PNG images to the content list only if the model supports image inputs.
            # For excluded models, we still force an explanation using tool output (text-only).
            if model_name != excluded_model:
                for rel_path in last_message.artifact:
                    if rel_path.endswith(".png"):
                        # Convert relative path to absolute based on current script location
                        abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                        if os.path.exists(abs_path):
                            with open(abs_path, "rb") as image_file:
                                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                            content_list.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            })
            
            # Always create a follow-up instruction message when artifacts exist:
            # even if images couldn't be loaded, the assistant should still provide an explanation.
            image_message = HumanMessage(content=content_list, name="image_assistant")
            if st.session_state.get("debug_logs"):
                print(f"[STAgent][graph] artifact_detected count={artifact_count} model={model_name} excluded_model={excluded_model}")

        if should_post_tool_flow:
            # Analysis on plots/tool output (no tools). Then continue the graph so the agent can decide next steps.
            messages_for_analysis = state["messages"] + [image_message] if has_artifacts else state["messages"]
            plain = self.all_models_plain.get(model_name) or llm
            try:
                analysis_response = plain.invoke(messages_for_analysis)
            except Exception as e:
                err = AIMessage(
                    content=(
                        "Model provider error during plot analysis "
                        f"({e.__class__.__name__}). Please retry the same request."
                    )
                )
                return Command(update={"messages": [err]}, goto="__end__")
            if getattr(analysis_response, "tool_calls", None):
                analysis_response = AIMessage(content=getattr(analysis_response, "content", ""))
            analysis_text = self._content_to_text(getattr(analysis_response, "content", ""))
            # Avoid noisy duplicated headers like "Step 1 — Analysis" and keep the UI clean.
            analysis_text = (analysis_text or "").strip()
            analysis_text = re.sub(r"^(?:Step\\s*1\\s*[—-]\\s*Analysis\\s*)+", "", analysis_text, flags=re.IGNORECASE).strip()
            analysis_response = AIMessage(content=analysis_text or "Analysis complete.")

            try:
                ss["render_last_message"] = True
            except Exception:
                pass
            return Command(update={"messages": [analysis_response]}, goto="modelNode")
        
        # Invoke the model (include any extra messages we injected)
        messages_for_model = state["messages"]
        try:
            response = llm.invoke(messages_for_model)
        except Exception as e:
            # Avoid crashing Streamlit on transient provider/network errors (e.g., 504).
            err_name = e.__class__.__name__
            friendly = AIMessage(
                content=(
                    f"Model provider request failed ({err_name}). "
                    "This is usually transient (gateway timeout/network). "
                    "Please retry the same step."
                )
            )
            return Command(update={"messages": [friendly]}, goto="__end__")
        
        # Determine next step based on response
        if response.tool_calls:
            # Safety gate: never run ligand-receptor tool without explicit user confirmation.
            try:
                tool_names = [tc.get("name") for tc in (response.tool_calls or []) if isinstance(tc, dict)]
            except Exception:
                tool_names = []
            if (
                "ligand_receptor_compute_squidpy" in tool_names
                and not self._has_explicit_lr_confirmation(state["messages"])
            ):
                ask = AIMessage(
                    content=(
                        "Before running ligand-receptor profiling, please explicitly confirm your plan. "
                        "Reply with confirmation (e.g., 'Yes, proceed') and subgroup choices "
                        "(source_groups/target_groups), or say all-to-all if intended."
                    )
                )
                return Command(update={"messages": [ask]}, goto="__end__")
            return Command(
                update={"messages": [response]},
                goto="tools",
            )
        else:
            try:
                ss["render_last_message"] = True
            except Exception:
                pass
            return Command(
                update={"messages": [response]},
                goto="conflictNode",
            )

    def _conflict_node(self, state: GraphsState, config: RunnableConfig) -> Command[Literal["__end__"]]:
        """Run conflict checking after the assistant produces analysis/explanation text."""
        ss = _get_session_state()
        try:
            model_name = config["configurable"].get("model", "gpt-4o")
        except Exception:
            model_name = "gpt-4o"

        msgs = state.get("messages", [])
        if len(msgs) < 2:
            return Command(update={}, goto="__end__")

        # Use the most recent AIMessage (even if the last message is a ToolMessage)
        last_ai_idx = None
        last_ai_msg = None
        for i in range(len(msgs) - 1, -1, -1):
            if isinstance(msgs[i], AIMessage):
                last_ai_idx = i
                last_ai_msg = msgs[i]
                break
        if last_ai_msg is None:
            return Command(update={}, goto="__end__")
        if getattr(last_ai_msg, "tool_calls", None):
            return Command(update={}, goto="__end__")

        if ss.get("last_conflict_checked_idx") == last_ai_idx:
            return Command(update={}, goto="__end__")

        recent_tool_msgs = [m for m in msgs[-20:] if isinstance(m, ToolMessage)]

        session_id = _resolve_session_id(config)
        # If a UI session exists, mirror the resolved id into it so the sidebar can find the right log.
        try:
            if not ss.get("trace_id"):
                ss["trace_id"] = session_id
        except Exception:
            pass
        assistant_text = self._content_to_text(getattr(last_ai_msg, "content", ""))

        # Gather recent literature excerpts from tool outputs
        # These are the sources we treat as "literature" for conflict checking.
        # If you add more literature tools, include them here.
        lit_tools = {"google_scholar_search", "deeper_research"}
        lit_msgs = [
            m for m in msgs[-50:]
            if isinstance(m, ToolMessage) and (getattr(m, "name", "") or "") in lit_tools
        ]
        literature_text = "\n\n".join(
            [f"[{getattr(m, 'name', '')}] {self._content_to_text(getattr(m, 'content', ''))[:5000]}" for m in lit_msgs[-5:]]
        )

        trigger_tool = (getattr(recent_tool_msgs[-1], "name", "") or "analysis") if recent_tool_msgs else "analysis"
        result = self._run_conflict_check(
            session_id=session_id,
            model_name=model_name,
            trigger_tool=trigger_tool,
            assistant_text=assistant_text,
            literature_text=literature_text,
            message_index=last_ai_idx,
        )
        try:
            ss["last_conflict_checked_idx"] = last_ai_idx
        except Exception:
            pass

        # Do not pollute the conversation by default.
        # Conflicts are surfaced via the Streamlit sidebar "Conflicts" tab.
        # If you explicitly want a chat message, set configurable.emit_conflict_message=True.
        emit_msg = False
        try:
            emit_msg = bool((config or {}).get("configurable", {}).get("emit_conflict_message"))
        except Exception:
            emit_msg = False
        if emit_msg:
            if result:
                conflicts = result.conflicts or []
                type_counts = {"internal_knowledge": 0, "literature": 0, "both": 0}
                for c in conflicts:
                    t = (getattr(c, "conflict_type", None) or "both")
                    if t in type_counts:
                        type_counts[t] += 1
                if conflicts:
                    lines = [
                        "Step 3 — Conflict Check",
                        "",
                        f"Conflicts detected: {len(conflicts)} "
                        f"(internal={type_counts['internal_knowledge']}, literature={type_counts['literature']}, both={type_counts['both']})",
                    ]
                    for c in conflicts[:6]:
                        lines.append(
                            f"- **{c.severity}** [{c.conflict_type}] (conf={c.confidence}): {c.claim}"
                        )
                        if c.suggested_resolution:
                            lines.append(f"  - suggested_resolution: {c.suggested_resolution}")
                else:
                    lines = [
                        "Step 3 — Conflict Check",
                        "",
                        "No conflicts detected.",
                    ]
                return Command(update={"messages": [AIMessage(content="\n".join(lines))]}, goto="__end__")

        return Command(update={}, goto="__end__")
    
    def _setup_graph(self):
        """Setup the graph structure."""
        self.graph.add_edge(START, "modelNode")
        self.graph.add_node("tools", tool_node)
        self.graph.add_node("modelNode", self._call_model)
        self.graph.add_node("conflictNode", self._conflict_node)
        self.graph.add_edge("tools", "modelNode")
        self.graph.add_edge("conflictNode", END)
        self.graph_runnable = self.graph.compile()
    
    def invoke(self, messages: List[AnyMessage], model_name: str) -> Dict[str, Any]:
        """Invoke the graph with the specified model."""
        max_recursion = get_config().max_recursion_steps
        config = {"recursion_limit": max_recursion, "configurable": {"model": model_name}}
        return self.graph_runnable.invoke(
            {"messages": messages, "input_messages_len": [len(messages)]}, 
            config=config
        )
    
    def get_available_models(self, provider_type: str = None) -> List[str]:
        """Get available models, optionally filtered by provider type."""
        if provider_type == "openai":
            return list(self.openai_models.keys())
        elif provider_type == "anthropic":
            return list(self.anthropic_models.keys())
        elif provider_type == "gemini":
            return list(self.gemini_models.keys())
        else:
            return list(self.all_models.keys())
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available."""
        return model_name in self.all_models


unified_graph: Optional["UnifiedLLMGraph"] = None


def _get_unified_graph() -> "UnifiedLLMGraph":
    """
    Lazy singleton initializer.

    Avoids import-time hard dependency on provider API keys (OpenAI/Anthropic/Gemini),
    while still allowing benchmark scripts to override `unified_graph` explicitly.
    """
    global unified_graph
    if unified_graph is None:
        unified_graph = UnifiedLLMGraph()
    return unified_graph


def invoke_our_graph(
    messages: List[AnyMessage],
    model_name: str,
    *,
    session_id: Optional[str] = None,
    emit_conflict_message: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Main entry point for graph invocation.
    This replaces both the original invoke_our_graph functions.
    """
    max_recursion = get_config().max_recursion_steps
    configurable: Dict[str, Any] = {"model": model_name}
    if session_id:
        configurable["session_id"] = session_id
    if emit_conflict_message is not None:
        configurable["emit_conflict_message"] = bool(emit_conflict_message)
    config = {"recursion_limit": max_recursion, "configurable": configurable}
    graph = _get_unified_graph()
    return graph.graph_runnable.invoke(
        {"messages": messages, "input_messages_len": [len(messages)]},
        config=config,
    )


def get_available_models_for_provider(provider_type: str) -> List[str]:
    """Get available models for a specific provider."""
    return _get_unified_graph().get_available_models(provider_type)


def is_model_available(model_name: str) -> bool:
    """Check if a model is available."""
    return _get_unified_graph().is_model_available(model_name)
