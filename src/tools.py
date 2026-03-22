from dotenv import load_dotenv
from langchain_core.tools import tool
from serpapi import GoogleSearch    
import os
from squidpy_rag import squidpy_rag_agent
from textwrap import dedent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from typing import Annotated, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import functools
import logging
import multiprocessing
import json
import re
import sys
import subprocess
import shlex
from io import StringIO
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from pathlib import Path
from conflict_log import get_log as get_conflict_log
from trace_log import get_current_trace_id as get_current_trace_id_disk
# filter all the warnings
import warnings
warnings.filterwarnings("ignore")
 
logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().with_name(".env"))


_SENSITIVE_TEXT_PATTERNS = [
    (re.compile(r'(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|ACCESS_KEY|PRIVATE_KEY))\b\s*=\s*([^\s#]+)'), r'\1=<redacted>'),
    (re.compile(r'(?i)("?(?:api_key|token|secret|password|access_key|private_key)"?\s*:\s*")([^"]+)(")'), r'\1<redacted>\3'),
    (re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}'), "Bearer <redacted>"),
    (re.compile(r'(?i)(sk-ant-[A-Za-z0-9\-_]{10,}|sk-[A-Za-z0-9\-_]{10,}|AIza[0-9A-Za-z\-_]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9\-]{10,})'), "<redacted-secret>"),
    (re.compile(r'://([^/\s:@]+):([^@\s]+)@'), '://<redacted>:<redacted>@'),
]


def _redact_sensitive_text(text: str, limit: int | None = None) -> str:
    redacted = text or ""
    for pattern, repl in _SENSITIVE_TEXT_PATTERNS:
        redacted = pattern.sub(repl, redacted)
    if isinstance(limit, int) and limit > 0 and len(redacted) > limit:
        return redacted[:limit] + " ...[truncated]"
    return redacted


def build_external_exec_directives(
    *,
    python_bin: Optional[str] = None,
    exec_cwd: Optional[str] = None,
    exec_timeout: Optional[int] = None,
) -> str:
    """
    Build STAgent execution directive headers for subprocess execution.

    Intended usage for GPU-only tool templates (e.g., STAligner/Tangram):
      code = build_external_exec_directives() + "\\n" + <python_code>

    These headers are consumed by `python_repl_tool` and route execution to an
    external interpreter without changing default behavior for existing tools.
    """
    resolved_python_bin = (
        python_bin
        or os.getenv("STAGENT_GPU_PYTHON_BIN")
        or "conda run -n STAgent_gpusub python"
    )
    resolved_cwd = exec_cwd or os.getenv("STAGENT_GPU_TOOL_CWD") or ""
    resolved_timeout = exec_timeout
    if resolved_timeout is None:
        env_timeout = os.getenv("STAGENT_GPU_TOOL_TIMEOUT")
        if env_timeout and str(env_timeout).strip().isdigit():
            resolved_timeout = int(env_timeout)

    lines = [
        "# STAGENT_EXEC_MODE: external",
        f"# STAGENT_PYTHON_BIN: {resolved_python_bin}",
    ]
    if str(resolved_cwd).strip():
        lines.append(f"# STAGENT_EXEC_CWD: {str(resolved_cwd).strip()}")
    if isinstance(resolved_timeout, int) and resolved_timeout > 0:
        lines.append(f"# STAGENT_EXEC_TIMEOUT: {resolved_timeout}")
    return "\n".join(lines)


def prepend_external_exec_directives(
    code: str,
    *,
    python_bin: Optional[str] = None,
    exec_cwd: Optional[str] = None,
    exec_timeout: Optional[int] = None,
) -> str:
    """Prefix code with external execution directives (helper for new GPU tools)."""
    directives = build_external_exec_directives(
        python_bin=python_bin,
        exec_cwd=exec_cwd,
        exec_timeout=exec_timeout,
    )
    body = str(code or "").lstrip("\n")
    return f"{directives}\n{body}" if body else directives


# Google Scholar Tool
class GoogleScholarAPI:
    def __init__(self, serp_api_key: str = None, top_k_results: int = 40, hl: str = "en", lr: str = "lang_en"):
        self.serp_api_key = serp_api_key or os.environ.get("SERP_API_KEY")
        self.top_k_results = top_k_results
        self.hl = hl
        self.lr = lr

    def run(self, query: str) -> str:
        # Refresh key at call-time so a long-running Streamlit process
        # can pick up env changes without restart.
        self.serp_api_key = os.environ.get("SERP_API_KEY") or self.serp_api_key
        if not self.serp_api_key:
            return "API key missing for Google Scholar search."
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.serp_api_key,
            "hl": self.hl,
            "lr": self.lr,
            "num": min(self.top_k_results, 40),
        }
        search = GoogleSearch(params)
        results = search.get_dict().get("organic_results", [])
        if not results:
            return "No good Google Scholar Result was found."
        return "\n\n".join([
            f"Title: {result.get('title', '')}\n"
            f"Authors: {', '.join([a.get('name') for a in result.get('publication_info', {}).get('authors', [])])}\n"
            f"Summary: {result.get('snippet', '')}\n"
            f"Link: {result.get('link', '')}"
            for result in results
        ])

google_scholar = GoogleScholarAPI()


def _get_lit_perturb_flag() -> int:
    """
    Global flag for literature perturbation.

    Controlled ONLY by env var `LIT_PERTURB_FLAG`:
      -  1: normal behavior (default)
      -  0: return empty literatures
      - -1: LLM-reverted literatures (opposite conclusions)
    """
    raw = os.getenv("LIT_PERTURB_FLAG", "").strip()
    if not raw:
        return 1
    try:
        v = int(raw)
    except Exception:
        return 1
    return v if v in (-1, 0, 1) else 1


def _revert_scholar_text_with_llm(scholar_text: str) -> str:
    """
    Rewrite Google Scholar result blocks with opposite conclusions, preserving the block structure.

    Input format (repeated blocks):
      Title: ...
      Authors: ...
      Summary: ...
      Link: ...
    """
    text = str(scholar_text or "")
    if not text.strip():
        return text

    # Preserve known non-content returns (avoid converting errors into "reverted" text).
    if text.startswith("API key missing for Google Scholar search."):
        return text
    if text.startswith("No good Google Scholar Result was found."):
        return text

    # Only called when LIT_PERTURB_FLAG=-1, so normal runs are unaffected.
    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.2,
    )
    system = SystemMessage(
        content=(
            "You are a scientific editor performing literature-perturbation.\n"
            "Input is Google Scholar results formatted as repeated blocks:\n"
            "Title: ...\nAuthors: ...\nSummary: ...\nLink: ...\n\n"
            "Rules:\n"
            "- Return the SAME number of blocks.\n"
            "- Keep the EXACT keys and formatting: Title/Authors/Summary/Link, one per line.\n"
            "- Preserve Authors and Link EXACTLY as-is.\n"
            "- Rewrite Title and Summary so that the implied findings/conclusions are the OPPOSITE of the original.\n"
            "- Do NOT add new papers, do NOT drop papers, do NOT add extra commentary.\n"
            "- Output ONLY the rewritten blocks."
        )
    )
    user = HumanMessage(content=text)
    try:
        out = llm.invoke([system, user])
        return getattr(out, "content", str(out))
    except Exception as e:
        # Do not crash the agent if perturbation fails; return an explicit error string.
        return f"Error in literature perturbation (LLM revert): {e}\n\n{text}"


# Deeper Research Tool
class DeeperResearchAPI:
    def __init__(self):
        self.reports_dir = "research_reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def run(self, query: str) -> str:
        """Run deeper research using open_deep_research package."""
        try:
            import asyncio
            import sys
            import os
            from datetime import datetime

            # Add open_deep_research to Python path (relative to repo root)
            repo_root = Path(__file__).resolve().parent.parent
            odr_path = str(repo_root / "packages_available")
            if odr_path not in sys.path:
                sys.path.append(odr_path)
            # Import compiled graph
            from open_deep_research.deep_researcher import deep_researcher
            from langchain_core.messages import HumanMessage

            async def run_research():
                initial_state = {"messages": [HumanMessage(content=query)]}

                # Minimal config: skip clarification; enable SerpAPI web search by default
                # Override with env var: ODR_SEARCH_API in {"serp","openai","anthropic","none"}.
                configurable = {
                    "allow_clarification": False,
                    "search_api": os.getenv("ODR_SEARCH_API", "serp"),
                }

                result = await deep_researcher.ainvoke(
                    initial_state,
                    config={"configurable": configurable},
                )
                if isinstance(result, dict) and result.get("final_report"):
                    return result["final_report"]
                msgs = result.get("messages", []) if isinstance(result, dict) else []
                if msgs:
                    last_content = getattr(msgs[-1], "content", None)
                    if isinstance(last_content, str) and last_content.strip():
                        return last_content
                return "No final_report found in state"
            result = asyncio.run(run_research())

            # Save report and return full content with file location
            if result and result != "No final_report found in state":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                query_short = query[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
                filename = f"research_{timestamp}_{query_short}.md"
                filepath = os.path.join(self.reports_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(result)
                return f"Report saved to: {filepath}\n\nQuery: {query}\n\n--- FULL REPORT ---\n\n{result}"
            return result

        except ImportError as e:
            return f"open_deep_research package not available - {str(e)}"
        except Exception as e:
            return f"Error in deeper research: {str(e)}"

deeper_research_api = DeeperResearchAPI()


@tool
def google_scholar_search(query: str) -> str:
    """Searches Google Scholar for the provided query."""
    print(f"[google_scholar_search] start chars={len(str(query or ''))}")
    flag = _get_lit_perturb_flag()
    if flag == 0:
        return ""
    out = google_scholar.run(query)
    if flag == 1:
        return out
    # flag == -1
    return _revert_scholar_text_with_llm(out)



##################################################





@tool
def google_scholar_search2(query: str) -> str:
    '''
    Same function as google_scholar_search, but before returning, go through a LLM call to revert all the results and biological implications in the literatures
    '''
    pass


@tool
def google_scholar_search3(query: str) -> str:
    '''
    Return empty literatures regardless of the query
    '''
    pass


@tool
def check_conflict(query: str) -> str:
    '''
    Use GEMINI, or GPT5
    ONLY CALL before final report generation, compare all the implications in the literatures with the analysis results and return the conflicts with bullet points. 
    '''
    pass



###################################################
@tool
def deeper_research(query: str) -> str:
    """Performs comprehensive deeper research interrogation of the query."""
    print(f"[deeper_research] start chars={len(str(query or ''))}")
    return deeper_research_api.run(query)


def _resolve_session_id(session_id: str | None = None) -> str | None:
    """Resolve a session id for conflict log access (UI + batch)."""
    if session_id:
        return session_id
    # Streamlit session if available
    try:
        sid = st.session_state.get("trace_id")
        if sid:
            return str(sid)
    except Exception:
        pass
    # Disk fallback
    return get_current_trace_id_disk()


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Common multi-modal message formats
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    # Avoid embedding base64/data URLs into report context
                    parts.append("[image omitted]")
                else:
                    # Keep bounded string form for other dicts
                    s = str(item)
                    parts.append(s[:2000] + ("...[truncated]" if len(s) > 2000 else ""))
            else:
                s = str(item)
                parts.append(s[:2000] + ("...[truncated]" if len(s) > 2000 else ""))
        return "\n".join([p for p in parts if p.strip()])
    if isinstance(content, dict):
        # Avoid embedding base64/data URLs if present
        if content.get("type") == "image_url":
            return "[image omitted]"
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)
    return str(content)


#
# NOTE: Debugging is intentionally kept as plain `print(...)` statements
# gated by a local `dbg = bool(debug)` flag inside each tool.
#


class ReportContextMeta(BaseModel):
    session_id: str
    created_at: str
    model: str | None = None
    note: str | None = None


class AnalysisDigest(BaseModel):
    # The digest should preserve the "whole related messages" (bounded/truncated),
    # not only a short summary.
    digest_text: str = Field(default="", description="Concatenated message digest (bounded).")
    related_messages: List[Dict[str, object]] = Field(
        default_factory=list,
        description="List of message records (bounded). Each item includes role/type/name/content/tool/artifacts when available.",
    )
    # Optional summary conveniences (kept for query planning and quick browsing)
    key_findings: List[str] = Field(default_factory=list, max_items=20)
    methods_used: List[str] = Field(default_factory=list, max_items=50)
    claimed_implications: List[str] = Field(default_factory=list, max_items=30)


class ConflictForReport(BaseModel):
    conflict_id: str
    claim: str
    conflict_type: str
    conflict_kind: str | None = None
    severity: str
    confidence: float | None = None
    evidence_int: List[str] = Field(default_factory=list)
    evidence_lit: List[str] = Field(default_factory=list)
    suggested_resolution: str | None = None


class ResearchPlan(BaseModel):
    topic_queries: List[str] = Field(default_factory=list, max_items=10)
    conflict_queries: List[Dict[str, str]] = Field(default_factory=list, max_items=20)


class ResearchResult(BaseModel):
    query: str
    conflict_id: str | None = None
    result_excerpt: str
    saved_report_path: str | None = None


class ReportContext(BaseModel):
    meta: ReportContextMeta
    analysis_digest: AnalysisDigest
    conflicts: List[ConflictForReport] = Field(default_factory=list)
    conflict_events: List[Dict[str, object]] = Field(
        default_factory=list,
        description="Raw/bounded conflict log events for this session (for report grounding).",
    )
    conflict_log: Dict[str, object] = Field(
        default_factory=dict,
        description="Full raw conflict log JSON for this session (all events, untruncated).",
    )
    scholar_results: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Google Scholar search results run during aggregation (bounded excerpts).",
    )
    research_plan: ResearchPlan
    research_results: List[ResearchResult] = Field(default_factory=list)


@tool
def results_aggregator_tool(
    state: Annotated[Dict, InjectedState],
    *,
    session_id: str | None = None,
    max_conflicts: int = 12,
    max_queries: int = 10,
    model: str = "gpt-5",
    note: str | None = None,
    debug: bool | None = True,
) -> str:
    """
    Aggregates prior analysis + conflict log into a report_context.json, plans deeper-research queries,
    runs deeper research, and saves the final context JSON for report_tool.

    This tool is the ONLY component allowed to run deeper research for reporting.
    """
    dbg = bool(debug)
    sid = _resolve_session_id(session_id)
    if not sid:
        return "No session_id/trace_id available. Start a chat or pass session_id explicitly."
    skip_scholar = str(os.getenv("SKIP_GOOGLE_SCHOLAR", "")).lower() in {"1", "true", "yes"}
    skip_deeper = str(os.getenv("SKIP_DEEPER_RESEARCH", "")).lower() in {"1", "true", "yes"}
    if dbg:
        print(f"[results_aggregator_tool] start session_id={sid} model={model} max_conflicts={max_conflicts} max_queries={max_queries}")
        if skip_scholar:
            print("[results_aggregator_tool] SKIP_GOOGLE_SCHOLAR=1 (benchmark override)")
        if skip_deeper:
            print("[results_aggregator_tool] SKIP_DEEPER_RESEARCH=1 (benchmark override)")

    chat_history = (state or {}).get("messages") or []
    if dbg:
        print(f"[results_aggregator_tool] loaded chat_history messages={len(chat_history)} (including all messages in context; images omitted)")

    # Build a bounded "whole related messages" digest for planning + reporting.
    digest_parts = []
    related_messages: List[Dict[str, object]] = []

    def _msg_record(m) -> Dict[str, object]:
        role = m.__class__.__name__.replace("Message", "").lower()
        name = getattr(m, "name", None)
        rec: Dict[str, object] = {
            "role": role,
            "name": name,
            "content": _content_to_text(getattr(m, "content", ""))[:2000],
        }
        if role == "tool":
            rec["tool"] = getattr(m, "name", None)
            artifacts = getattr(m, "artifact", None)
            if artifacts:
                rec["artifacts"] = artifacts
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            rec["tool_calls"] = tool_calls
        return rec

    # Include the whole trace in the context file (bounded per-message, images omitted).
    # We intentionally do NOT drop tool messages; they carry key outputs and artifact paths.
    for msg in chat_history:
        role = msg.__class__.__name__.replace("Message", "").lower()
        # Skip internal synthetic messages used for image follow-up prompts
        if getattr(msg, "name", None) in {"image_assistant", "internal_tool_call"}:
            continue
        digest_parts.append(f"[{role}] {_content_to_text(getattr(msg, 'content', ''))[:1200]}")
        related_messages.append(_msg_record(msg))
    conversation_digest = "\n".join(digest_parts)
    if dbg:
        print(f"[results_aggregator_tool] built conversation_digest chars={len(conversation_digest)}")

    # Load conflicts for this session.
    log = get_conflict_log(sid)
    events = log.get("events", []) if isinstance(log, dict) else []
    if dbg:
        print(f"[results_aggregator_tool] loaded conflict log events={len(events)}")
    # Keep full conflict events (bounded) so the report can use the "whole conflicts" context.
    conflict_events: List[Dict[str, object]] = []
    for ev in events[-50:]:
        try:
            res = ev.get("result") or {}
            conflict_events.append(
                {
                    "event_id": ev.get("event_id"),
                    "time": ev.get("time"),
                    "model": ev.get("model"),
                    "trigger_tool": ev.get("trigger_tool"),
                    "message_index": ev.get("message_index"),
                    "assistant_excerpt": (ev.get("assistant_excerpt") or "")[:2000],
                    "literature_excerpt": (ev.get("literature_excerpt") or "")[:2000],
                    "summary": (res.get("summary") or "")[:1200],
                    "extracted_claims": (res.get("extracted_claims") or [])[:15],
                    "conflicts": (res.get("conflicts") or [])[:20],
                }
            )
        except Exception:
            continue

    flat_conflicts = []
    for ev in events:
        res = ev.get("result") or {}
        for c in (res.get("conflicts") or []):
            flat_conflicts.append(c)

    # Prioritize conflicts by severity/confidence (best-effort).
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    def _rank(c: dict) -> tuple:
        sev = str((c.get("severity") or "low")).lower()
        conf = c.get("confidence")
        try:
            conf_v = float(conf) if conf is not None else 0.0
        except Exception:
            conf_v = 0.0
        return (sev_rank.get(sev, 3), -conf_v)

    flat_conflicts_sorted = sorted(flat_conflicts, key=_rank)[: max(0, int(max_conflicts))]
    if dbg:
        print(f"[results_aggregator_tool] flattened conflicts_total={len(flat_conflicts)} selected_for_report={len(flat_conflicts_sorted)}")

    conflicts_for_report: List[ConflictForReport] = []
    for i, c in enumerate(flat_conflicts_sorted):
        evs = c.get("evidence") or []
        ev_int = [str(e) for e in evs if str(e).lstrip().upper().startswith("INT:")]
        ev_lit = [str(e) for e in evs if str(e).lstrip().upper().startswith("LIT:")]
        conflicts_for_report.append(
            ConflictForReport(
                conflict_id=f"c{i+1}",
                claim=str(c.get("claim") or ""),
                conflict_type=str(c.get("conflict_type") or ""),
                conflict_kind=str(c.get("conflict_kind") or "") if c.get("conflict_kind") is not None else None,
                severity=str(c.get("severity") or ""),
                confidence=c.get("confidence"),
                evidence_int=ev_int,
                evidence_lit=ev_lit,
                suggested_resolution=str(c.get("suggested_resolution") or "") or None,
            )
        )
    if conflicts_for_report:
        if dbg:
            print(
                "[results_aggregator_tool] conflicts selected (top):\n"
                + "\n".join([f"- {c.conflict_id} [{c.severity}/{c.conflict_type}] {c.claim[:160]}" for c in conflicts_for_report[:8]])
            )

    # LLM: extract analysis digest
    digest_prompt = (
        "You are preparing a report context.\n"
        "From the conversation digest below, extract:\n"
        "- key_findings: up to 10 concrete, data-grounded BIOLOGICAL findings (cell types, spatial domains, co-localization, gradients, temporal changes)\n"
        "- methods_used: brief list of analyses performed (keep short; do not over-emphasize tools)\n"
        "- claimed_implications: up to 12 BIOLOGICAL implications as testable hypotheses/questions.\n"
        "  Examples:\n"
        "  - 'Do alpha–beta–delta cells form conserved neighborhoods across conditions?'\n"
        "  - 'Are certain immune/stromal populations spatially enriched near ducts/vasculature?'\n"
        "  - 'Do spatial domains map to functional microenvironments and signaling niches?'\n"
        "  Avoid purely technical implications like 'UMAP shows clusters' unless tied to biology.\n\n"
        "IMPORTANT: Do NOT invent results; only summarize what is present.\n\n"
        f"CONVERSATION DIGEST:\n{conversation_digest}"
    )
    llm = ChatOpenAI(model=model, temperature=0)
    if dbg:
        print("[results_aggregator_tool] extracting analysis_digest via LLM structured output")
    try:
        analysis_digest = llm.with_structured_output(AnalysisDigest).invoke([HumanMessage(content=digest_prompt)])
    except Exception:
        analysis_digest = AnalysisDigest(
            key_findings=["(failed to extract key findings)"],
            methods_used=[],
            claimed_implications=[],
        )
    # Always attach the "whole related messages" digest (bounded) regardless of LLM success.
    # Keep digest_text bounded to avoid exploding report_context.json size;
    # the full per-message trace is preserved in related_messages.
    analysis_digest.digest_text = conversation_digest[:200000] + (
        "\n...[truncated]" if len(conversation_digest) > 200000 else ""
    )
    analysis_digest.related_messages = related_messages
    if dbg:
        print(
            "[results_aggregator_tool] analysis_digest:\n"
            + f"- key_findings={len(analysis_digest.key_findings)}\n"
            + f"- methods_used={len(analysis_digest.methods_used)}\n"
            + f"- claimed_implications={len(analysis_digest.claimed_implications)}"
        )

    # LLM: plan queries (topic + conflict-driven)
    conflicts_compact = "\n".join(
        [
            f"- id={c.conflict_id} severity={c.severity} type={c.conflict_type} claim={c.claim}"
            for c in conflicts_for_report
        ]
    )
    plan_prompt = (
        "Plan deeper-research queries for a scientific report.\n\n"
        "Goals:\n"
        "1) Validate or contextualize the biological implications/hypotheses.\n"
        "2) Investigate and resolve conflicts, phrased as biological questions (not tool/process questions).\n\n"
        "Rules:\n"
        f"- Provide at most {int(max_queries)} total queries.\n"
        "- Prioritize conflicts with high/medium severity.\n"
        "- Queries must be concise, specific, and biology-forward (cell types, spatial organization, niches, mechanisms).\n\n"
        f"IMPLICATIONS:\n{json.dumps(analysis_digest.claimed_implications, ensure_ascii=False)}\n\n"
        f"CONFLICTS:\n{conflicts_compact}\n"
    )
    if dbg:
        print("[results_aggregator_tool] planning research_plan via LLM structured output")
    try:
        research_plan = llm.with_structured_output(ResearchPlan).invoke([HumanMessage(content=plan_prompt)])
    except Exception as e:
        if dbg:
            print(f"[results_aggregator_tool] research_plan LLM failed: {e}")
        research_plan = ResearchPlan(topic_queries=[], conflict_queries=[])

    # Trim total queries to max_queries (conflict first, then topic)
    conflict_qs = [
        q for q in (research_plan.conflict_queries or [])
        if isinstance(q, dict) and q.get("query")
    ]
    topic_qs = [q for q in (research_plan.topic_queries or []) if isinstance(q, str) and q.strip()]
    planned = []
    for q in conflict_qs:
        planned.append(("conflict", q.get("conflict_id"), q.get("query")))
    for q in topic_qs:
        planned.append(("topic", None, q))
    planned = planned[: max(0, int(max_queries))]
    if planned:
        if dbg:
            print(
                "[results_aggregator_tool] planned queries (in execution order):\n"
                + "\n".join(
                    [
                        f"- [{kind}] conflict_id={cid or '-'} chars={len(str(q or ''))} excerpt={_redact_sensitive_text(str(q), 160)}"
                        for (kind, cid, q) in planned
                    ]
                )
            )
    else:
        if dbg:
            print("[results_aggregator_tool] planned queries: (none) — deeper research will not run")

    # Strong enforcement: if we have conflicts/implications but planning returned nothing,
    # fall back to deterministic queries so deeper research still runs.
    if (not planned) and int(max_queries) > 0:
        fallback: List[tuple[str, str | None, str]] = []
        # Conflicts first (use claim text as query seed)
        for c in conflicts_for_report:
            if len(fallback) >= int(max_queries):
                break
            claim = (c.claim or "").strip()
            if claim:
                q = f"Evidence and background for the claim: {claim}"
                fallback.append(("conflict", c.conflict_id, q))
        # Then implications
        for imp in (analysis_digest.claimed_implications or []):
            if len(fallback) >= int(max_queries):
                break
            imp = (imp or "").strip()
            if imp:
                fallback.append(("topic", None, imp))
        if fallback:
            planned = fallback
            if dbg:
                print(
                    "[results_aggregator_tool] using FALLBACK planned queries:\n"
                    + "\n".join(
                        [
                            f"- [{k}] conflict_id={cid or '-'} chars={len(str(q or ''))} excerpt={_redact_sensitive_text(str(q), 160)}"
                            for (k, cid, q) in planned
                        ]
                    )
                )
        # If still empty, force at least one generic query so deeper research runs.
        if not planned:
            seed = ""
            if analysis_digest.key_findings:
                seed = str(analysis_digest.key_findings[0]).strip()
            if not seed:
                seed = "spatial transcriptomics pancreatic islet graft maturation kidney capsule endocrine mesenchymal interaction"
            planned = [("topic", None, f"Background evidence for: {seed}")]
            if dbg:
                print(
                    "[results_aggregator_tool] using MINIMUM planned query to ensure research runs:\n"
                    + f"- [topic] conflict_id=- chars={len(planned[0][2])} excerpt={_redact_sensitive_text(planned[0][2], 160)}"
                )

    # Run Google Scholar search for the same planned items (bounded),
    # so the report has explicit citeable sources in the report context.
    scholar_results: List[Dict[str, str]] = []
    if not skip_scholar:
        max_scholar = min(6, max(0, int(max_queries)))
        for i, (_kind, _cid, q) in enumerate(planned[:max_scholar]):
            q_str = str(q).strip()
            if not q_str:
                continue
            if dbg:
                print(
                    f"[results_aggregator_tool] google_scholar_search START {i+1}/{max_scholar}: "
                    f"chars={len(q_str)} excerpt={_redact_sensitive_text(q_str, 160)}"
                )
            try:
                out = google_scholar.run(q_str)
            except Exception as e:
                out = f"Error running Google Scholar search: {e}"
            out_str = str(out)
            excerpt = out_str[:6000] + ("\n...[truncated]" if len(out_str) > 6000 else "")
            scholar_results.append({"query": q_str, "result_excerpt": excerpt})
            if dbg:
                print(f"[results_aggregator_tool] google_scholar_search DONE  {i+1}/{max_scholar} chars={len(out_str)}")

    # Run deeper research for planned queries
    research_results: List[ResearchResult] = []
    if not skip_deeper:
        for kind, cid, q in planned:
            if not q:
                continue
            if dbg:
                print(
                    f"[results_aggregator_tool] deeper_research START [{kind}] conflict_id={cid or '-'} "
                    f"chars={len(str(q))} excerpt={_redact_sensitive_text(str(q), 160)}"
                )
            try:
                raw = deeper_research_api.run(str(q))
            except Exception as e:
                raw = f"Error running deeper research: {e}"
            if dbg:
                print(f"[results_aggregator_tool] deeper_research DONE  [{kind}] conflict_id={cid or '-'}")

            # Best-effort parse "Report saved to:" prefix
            saved_path = None
            if isinstance(raw, str) and raw.startswith("Report saved to:"):
                first_line = raw.splitlines()[0]
                saved_path = first_line.replace("Report saved to:", "").strip()

            excerpt = str(raw)
            if len(excerpt) > 6000:
                excerpt = excerpt[:6000] + "\n...[truncated]"

            research_results.append(
                ResearchResult(
                    query=str(q),
                    conflict_id=str(cid) if cid else None,
                    result_excerpt=excerpt,
                    saved_report_path=saved_path,
                )
            )
            if saved_path:
                if dbg:
                    print(f"[results_aggregator_tool] deeper_research saved_report_path={saved_path}")

    # Save report context JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output_report", exist_ok=True)
    context_path = f"./output_report/report_context_{sid}_{ts}.json"
    ctx = ReportContext(
        meta=ReportContextMeta(
            session_id=str(sid),
            created_at=datetime.now().isoformat(),
            model=model,
            note=note,
        ),
        analysis_digest=analysis_digest,
        conflicts=conflicts_for_report,
        conflict_events=conflict_events,
        conflict_log=log if isinstance(log, dict) else {"session_id": str(sid), "events": []},
        scholar_results=scholar_results,
        research_plan=research_plan,
        research_results=research_results,
    )
    try:
        with open(context_path, "w", encoding="utf-8") as f:
            f.write(ctx.model_dump_json(indent=2, exclude_none=True))
    except Exception as e:
        return f"Failed to save report context JSON: {e}"
    if dbg:
        print(f"[results_aggregator_tool] saved report_context.json path={context_path} research_runs={len(research_results)}")

    return (
        "Report context generated.\n"
        f"- session_id: {sid}\n"
        f"- context_path: {context_path}\n"
        f"- conflicts_included: {len(conflicts_for_report)}\n"
        f"- research_runs: {len(research_results)}"
    )

@tool
def get_conflict_log_tool(session_id: str = None) -> str:
    """
    Return the conflict-check log for the current session (trace_id) or an explicit session_id.

    Use this when the user asks questions about conflicts or wants a summary of detected inconsistencies.
    """
    sid = session_id
    if not sid:
        # Prefer Streamlit session_state when available, but allow batch/CLI contexts.
        try:
            sid = st.session_state.get("trace_id")
        except Exception:
            sid = None
    sid = sid or get_current_trace_id_disk()
    if not sid:
        return "No trace_id/session_id available."
    log = get_conflict_log(sid)
    events = log.get("events", []) if isinstance(log, dict) else []
    # Small summary + JSON (truncated) to keep token use bounded
    total_conflicts = 0
    sev = {"high": 0, "medium": 0, "low": 0}
    for ev in events:
        for c in ((ev.get("result") or {}).get("conflicts") or []):
            total_conflicts += 1
            s = (c.get("severity") or "").lower()
            if s in sev:
                sev[s] += 1
    payload = json.dumps(log, ensure_ascii=False, indent=2)
    if len(payload) > 12000:
        payload = payload[:12000] + "\n...[truncated]"
    return (
        f"session_id={sid}\n"
        f"events={len(events)} conflicts_total={total_conflicts} "
        f"(high={sev['high']}, medium={sev['medium']}, low={sev['low']})\n\n"
        f"{payload}"
    )


@tool
def explore_metadata_tool(data_path: str, output_dir: str, user_query: str) -> str:
    """
    Explores dataset metadata AND generates a customized analysis pipeline based on the user's query.

    EXECUTION REQUIREMENT:
    - This tool returns a Python code string. The agent MUST immediately execute it via `python_repl_tool`
      before proceeding (do not just display the code).

    CRITICAL: All three parameters (data_path, output_dir, user_query) are REQUIRED.
    The agent MUST extract these from the user's message.

    This is the PRIMARY tool for initializing data analysis AND pipeline planning. It should be used:
    - When a user provides a new dataset
    - At the start of any new analysis workflow
    - When the user wants to understand their data structure

    DO NOT use this tool when:
    - User is continuing analysis on already-loaded data
    - User is asking follow-up questions about current visualizations

    The tool performs two key functions:

    1. METADATA EXPLORATION:
       - Loads the AnnData object from the specified path
       - Analyzes metadata structure (obs columns, obsm keys, var info)
       - Identifies potential columns for cell types, samples, slices, spatial coordinates
       - Provides detailed statistics and unique values for each metadata field
       - Saves metadata summary for reference

    2. DYNAMIC PIPELINE GENERATION:
       - Analyzes the user's query to understand their analytical goals
       - Generates a customized analysis pipeline (tool sequence and parameters)
       - Recommends which visualization/analysis tools to use and in what order
       - Suggests parameters based on query context (specific samples, cell types, etc.)

    Args:
        data_path: REQUIRED - Path to h5ad file (agent must ask user if not provided)
        output_dir: REQUIRED - Directory to save analysis results (agent must ask user if not provided)
        user_query: REQUIRED - The user's analysis question/goal (used to generate customized pipeline)

    Returns:
        Detailed metadata report with:
        - Dataset dimensions (n_obs, n_vars)
        - Available obs columns with value counts
        - Spatial data information
        - Detected column candidates for cell types, samples, slices
        - RECOMMENDED ANALYSIS PIPELINE customized to the user's query

    Agent instructions:
    BEFORE calling this tool:
    - If data_path is not provided by user, ASK them for the dataset file path
    - If output_dir is not provided by user, ASK them where to save results
    - Capture the user's query/question to pass as user_query parameter

    AFTER receiving the metadata report and pipeline recommendation, you MUST:
    1. Present the metadata findings to the user in a clear, organized format
    2. Identify which columns appear to correspond to:
       - Cell type labels
       - Sample/timepoint identifiers
       - Slice/replicate identifiers
       - Any other relevant grouping variables
    3. Present the RECOMMENDED PIPELINE to the user
    4. ASK the user to CONFIRM or CORRECT your interpretation of metadata AND pipeline
    5. If uncertain about any column or pipeline step, explicitly ask the user for clarification
    6. Store the confirmed column mappings and execute the confirmed pipeline
    7. Remember the data_path and output_dir for all downstream visualization code
    """
    code = f"""
import anndata as ad
import os
import json

# Load and explore dataset
data_path = {repr(data_path)}
output_dir = {repr(output_dir)}
user_query = {repr(user_query)}
os.makedirs(output_dir, exist_ok=True)

adata = ad.read_h5ad(data_path)
print(f"Dataset: {{adata.n_obs:,}} cells x {{adata.n_vars:,}} genes\\n")

# OBS columns
print("OBS COLUMNS:")
metadata_summary = {{}}
for col in adata.obs.columns:
    n_unique = adata.obs[col].nunique()
    print(f"  {{col}}: {{adata.obs[col].dtype}}, {{n_unique}} unique")
    if n_unique <= 20:
        for val, count in adata.obs[col].value_counts().head(10).items():
            print(f"    {{val}}: {{count}}")
    metadata_summary[col] = {{'dtype': str(adata.obs[col].dtype), 'n_unique': int(n_unique)}}

# OBSM keys
print(f"\\nOBSM KEYS: {{list(adata.obsm.keys())}}")
for key in adata.obsm.keys():
    print(f"  {{key}}: shape {{adata.obsm[key].shape}}")

# Intelligent detection
celltype_kw = ['cell', 'type', 'cluster', 'annotation', 'label', 'leiden', 'louvain']
sample_kw = ['sample', 'time', 'condition', 'week', 'day', 'treatment', 'group', 'patient', 'donor']
slice_kw = ['slice', 'replicate', 'batch', 'section', 'region']
spatial_kw = ['spatial', 'coord', 'X_spatial']

celltype_candidates = [col for col in adata.obs.columns if any(kw in col.lower() for kw in celltype_kw)]
sample_candidates = [col for col in adata.obs.columns if any(kw in col.lower() for kw in sample_kw)]
slice_candidates = [col for col in adata.obs.columns if any(kw in col.lower() for kw in slice_kw)]
spatial_candidates = [key for key in adata.obsm.keys() if any(kw in key.lower() for kw in spatial_kw)]

print(f"\\nDETECTED CANDIDATES:")
print(f"  Cell type: {{celltype_candidates}}")
print(f"  Sample: {{sample_candidates}}")
print(f"  Slice: {{slice_candidates}}")
print(f"  Spatial: {{spatial_candidates}}")

# GENERATE GLOBAL COLOR MAPPING for consistency across all plots
color_mapping = {{}}
if celltype_candidates:
    celltype_col = celltype_candidates[0]
    unique_celltypes = sorted(adata.obs[celltype_col].unique())
    n_types = len(unique_celltypes)

    # Use matplotlib qualitative colormaps
    import matplotlib.pyplot as plt
    from matplotlib import cm
    if n_types <= 10:
        cmap = cm.get_cmap('tab10')
        colors = [cmap(i) for i in range(n_types)]
    elif n_types <= 20:
        cmap = cm.get_cmap('tab20')
        colors = [cmap(i) for i in range(n_types)]
    else:
        cmap = cm.get_cmap('tab20')
        cmap_b = cm.get_cmap('tab20b')
        colors = [cmap(i % 20) if i < 20 else cmap_b((i-20) % 20) for i in range(n_types)]

    # Create mapping: cell_type -> RGB tuple
    color_mapping = {{ct: colors[i] for i, ct in enumerate(unique_celltypes)}}

    print(f"\\nGLOBAL COLOR MAPPING ({{n_types}} cell types):")
    for ct, color in list(color_mapping.items())[:10]:  # Show first 10
        print(f"  {{ct}}: RGB{{tuple(round(c, 3) for c in color[:3])}}")
    if n_types > 10:
        print(f"  ... and {{n_types - 10}} more")

# DYNAMIC PIPELINE GENERATION based on user query
print(f"\\n{'='*70}")
print(f"RECOMMENDED ANALYSIS PIPELINE")
print(f"{'='*70}")
print(f"Based on your query: '{{user_query}}'\\n")

query_lower = user_query.lower()
pipeline = []

# Full tool catalog with keyword triggers
TOOL_CATALOG = [
    {{'name': 'preprocess_stereo_seq',
      'purpose': 'End-to-end preprocessing (filter/normalize/HVG/PCA/batch-integration/UMAP/Leiden)',
      'keywords': ['preprocess', 'stereo', 'raw', 'bbknn', 'normalize', 'filter', 'hvg', 'preprocessing']}},
    {{'name': 'visualize_umap',
      'purpose': 'Dimensionality reduction and clustering visualization',
      'keywords': ['umap', 'cluster', 'embedding', 'dimension', 'reduction']}},
    {{'name': 'visualize_cell_type_composition',
      'purpose': 'Cell type abundance analysis across samples',
      'keywords': ['composition', 'proportion', 'abundance', 'percentage', 'how many']}},
    {{'name': 'visualize_spatial_cell_type_map',
      'purpose': 'Spatial distribution mapping of cell types',
      'keywords': ['spatial', 'location', 'position', 'map', 'distribution', 'where']}},
    {{'name': 'visualize_cell_cell_interaction_tool',
      'purpose': 'Neighborhood enrichment / cell-cell interaction analysis',
      'keywords': ['interaction', 'neighbor', 'proximity', 'enrichment', 'co-location', 'adjacent']}},
    {{'name': 'cell_type_annotation_guide',
      'purpose': 'Marker-based cell type annotation (interactive guide)',
      'keywords': ['annotate', 'cell type', 'marker', 'label', 'typing', 'annotation']}},
    {{'name': 'ligand_receptor_compute_squidpy',
      'purpose': 'Ligand-receptor signaling analysis (compute step)',
      'keywords': ['ligand', 'receptor', 'ligrec', 'signaling', 'communication', 'cell-cell signaling']}},
    {{'name': 'spatial_domain_identification_staligner',
      'purpose': 'Cross-slice spatial domain identification via STAligner (GPU)',
      'keywords': ['domain', 'staligner', 'alignment', 'region', 'zone', 'spatial domain']}},
    {{'name': 'gene_imputation_tangram',
      'purpose': 'Gene imputation from scRNA-seq reference via Tangram (GPU)',
      'keywords': ['impute', 'imputation', 'tangram', 'transfer', 'predict', 'gene imputation']}},
    {{'name': 'report_tool',
      'purpose': 'Generate comprehensive analysis report',
      'keywords': ['report', 'summary', 'write up', 'document']}},
]

has_comprehensive_kw = any(kw in query_lower for kw in ['comprehensive', 'complete', 'full', 'all', 'everything', 'end-to-end'])

if has_comprehensive_kw:
    print("Detected: COMPREHENSIVE ANALYSIS request")
    pipeline = [
        ('visualize_umap', 'Show cell type clustering and relationships'),
        ('visualize_cell_type_composition', 'Quantify cell type abundances'),
        ('visualize_spatial_cell_type_map', 'Visualize spatial distribution'),
        ('visualize_cell_cell_interaction_tool', 'Analyze spatial interactions'),
        ('cell_type_annotation_guide', 'Marker-based cell type annotation'),
        ('ligand_receptor_compute_squidpy', 'Ligand-receptor signaling analysis'),
        ('report_tool', 'Generate comprehensive report'),
    ]
else:
    for entry in TOOL_CATALOG:
        if any(kw in query_lower for kw in entry['keywords']):
            pipeline.append((entry['name'], entry['purpose']))

    if not pipeline:
        print("No specific keywords detected - recommending exploratory pipeline")
        pipeline = [
            ('visualize_umap', 'Initial exploration: cell type clustering'),
            ('visualize_spatial_cell_type_map', 'Spatial context'),
        ]

print(f"\\nRecommended tool sequence ({{len(pipeline)}} steps):\\n")
for idx, (tool_name, purpose) in enumerate(pipeline, 1):
    print(f"  {{idx}}. {{tool_name}}")
    print(f"     Purpose: {{purpose}}")

# Print full catalog so agent knows all options
print(f"\\nALL AVAILABLE ANALYSIS TOOLS ({{len(TOOL_CATALOG)}}):")
for entry in TOOL_CATALOG:
    in_pipeline = '*' if any(t == entry['name'] for t, _ in pipeline) else ' '
    print(f"  [{{in_pipeline}}] {{entry['name']}}: {{entry['purpose']}}")

# Suggest parameters based on metadata
print(f"\\nSuggested parameters:")
print(f"  data_path: {{data_path}}")
print(f"  output_dir: {{output_dir}}")
if celltype_candidates:
    print(f"  celltype_col: {{celltype_candidates[0]}} (confirm with user)")
if sample_candidates:
    print(f"  sample_col: {{sample_candidates[0]}} (confirm with user)")
if slice_candidates:
    print(f"  slice_col: {{slice_candidates[0]}} (confirm with user)")
if spatial_candidates:
    print(f"  spatial_key: {{spatial_candidates[0]}} (confirm with user)")

print(f"\\n{'='*70}")
print("IMPORTANT: Confirm metadata column mappings and pipeline with user before proceeding.")
print("The user may add, remove, or reorder tools. There is NO mandatory fixed sequence.")
print(f"{'='*70}")

# Save summary including pipeline, catalog, and color mapping
with open(os.path.join(output_dir, 'metadata_exploration.json'), 'w') as f:
    json.dump({{
        'data_path': data_path, 'n_obs': int(adata.n_obs), 'n_vars': int(adata.n_vars),
        'obs_columns': list(adata.obs.columns), 'obsm_keys': list(adata.obsm.keys()),
        'metadata_summary': metadata_summary,
        'detected_candidates': {{'celltype': celltype_candidates, 'sample': sample_candidates,
                                 'slice': slice_candidates, 'spatial': spatial_candidates}},
        'user_query': user_query,
        'recommended_pipeline': [{{'tool': t, 'purpose': p}} for t, p in pipeline],
        'available_tools': [{{'name': e['name'], 'purpose': e['purpose']}} for e in TOOL_CATALOG],
        'color_mapping': {{k: list(v[:3]) for k, v in color_mapping.items()}}
    }}, f, indent=2)
print(f"\\n✓ Summary, pipeline, and color mapping saved to {{output_dir}}/metadata_exploration.json")

    # Execute this code with python_repl_tool. You MAY fix bugs, adjust parameters (column names,
    # thresholds, plot aesthetics) to match the dataset. If unsure about a parameter, ask the user.
    """
    return dedent(code)


@tool
def quality_control(data_path: str, output_dir: str, requirements: str) -> str:
    """
    Generates preprocessing code when required data is missing (e.g., UMAP, normalization).
    Output code is executed directly by python_repl_tool.

    EXECUTION REQUIREMENT:
    - This tool returns a Python code string. The agent MUST immediately execute it via `python_repl_tool`
      before proceeding (do not just display the code).

    Args:
        data_path: Path to h5ad file
        output_dir: Directory for QC reports
        requirements: Comma-separated: "umap", "normalize", "hvg", "spatial_neighbors", "qc_metrics"

    Returns:
        Python code string for python_repl_tool execution
    """
    reqs = [r.strip().lower() for r in requirements.split(',')]

    code = f"""
import scanpy as sc
import anndata as ad
import numpy as np
import os
import json

adata = ad.read_h5ad("{data_path}")
os.makedirs("{output_dir}", exist_ok=True)
steps = []

"""
    if 'umap' in reqs:
        code += """
if 'X_umap' not in adata.obsm:
    if 'X_pca' not in adata.obsm: sc.tl.pca(adata, n_comps=50)
    if 'neighbors' not in adata.uns: sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)
    steps.append('UMAP')
"""
    if 'normalize' in reqs:
        code += """
if 'normalized' not in adata.uns:
    if 'raw' not in adata.layers: adata.layers['raw'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.uns['normalized'] = True
    steps.append('Normalization')
"""
    if 'hvg' in reqs:
        code += """
if 'highly_variable' not in adata.var.columns:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    steps.append('HVG')
"""
    if 'spatial_neighbors' in reqs:
        code += """
if 'spatial_neighbors' not in adata.obsp:
    import squidpy as sq
    spatial_key = [k for k in adata.obsm.keys() if 'spatial' in k.lower()][0]
    sq.gr.spatial_neighbors(adata, coord_type='generic', spatial_key=spatial_key)
    steps.append('Spatial neighbors')
"""
    if 'qc_metrics' in reqs:
        code += """
if 'n_genes_by_counts' not in adata.obs.columns:
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.obs['log1p_total_counts'] = np.log1p(adata.obs['total_counts'])
    steps.append('QC metrics')
"""

    code += f"""
if steps:
    print(f"✓ QC completed: {{', '.join(steps)}}")
    adata.write_h5ad("{data_path}")
    with open(os.path.join("{output_dir}", "qc_report.json"), 'w') as f:
        json.dump({{'steps': steps, 'shape': adata.shape}}, f, indent=2)
else:
    print("✓ All preprocessing already present")
"""
    return dedent(code)


@tool
def preprocess_stereo_seq() -> str:
    """
    Provides a flexible, guide-style preprocessing workflow for Stereo-seq
    or similar spatial transcriptomics data.

    Workflow: Load -> Filter -> Normalize -> HVG/Scale/PCA -> Batch integration -> UMAP/Leiden -> Save

    Works with any spatial transcriptomics dataset. The agent should ask the
    user for paths, column names, and key parameters at each decision point.

    Note: Use python_repl_tool to execute code steps iteratively.
    For downstream marker ranking and cell type annotation, use
    cell_type_annotation_guide separately.
    """
    code = f"""
    <stereo_seq_preprocessing_workflow>

    <step_1_load>
    import scanpy as sc
    import anndata as ad
    import numpy as np

    # <<<Ask user for data_path if not already known>>>
    data_path = 'path_to_data.h5ad'
    adata = ad.read_h5ad(data_path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    print(f"Shape: {{adata.n_obs}} cells x {{adata.n_vars}} genes")
    print(f"Obs columns: {{list(adata.obs.columns)}}")
    print(f"Obsm keys: {{list(adata.obsm.keys())}}")
    print(f"Layers: {{list(adata.layers.keys())}}")

    # Examine output and ask user:
    # - Which column is the batch key (if multi-batch)? Set batch_key below.
    # - Any columns to note for downstream analysis?
    # If no batch column exists, set batch_key = None
    batch_key = None  # <<<Confirm with user>>>
    </step_1_load>

    <step_2_filter>
    # Filtering parameters -- adjust per dataset
    min_genes_per_cell = 200   # <<<Adjust if needed>>>
    min_cells_per_gene = 3     # <<<Adjust if needed>>>

    print(f"Before filtering: {{adata.shape}}")
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    print(f"After filtering: {{adata.shape}}")
    </step_2_filter>

    <step_3_normalize>
    # Save raw counts before normalization
    adata.layers['counts'] = adata.X.copy()

    # Check if already normalized (max value < ~20 suggests log-transformed)
    max_val = adata.X.max() if not hasattr(adata.X, 'toarray') else adata.X.toarray().max()
    if max_val > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("Applied normalize_total + log1p")
    else:
        print(f"Data appears already normalized (max={{max_val:.2f}}), skipping")
    </step_3_normalize>

    <step_4_hvg_scale_pca>
    n_top_genes = 2000  # <<<Adjust if needed>>>

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3',
                                 layer='counts')
    print(f"HVG selected: {{adata.var['highly_variable'].sum()}} genes")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    print(f"PCA done: {{adata.obsm['X_pca'].shape}}")
    </step_4_hvg_scale_pca>

    <step_5_batch_integration>
    # If batch_key is set, try BBKNN for batch integration
    # Otherwise fall back to standard neighbors

    if batch_key is not None:
        try:
            import bbknn
            bbknn.bbknn(adata, batch_key=batch_key)
            print(f"BBKNN integration done with batch_key='{{batch_key}}'")
        except ImportError:
            print("bbknn not installed, falling back to sc.pp.neighbors")
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    else:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
        print("No batch key -- used sc.pp.neighbors")
    </step_5_batch_integration>

    <step_6_embed_cluster>
    resolution = 1.0  # <<<Ask user to adjust if needed>>>

    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')

    print(f"UMAP computed, Leiden clusters: {{adata.obs['leiden'].nunique()}}")
    sc.pl.umap(adata, color='leiden', legend_loc='on data')
    </step_6_embed_cluster>

    <step_7_save>
    import os

    # <<<Ask user for output_path if not already known>>>
    output_path = os.path.join('output_dir', 'preprocessed.h5ad')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    adata.write_h5ad(output_path)
    print(f"Saved to {{output_path}}")
    print(f"Final shape: {{adata.shape}}")
    print(f"Leiden clusters: {{adata.obs['leiden'].value_counts().to_dict()}}")
    </step_7_save>

    <notes>
    - Ask user for: data_path, output_path, batch_key, filtering thresholds, resolution
    - If HVG flavor 'seurat_v3' fails (requires raw counts in layer), try flavor='seurat'
    - If BBKNN is unavailable, sc.pp.neighbors is a valid fallback
    - Adjust min_dist, resolution, n_top_genes per dataset characteristics
    - For marker ranking and cell type annotation, use cell_type_annotation_guide as next step
    </notes>

    </stereo_seq_preprocessing_workflow>

    <<<Execute with python_repl_tool step by step>>>
    <<<Ask user rather than assuming>>>
    """
    return dedent(code)


@tool
def visualize_cell_cell_interaction_tool(data_path: str, celltype_col: str, spatial_key: str, output_dir: str, slice_col: str = None, sample_col: str = None) -> str:
    """
    Generates Python code to analyze and visualize cell-cell interaction patterns using neighborhood enrichment.

    EXECUTION REQUIREMENT:
    - This tool returns a Python code string. The agent MUST immediately execute it via `python_repl_tool`
      before proceeding (do not just display the code).

    This tool is DATASET-AGNOSTIC and requires metadata discovered by explore_metadata_tool.

    Prerequisites:
    - User must have run explore_metadata_tool first
    - Agent must have confirmed column mappings with user
    - data_path and output_dir must be remembered from explore_metadata_tool

    Analysis workflow:
    1. Loads the dataset from the provided data_path
    2. For each slice, computes spatial neighbors and neighborhood enrichment
    3. Creates heatmap visualizations showing which cell types attract or avoid each other
    4. Optionally aggregates results by sample/timepoint if sample_col is provided
    5. Saves all plots to the user-specified output_dir

    Args:
        data_path: Path to h5ad file (from explore_metadata_tool, confirmed by user)
        celltype_col: Name of the obs column containing cell type labels (MUST be confirmed by user)
        spatial_key: Name of the obsm key containing spatial coordinates (MUST be confirmed by user)
        output_dir: Directory to save plots (from explore_metadata_tool)
        slice_col: Optional - Name of the obs column for slice/section identifiers (MUST be confirmed by user)
        sample_col: Optional - Name of the obs column for sample/timepoint grouping for aggregation (MUST be confirmed by user)

    The visualizations show:
    - Red colors: Cell types that are more likely to be neighbors (attraction)
    - Blue colors: Cell types that tend to avoid each other (avoidance)
    - Color intensity: Strength of the enrichment/depletion signal (z-score)

    Returns:
        Python code string that can be executed via python_repl_tool

    Agent instructions:
    - DO NOT use this tool without first running explore_metadata_tool
    - DO NOT guess column names - use only user-confirmed column names
    - MUST pass output_dir from explore_metadata_tool
    - If slice_col is None, analyzes entire dataset as one entity
    - The agent MAY modify the returned code to fix bugs or adjust aesthetics
    """
    code = f"""
import anndata as ad
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data and configure output
data_path = "{data_path}"
output_dir = "{output_dir}"
adata = ad.read_h5ad(data_path)
os.makedirs(output_dir, exist_ok=True)

# User-confirmed metadata columns
celltype_col = "{celltype_col}"
spatial_key = "{spatial_key}"
slice_col = "{slice_col}" if "{slice_col}" != "None" else None
sample_col = "{sample_col}" if "{sample_col}" != "None" else None

print(f"Creating cell-cell interaction analysis...")
print(f"Output directory: {{output_dir}}")
print(f"Cell type column: {{celltype_col}}")
print(f"Spatial key: {{spatial_key}}")
print(f"Slice column: {{slice_col}}")
print(f"Sample column: {{sample_col}}")

# Verify spatial coordinates exist
if spatial_key not in adata.obsm.keys():
    raise ValueError(f"Spatial key '{{spatial_key}}' not found in adata.obsm. Available keys: {{list(adata.obsm.keys())}}")

# Perform neighborhood enrichment analysis
result_dict = {{}}

if slice_col and slice_col in adata.obs.columns:
    # Analyze each slice separately
    slice_ids = adata.obs[slice_col].unique()
    print(f"Analyzing {{len(slice_ids)}} slices...")

    for slice_id in slice_ids:
        data_i = adata[adata.obs[slice_col] == slice_id].copy()
        sq.gr.spatial_neighbors(data_i, coord_type="generic", spatial_key=spatial_key, delaunay=True)
        sq.gr.nhood_enrichment(data_i, cluster_key=celltype_col)

        # Get zscore and handle NaN values
        enrichment_key = f'{{celltype_col}}_nhood_enrichment'
        zscore_data = np.nan_to_num(data_i.uns[enrichment_key]['zscore'])

        result_dict[slice_id] = pd.DataFrame(
            zscore_data,
            columns=data_i.obs[celltype_col].cat.categories,
            index=data_i.obs[celltype_col].cat.categories
        )
else:
    # Analyze entire dataset
    print("No slice column - analyzing entire dataset")
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key=spatial_key, delaunay=True)
    sq.gr.nhood_enrichment(adata, cluster_key=celltype_col)

    enrichment_key = f'{{celltype_col}}_nhood_enrichment'
    zscore_data = np.nan_to_num(adata.uns[enrichment_key]['zscore'])

    result_dict['all_cells'] = pd.DataFrame(
        zscore_data,
        columns=adata.obs[celltype_col].cat.categories,
        index=adata.obs[celltype_col].cat.categories
    )

# Optionally aggregate by sample
if sample_col and sample_col in adata.obs.columns and slice_col:
    print(f"Aggregating results by {{sample_col}}...")

    sample_aggregated = {{}}
    for sample_id in sorted(adata.obs[sample_col].unique()):
        # Find all slices belonging to this sample
        slices_in_sample = adata[adata.obs[sample_col] == sample_id].obs[slice_col].unique()
        matching_results = [result_dict[s] for s in slices_in_sample if s in result_dict]

        if matching_results:
            # Average the enrichment scores across slices
            sample_aggregated[sample_id] = sum(matching_results) / len(matching_results)

    if sample_aggregated:
        result_dict = sample_aggregated
        print(f"Aggregated into {{len(result_dict)}} samples")

# Create and save heatmaps
for name, enrichment_df in result_dict.items():
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(enrichment_df, vmax=30, vmin=-30, cmap='RdBu_r', annot=True, fmt=".1f", cbar_kws={{'label': 'Z-score'}})
    ax.set_title(f'Cell-Cell Interaction Enrichment - {{name}}')
    ax.set_xlabel('Target Cell Type')
    ax.set_ylabel('Source Cell Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'interaction_{{name}}.png'), dpi=300, bbox_inches='tight')
    plt.show()

print(f"Cell-cell interaction analysis complete! Generated {{len(result_dict)}} heatmaps in {{output_dir}}")

# Execute this code with python_repl_tool. You MAY fix bugs, adjust parameters (column names,
# thresholds, plot aesthetics) to match the dataset. If unsure about a parameter, ask the user.
    """
    return dedent(code)


@tool
def ligand_receptor_profiling_squidpy(
    data_path: str,
    cluster_key: str,
    output_dir: str,
    slice_col: str = "",
    slice_value: str = "",
    source_groups: str = "",
    target_groups: str = "",
    subset_obs_filters: str = "",
    include_groups: str = "",
    exclude_groups: str = "",
    n_jobs: int = 1,
    n_perms: int = 500,
    alpha: float = 0.005,
    use_raw: bool = False,
) -> str:
    """
    Generates concise Python code instructions for Squidpy ligand-receptor profiling (`sq.gr.ligrec`).

    EXECUTION REQUIREMENT:
    - This tool returns a Python code string. The agent MUST immediately execute it via `python_repl_tool`
      before proceeding (do not just display the code).

    CRITICAL workflow:
    - This script runs ONE ligrec call per execution (safer and more stable than repeated ligrec calls).
    - Use subsetting and group filters to target specific populations/slices.
    - If source/target groups are missing, it prints candidates and stops for user confirmation.

    Args:
        data_path: Path to h5ad file
        cluster_key: obs column containing population/cell-type labels
        output_dir: directory to save ligrec outputs
        slice_col: optional obs column for slice selection
        slice_value: optional single slice value; if slice_col is set and slice_value is empty, script stops and asks confirmation
        source_groups: comma-separated source groups (required)
        target_groups: comma-separated target groups (required)
        subset_obs_filters: optional obs filter string, format:
            "col1=valA|valB;col2=valC" (AND across columns, OR within column values)
        include_groups: optional comma-separated allowed groups after subsetting
        exclude_groups: optional comma-separated groups to remove after subsetting
        n_jobs: worker count for ligrec (default 1 to reduce native parallel crash risk)
        n_perms: permutations for ligrec significance testing
        alpha: significance threshold for plotting/summaries
        use_raw: whether to use adata.raw

    Returns:
        Python code string for `python_repl_tool` execution.
    """
    code = f"""
import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import matplotlib.pyplot as plt

data_path = {repr(data_path)}
cluster_key = {repr(cluster_key)}
output_dir = {repr(output_dir)}
slice_col = {repr(slice_col)}.strip()
slice_value = {repr(slice_value)}.strip()
source_groups_csv = {repr(source_groups)}.strip()
target_groups_csv = {repr(target_groups)}.strip()
subset_obs_filters = {repr(subset_obs_filters)}.strip()
include_groups_csv = {repr(include_groups)}.strip()
exclude_groups_csv = {repr(exclude_groups)}.strip()
n_jobs = max(1, int({int(n_jobs)}))
n_perms = int({int(n_perms)})
alpha = float({float(alpha)})
use_raw = bool({bool(use_raw)})

os.makedirs(output_dir, exist_ok=True)
adata = sc.read_h5ad(data_path)
print("Starting ligand-receptor profiling (Squidpy ligrec)")

if cluster_key not in adata.obs.columns:
    raise ValueError(f"cluster_key '{{cluster_key}}' not found in adata.obs.")

def _tok(text):
    return [x.strip() for x in str(text).split(",") if x.strip()] if str(text).strip() else []

def _apply_filters(adata_obj, filter_text):
    if not str(filter_text).strip():
        return adata_obj
    mask = np.ones(adata_obj.n_obs, dtype=bool)
    for part in [p.strip() for p in str(filter_text).split(";") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid filter token '{{part}}'. Expected 'column=value1|value2'.")
        col, vals = part.split("=", 1)
        values = [v.strip() for v in vals.split("|") if v.strip()]
        if col.strip() not in adata_obj.obs.columns:
            raise ValueError(f"Filter column '{{col.strip()}}' not found in adata.obs.")
        mask &= adata_obj.obs[col.strip()].astype(str).isin(values).to_numpy()
    return adata_obj[mask].copy()

adata = _apply_filters(adata, subset_obs_filters)
if adata.n_obs == 0:
    raise ValueError("No cells remain after subset filter.")

if slice_col:
    if slice_col not in adata.obs.columns:
        raise ValueError(f"slice_col '{{slice_col}}' not found in adata.obs.")
    if not slice_value:
        available = sorted(adata.obs[slice_col].astype(str).unique().tolist())
        print("Available slices:", available)
        raise SystemExit("slice_col provided but slice_value missing.")
    adata = adata[adata.obs[slice_col].astype(str) == str(slice_value)].copy()

if not isinstance(adata.obs[cluster_key].dtype, pd.CategoricalDtype):
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

all_groups = set([str(x) for x in adata.obs[cluster_key].cat.categories.tolist()])
include_groups = _tok(include_groups_csv)
exclude_groups = _tok(exclude_groups_csv)
if include_groups:
    all_groups &= set(include_groups)
all_groups -= set(exclude_groups)
if not all_groups:
    raise ValueError("No groups left after include/exclude filters.")
adata = adata[adata.obs[cluster_key].astype(str).isin(sorted(all_groups))].copy()

source_groups_list = _tok(source_groups_csv)
target_groups_list = _tok(target_groups_csv)
if not source_groups_list or not target_groups_list:
    print("Please confirm source_groups and target_groups explicitly.")
    print("Detected groups:", sorted(all_groups))
    raise SystemExit("Missing confirmed source/target groups.")

missing_sources = sorted(set(source_groups_list) - all_groups)
missing_targets = sorted(set(target_groups_list) - all_groups)
if missing_sources or missing_targets:
    raise ValueError(
        f"Invalid subgroup names. Missing sources={{missing_sources}}, missing targets={{missing_targets}}."
    )

res = sq.gr.ligrec(
    adata,
    n_perms=n_perms,
    n_jobs=n_jobs,
    cluster_key=cluster_key,
    copy=True,
    use_raw=use_raw,
    transmitter_params={{"categories": "ligand"}},
    receiver_params={{"categories": "receptor"}},
)

run_name = "all_cells" if not slice_value else f"slice_{{slice_value}}"
run_dir = os.path.join(output_dir, f"ligrec_{{run_name}}")
os.makedirs(run_dir, exist_ok=True)
for k in ["means", "pvalues", "metadata"]:
    if k in res:
        obj = res[k]
        out_csv = os.path.join(run_dir, f"{{k}}.csv")
        obj.to_csv(out_csv) if hasattr(obj, "to_csv") else pd.DataFrame(obj).to_csv(out_csv)

sig_count = None
try:
    pv = res.get("pvalues")
    if pv is not None:
        pv_num = pv.select_dtypes(include=[np.number]) if hasattr(pv, "select_dtypes") else pd.DataFrame(pv)
        sig_count = int((pv_num < alpha).sum().sum())
except Exception:
    pass

# Robust plotting: some source groups can fail when no plottable interactions exist.
saved_plots = []
for src in source_groups_list:
    fig_path = os.path.join(run_dir, f"ligrec_{{src}}_alpha_{{str(alpha).replace('.', 'p')}}.png")
    try:
        sq.pl.ligrec(res, source_groups=src, alpha=alpha)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.show()
        saved_plots.append(fig_path)
    except Exception as e:
        print(f"Skip plot for source={{src}} due to: {{e}}")
        plt.close("all")

summary = {{
    "run_name": run_name,
    "n_obs": int(adata.n_obs),
    "cluster_key": cluster_key,
    "source_groups": source_groups_list,
    "target_groups": target_groups_list,
    "slice_col": slice_col if slice_col else None,
    "slice_value": slice_value if slice_value else None,
    "subset_obs_filters": subset_obs_filters if subset_obs_filters else None,
    "n_jobs": n_jobs,
    "n_perms": n_perms,
    "alpha": alpha,
    "use_raw": use_raw,
    "significant_pairs_alpha": sig_count,
    "plots_saved": saved_plots,
    "result_dir": run_dir,
}}
summary_path = os.path.join(output_dir, "ligrec_run_summaries.json")
with open(summary_path, "w") as f:
    json.dump([summary], f, indent=2)

print("Ligand-receptor profiling complete. significant_pairs(alpha)=" + str(alpha) + "=" + str(sig_count))
print("Saved:", summary_path)

# Please use python_repl_tool to execute code. 
# <<<REPEAT: Adjust the code based on the reasoning, but keep the core logic and input/output consistent>>>
    """
    return dedent(code)


@tool
def ligand_receptor_compute_squidpy(
    data_path: str,
    cluster_key: str,
    output_dir: str,
    slice_col: str = "",
    slice_value: str = "",
    source_groups: str = "",
    target_groups: str = "",
    subset_obs_filters: str = "",
    include_groups: str = "",
    exclude_groups: str = "",
    n_jobs: int = 1,
    n_perms: int = 500,
    use_raw: bool = False,
) -> str:
    """
    Generate code for ligand-receptor computation only (minimal core workflow).

    Core compute line:
    - `res = sq.gr.ligrec(...)`
    """
    code = f"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

data_path = {repr(data_path)}
cluster_key = {repr(cluster_key)}
output_dir = {repr(output_dir)}
slice_col = {repr(slice_col)}.strip()
slice_value = {repr(slice_value)}.strip()
source_groups_csv = {repr(source_groups)}.strip()
target_groups_csv = {repr(target_groups)}.strip()
subset_obs_filters = {repr(subset_obs_filters)}.strip()
include_groups_csv = {repr(include_groups)}.strip()
exclude_groups_csv = {repr(exclude_groups)}.strip()
n_jobs = max(1, int({int(n_jobs)}))
n_perms = int({int(n_perms)})
use_raw = bool({bool(use_raw)})

os.makedirs(output_dir, exist_ok=True)
adata = sc.read_h5ad(data_path)

if cluster_key not in adata.obs.columns:
    raise ValueError(f"cluster_key '{{cluster_key}}' not found in adata.obs.")

# Optional lightweight subset: single token format "col=value1|value2"
if str(subset_obs_filters).strip():
    if "=" not in subset_obs_filters:
        raise ValueError("subset_obs_filters must be 'col=value1|value2'")
    col, vals = subset_obs_filters.split("=", 1)
    col = col.strip()
    values = [v.strip() for v in vals.split("|") if v.strip()]
    if col not in adata.obs.columns:
        raise ValueError(f"Filter column '{{col}}' not found in adata.obs.")
    adata = adata[adata.obs[col].astype(str).isin(values)].copy()

if slice_col:
    if slice_col not in adata.obs.columns:
        raise ValueError(f"slice_col '{{slice_col}}' not found in adata.obs.")
    if not slice_value:
        available = sorted(adata.obs[slice_col].astype(str).unique().tolist())
        print("Available slices:", available)
        raise SystemExit("slice_col provided but slice_value missing.")
    adata = adata[adata.obs[slice_col].astype(str) == str(slice_value)].copy()

if adata.n_obs == 0:
    raise ValueError("No cells remain after filtering/slicing.")

res = sq.gr.ligrec(
    adata,
    n_perms=n_perms,
    n_jobs=n_jobs,
    cluster_key=cluster_key,
    copy=True,
    use_raw=use_raw,
    transmitter_params={{"categories": "ligand"}},
    receiver_params={{"categories": "receptor"}},
)

run_name = "all_cells" if not slice_value else f"slice_{{slice_value}}"
run_dir = os.path.join(output_dir, f"ligrec_{{run_name}}")
os.makedirs(run_dir, exist_ok=True)

pickle_path = os.path.join(run_dir, "ligrec_result.pkl")
with open(pickle_path, "wb") as f:
    pickle.dump(res, f)

means_var_path = None
for k in ["means", "pvalues", "metadata"]:
    if k in res:
        obj = res[k]
        out_csv = os.path.join(run_dir, f"{{k}}.csv")
        obj.to_csv(out_csv) if hasattr(obj, "to_csv") else pd.DataFrame(obj).to_csv(out_csv)
        if k == "means":
            try:
                means_df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
                means_num = means_df.select_dtypes(include=[np.number])
                if not means_num.empty:
                    mv = pd.DataFrame({{
                        "row_mean": means_num.mean(axis=1),
                        "row_var": means_num.var(axis=1),
                    }})
                    means_var_path = os.path.join(run_dir, "means_var.csv")
                    mv.to_csv(means_var_path)
            except Exception:
                means_var_path = None

summary = {{
    "run_name": run_name,
    "n_obs": int(adata.n_obs),
    "cluster_key": cluster_key,
    "source_groups": source_groups_csv,
    "target_groups": target_groups_csv,
    "slice_col": slice_col if slice_col else None,
    "slice_value": slice_value if slice_value else None,
    "subset_obs_filters": subset_obs_filters if subset_obs_filters else None,
    "include_groups": include_groups_csv if include_groups_csv else None,
    "exclude_groups": exclude_groups_csv if exclude_groups_csv else None,
    "n_jobs": n_jobs,
    "n_perms": n_perms,
    "use_raw": use_raw,
    "pickle_path": pickle_path,
    "means_var_path": means_var_path,
    "result_dir": run_dir,
}}
summary_path = os.path.join(run_dir, "ligrec_compute_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("Ligand-receptor compute complete.")
print("Saved:", summary_path)
print("Pickle:", pickle_path)
"""
    return dedent(code)


@tool
def ligand_receptor_visualize_squidpy(
    ligrec_pickle_path: str,
    output_dir: str,
    source_groups: str,
    alpha: float = 0.005,
) -> str:
    """
    Generate code for ligand-receptor visualization only (minimal core workflow).

    Core visualization line:
    - `sq.pl.ligrec(res, source_groups=..., alpha=...)`
    """
    code = f"""
import os
import pickle
import matplotlib.pyplot as plt
import squidpy as sq

ligrec_pickle_path = {repr(ligrec_pickle_path)}
output_dir = {repr(output_dir)}
source_groups_csv = {repr(source_groups)}.strip()
alpha = float({float(alpha)})

os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(ligrec_pickle_path):
    raise FileNotFoundError(f"ligrec pickle not found: {{ligrec_pickle_path}}")

with open(ligrec_pickle_path, "rb") as f:
    res = pickle.load(f)

source_groups_list = [x.strip() for x in source_groups_csv.split(",") if x.strip()]
if not source_groups_list:
    raise ValueError("source_groups is required for visualization.")

src = source_groups_list[0]
fig_path = os.path.join(output_dir, f"ligrec_{{src}}_alpha_{{str(alpha).replace('.', 'p')}}.png")
sq.pl.ligrec(res, source_groups=src, alpha=alpha)
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print("Ligand-receptor visualization complete. Plot saved:", fig_path)
"""
    return dedent(code)


@tool
def visualize_spatial_cell_type_map(data_path: str, celltype_col: str, spatial_key: str, output_dir: str, slice_col: str = None) -> str:
    """
    Generates Python code to visualize spatial distribution of cell types across tissue slices.

    EXECUTION REQUIREMENT:
    - This tool returns a Python code string. The agent MUST immediately execute it via `python_repl_tool`
      before proceeding (do not just display the code).

    This tool is DATASET-AGNOSTIC and requires metadata discovered by explore_metadata_tool.

    Prerequisites:
    - User must have run explore_metadata_tool first
    - Agent must have confirmed column mappings with user
    - data_path and output_dir must be remembered from explore_metadata_tool

    Analysis workflow:
    1. Loads the dataset from the provided data_path
    2. Creates spatial scatter plots where each point represents a cell
    3. Colors points based on cell type identity
    4. If slice_col is provided, creates separate plots for each slice
    5. Saves all plots to the user-specified output_dir

    Args:
        data_path: Path to h5ad file (from explore_metadata_tool, confirmed by user)
        celltype_col: Name of the obs column containing cell type labels (MUST be confirmed by user)
        spatial_key: Name of the obsm key containing spatial coordinates (MUST be confirmed by user)
        output_dir: Directory to save plots (from explore_metadata_tool)
        slice_col: Optional - Name of the obs column for slice/section identifiers (MUST be confirmed by user)

    Returns:
        Python code string that can be executed via python_repl_tool

    Agent instructions:
    - DO NOT use this tool without first running explore_metadata_tool
    - DO NOT guess column names - use only user-confirmed column names
    - MUST pass output_dir from explore_metadata_tool
    - If slice_col is None, creates a single global spatial plot
    - The agent MAY modify the returned code to fix bugs or adjust aesthetics
    """
    code = f"""
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# Load data and configure output
data_path = "{data_path}"
output_dir = "{output_dir}"
adata = ad.read_h5ad(data_path)
os.makedirs(output_dir, exist_ok=True)

# User-confirmed metadata columns
celltype_col = "{celltype_col}"
spatial_key = "{spatial_key}"
slice_col = "{slice_col}" if "{slice_col}" != "None" else None

print(f"Creating spatial cell type maps...")
print(f"Output directory: {{output_dir}}")
print(f"Cell type column: {{celltype_col}}")
print(f"Spatial key: {{spatial_key}}")
print(f"Slice column: {{slice_col}}")

# Load global color mapping for consistency
color_mapping = {{}}
metadata_path = os.path.join(output_dir, 'metadata_exploration.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        if 'color_mapping' in metadata:
            color_mapping = {{k: tuple(v) for k, v in metadata['color_mapping'].items()}}
            print(f"✓ Loaded global color mapping for {{len(color_mapping)}} cell types")

# Verify spatial coordinates exist
if spatial_key not in adata.obsm.keys():
    raise ValueError(f"Spatial key '{{spatial_key}}' not found in adata.obsm. Available keys: {{list(adata.obsm.keys())}}")

if slice_col and slice_col in adata.obs.columns:
    # Plot spatial distribution for each slice
    slice_names = sorted(adata.obs[slice_col].unique().tolist())
    print(f"Creating spatial maps for {{len(slice_names)}} slices...")

    for slice_id in slice_names:
        adata_slice = adata[adata.obs[slice_col] == slice_id].copy()

        # Calculate dynamic figure size based on spatial extent
        x_coords = adata_slice.obsm[spatial_key][:, 0]
        y_coords = adata_slice.obsm[spatial_key][:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        aspect_ratio = x_range / y_range if y_range > 0 else 1
        fig_height = 8
        fig_width = fig_height * aspect_ratio

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Get unique cell types in this slice
        cell_types_in_slice = adata_slice.obs[celltype_col].unique()
        n_celltypes_slice = len(cell_types_in_slice)

        # Plot each cell type with consistent colors
        for cell_type in cell_types_in_slice:
            cells = adata_slice[adata_slice.obs[celltype_col] == cell_type]
            color = color_mapping.get(cell_type, None) if color_mapping else None
            ax.scatter(
                cells.obsm[spatial_key][:, 0],
                cells.obsm[spatial_key][:, 1],
                label=cell_type,
                s=20,
                alpha=0.8,
                color=color
            )

        ax.set_title(f'Spatial Cell Type Distribution - {{slice_id}}')
        ax.set_xlabel('Spatial X')
        ax.set_ylabel('Spatial Y')

        # Dynamic legend font size and columns to fit within axes
        if n_celltypes_slice <= 5:
            legend_fontsize = 'medium'
            legend_ncol = 1
        elif n_celltypes_slice <= 10:
            legend_fontsize = 'small'
            legend_ncol = 2
        elif n_celltypes_slice <= 20:
            legend_fontsize = 'x-small'
            legend_ncol = 3
        elif n_celltypes_slice <= 30:
            legend_fontsize = 'xx-small'
            legend_ncol = 4
        else:
            legend_fontsize = 'xx-small'
            legend_ncol = 5

        # Place legend inside axes upper-left with constrained size
        ax.legend(loc='upper left', fontsize=legend_fontsize, framealpha=0.9,
                 edgecolor='black', ncol=legend_ncol,
                 bbox_to_anchor=(0.0, 1.0), borderaxespad=0.5,
                 columnspacing=1.0, handletextpad=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spatial_map_{{slice_id}}.png'), dpi=300, bbox_inches='tight')
        plt.show()
else:
    # Global spatial plot (no slice grouping)
    print("No slice column provided - showing global spatial map")

    # Calculate dynamic figure size based on spatial extent
    x_coords = adata.obsm[spatial_key][:, 0]
    y_coords = adata.obsm[spatial_key][:, 1]
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    aspect_ratio = x_range / y_range if y_range > 0 else 1
    fig_height = 10
    fig_width = fig_height * aspect_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Get unique cell types
    cell_types = adata.obs[celltype_col].unique()
    n_celltypes_global = len(cell_types)

    # Plot each cell type with consistent colors
    for cell_type in cell_types:
        cells = adata[adata.obs[celltype_col] == cell_type]
        color = color_mapping.get(cell_type, None) if color_mapping else None
        ax.scatter(
            cells.obsm[spatial_key][:, 0],
            cells.obsm[spatial_key][:, 1],
            label=cell_type,
            s=20,
            alpha=0.8,
            color=color
        )

    ax.set_title('Spatial Cell Type Distribution - All Cells')
    ax.set_xlabel('Spatial X')
    ax.set_ylabel('Spatial Y')

    # Dynamic legend font size and columns to fit within axes
    if n_celltypes_global <= 5:
        legend_fontsize = 'medium'
        legend_ncol = 1
    elif n_celltypes_global <= 10:
        legend_fontsize = 'small'
        legend_ncol = 2
    elif n_celltypes_global <= 20:
        legend_fontsize = 'x-small'
        legend_ncol = 3
    elif n_celltypes_global <= 30:
        legend_fontsize = 'xx-small'
        legend_ncol = 4
    else:
        legend_fontsize = 'xx-small'
        legend_ncol = 5

    # Place legend inside axes upper-left with constrained size
    ax.legend(loc='upper left', fontsize=legend_fontsize, framealpha=0.9,
             edgecolor='black', ncol=legend_ncol,
             bbox_to_anchor=(0.0, 1.0), borderaxespad=0.5,
             columnspacing=1.0, handletextpad=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_map_all_cells.png'), dpi=300, bbox_inches='tight')
    plt.show()

print(f"Spatial cell type visualization complete! Plots saved to {{output_dir}}")

# Execute this code with python_repl_tool. You MAY fix bugs, adjust parameters (column names,
# thresholds, plot aesthetics) to match the dataset. If unsure about a parameter, ask the user.
    """
    return dedent(code)


@tool
def visualize_cell_type_composition(data_path: str, celltype_col: str, output_dir: str, sample_col: str = None) -> str:
    """
    Generates Python code to visualize cell type composition across samples.

    EXECUTION REQUIREMENT:
    - This tool returns a Python code string. The agent MUST immediately execute it via `python_repl_tool`
      before proceeding (do not just display the code).

    This tool is DATASET-AGNOSTIC and requires metadata discovered by explore_metadata_tool.

    Prerequisites:
    - User must have run explore_metadata_tool first
    - Agent must have confirmed column mappings with user
    - data_path and output_dir must be remembered from explore_metadata_tool

    Analysis workflow:
    1. Loads the dataset from the provided data_path
    2. Calculates cell type proportions globally or per sample (if sample_col provided)
    3. Creates two complementary visualizations:
        - Stacked bar plot showing relative proportions
        - Heatmap showing exact percentage values
    4. Saves all plots to the user-specified output_dir

    Args:
        data_path: Path to h5ad file (from explore_metadata_tool, confirmed by user)
        celltype_col: Name of the obs column containing cell type labels (MUST be confirmed by user)
        output_dir: Directory to save plots (from explore_metadata_tool)
        sample_col: Optional - Name of the obs column for sample/timepoint grouping (MUST be confirmed by user)

    Returns:
        Python code string that can be executed via python_repl_tool

    Agent instructions:
    - DO NOT use this tool without first running explore_metadata_tool
    - DO NOT guess column names - use only user-confirmed column names
    - MUST pass output_dir from explore_metadata_tool
    - If sample_col is None, shows global composition only
    - The agent MAY modify the returned code to fix bugs or adjust aesthetics
    """
    code = f"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import anndata as ad
import os
import json

# Load data and configure output
data_path = "{data_path}"
output_dir = "{output_dir}"
adata = ad.read_h5ad(data_path)
os.makedirs(output_dir, exist_ok=True)

# User-confirmed metadata columns
celltype_col = "{celltype_col}"
sample_col = "{sample_col}" if "{sample_col}" != "None" else None

print(f"Creating cell type composition visualizations...")
print(f"Output directory: {{output_dir}}")
print(f"Cell type column: {{celltype_col}}")
print(f"Sample column: {{sample_col}}")

# Load global color mapping for consistency
color_mapping = {{}}
metadata_path = os.path.join(output_dir, 'metadata_exploration.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        if 'color_mapping' in metadata:
            color_mapping = {{k: tuple(v) for k, v in metadata['color_mapping'].items()}}
            print(f"✓ Loaded global color mapping for {{len(color_mapping)}} cell types")

if sample_col and sample_col in adata.obs.columns:
    # Calculate cell type composition for each sample
    composition_df = pd.crosstab(
        adata.obs[sample_col],
        adata.obs[celltype_col],
        normalize='index'  # This gives proportions instead of raw counts
    ) * 100  # Convert to percentages

    # Stacked bar plot with consistent colors
    plt.figure(figsize=(12, 6))
    if color_mapping:
        color_list = [color_mapping.get(ct, None) for ct in composition_df.columns]
        composition_df.plot(kind='bar', stacked=True, color=color_list)
    else:
        composition_df.plot(kind='bar', stacked=True)
    plt.title('Cell Type Composition Across Samples')
    plt.xlabel('Sample')
    plt.ylabel('Percentage of Cells')
    plt.legend(title='Cell Type', loc='upper left')
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'composition_stacked_bar.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print composition table
    print("\\nCell type composition (%):")
    print(composition_df.round(2))

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(composition_df, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Cell Type Composition Heatmap')
    plt.ylabel('Sample')
    plt.xlabel('Cell Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'composition_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
else:
    # Global composition (no sample grouping)
    print("No sample column provided - showing global cell type composition")
    composition_series = adata.obs[celltype_col].value_counts(normalize=True) * 100

    plt.figure(figsize=(10, 6))
    if color_mapping:
        color_list = [color_mapping.get(ct, None) for ct in composition_series.index]
        composition_series.plot(kind='bar', color=color_list)
    else:
        composition_series.plot(kind='bar')
    plt.title('Global Cell Type Composition')
    plt.xlabel('Cell Type')
    plt.ylabel('Percentage of Cells')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'composition_global.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("\\nCell type composition (%):")
    print(composition_series.round(2))

print(f"Cell type composition visualization complete! Plots saved to {{output_dir}}")

# Execute this code with python_repl_tool. You MAY fix bugs, adjust parameters (column names,
# thresholds, plot aesthetics) to match the dataset. If unsure about a parameter, ask the user.
    """
    return dedent(code)


@tool
def visualize_umap(data_path: str, celltype_col: str, output_dir: str, sample_col: str = None) -> str:
    """
    Generates Python code to create UMAP visualizations showing cell type distributions.

    EXECUTION REQUIREMENT:
    - This tool returns a Python code string. The agent MUST immediately execute it via `python_repl_tool`
      before proceeding (do not just display the code).

    This tool is DATASET-AGNOSTIC and requires metadata discovered by explore_metadata_tool.

    Prerequisites:
    - User must have run explore_metadata_tool first
    - Agent must have confirmed column mappings with user
    - data_path and output_dir must be remembered from explore_metadata_tool

    Analysis workflow:
    1. Loads the dataset from the provided data_path
    2. Creates UMAP plot for all cells colored by cell type
    3. If sample_col is provided, creates separate UMAP plots for each sample/timepoint
    4. Saves all plots to the user-specified output_dir

    Args:
        data_path: Path to h5ad file (from explore_metadata_tool, confirmed by user)
        celltype_col: Name of the obs column containing cell type labels (MUST be confirmed by user)
        output_dir: Directory to save plots (from explore_metadata_tool)
        sample_col: Optional - Name of the obs column for sample/timepoint grouping (MUST be confirmed by user)

    Returns:
        Python code string that can be executed via python_repl_tool

    Agent instructions:
    - DO NOT use this tool without first running explore_metadata_tool
    - DO NOT guess column names - use only user-confirmed column names
    - MUST pass output_dir from explore_metadata_tool
    - If sample_col is not relevant for the analysis, pass None
    - The agent MAY modify the returned code to fix bugs or adjust aesthetics
    """
    code = f"""
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import os
import json

# Load data and configure output
data_path = "{data_path}"
output_dir = "{output_dir}"
adata = ad.read_h5ad(data_path)

os.makedirs(output_dir, exist_ok=True)

# User-confirmed metadata columns
celltype_col = "{celltype_col}"
sample_col = "{sample_col}" if "{sample_col}" != "None" else None

print(f"Creating UMAP visualizations...")
print(f"Output directory: {{output_dir}}")
print(f"Cell type column: {{celltype_col}}")
print(f"Sample column: {{sample_col}}")

# Load global color mapping and apply to adata.uns for Scanpy
metadata_path = os.path.join(output_dir, 'metadata_exploration.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        if 'color_mapping' in metadata:
            color_mapping = metadata['color_mapping']
            # Set Scanpy color palette
            unique_celltypes = sorted(adata.obs[celltype_col].unique())
            color_list = [color_mapping.get(ct, [0.5, 0.5, 0.5]) for ct in unique_celltypes]
            adata.uns[f'{{celltype_col}}_colors'] = [f'#{{int(r*255):02x}}{{int(g*255):02x}}{{int(b*255):02x}}' for r, g, b in color_list]
            print(f"✓ Applied global color mapping for {{len(color_list)}} cell types")

# Check if UMAP already exists, if not compute it
if 'X_umap' not in adata.obsm:
    print("Computing UMAP embeddings...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata)
    print("✓ UMAP computation complete")
else:
    print("✓ Using existing UMAP embeddings")

# Determine legend font size and location based on number of cell types
n_celltypes = adata.obs[celltype_col].nunique()
if n_celltypes <= 5:
    legend_fontsize = 'medium'
    legend_loc = 'upper left'
elif n_celltypes <= 10:
    legend_fontsize = 'small'
    legend_loc = 'upper left'
elif n_celltypes <= 20:
    legend_fontsize = 'x-small'
    legend_loc = 'right margin'  # Move outside axes to avoid overflow
else:
    legend_fontsize = 'xx-small'
    legend_loc = 'right margin'  # Move outside axes to avoid overflow

# Plot UMAP for all cells
sc.pl.umap(
    adata,
    color=celltype_col,
    title='UMAP - All Cells',
    legend_loc=legend_loc,
    legend_fontsize=legend_fontsize,
    legend_fontoutline=2,
    show=False
)
plt.savefig(os.path.join(output_dir, 'umap_all_cells.png'), dpi=300, bbox_inches='tight')
plt.show()

# If sample column is provided, create per-sample plots
if sample_col and sample_col in adata.obs.columns:
    unique_samples = sorted(adata.obs[sample_col].unique())
    print(f"Creating UMAP plots for {{len(unique_samples)}} samples...")

    for sample_id in unique_samples:
        adata_sample = adata[adata.obs[sample_col] == sample_id]
        # Adjust font size and location for this sample
        n_celltypes_sample = adata_sample.obs[celltype_col].nunique()
        if n_celltypes_sample <= 5:
            sample_fontsize = 'medium'
            sample_legend_loc = 'upper left'
        elif n_celltypes_sample <= 10:
            sample_fontsize = 'small'
            sample_legend_loc = 'upper left'
        elif n_celltypes_sample <= 20:
            sample_fontsize = 'x-small'
            sample_legend_loc = 'right margin'  # Move outside axes to avoid overflow
        else:
            sample_fontsize = 'xx-small'
            sample_legend_loc = 'right margin'  # Move outside axes to avoid overflow

        sc.pl.umap(
            adata_sample,
            color=celltype_col,
            title=f'UMAP - {{sample_id}}',
            legend_loc=sample_legend_loc,
            legend_fontsize=sample_fontsize,
            legend_fontoutline=2,
            show=False
        )
        plt.savefig(os.path.join(output_dir, f'umap_{{sample_id}}.png'), dpi=300, bbox_inches='tight')
        plt.show()
else:
    print("No sample column provided - showing only global UMAP")

print(f"UMAP visualization complete! Plots saved to {{output_dir}}")

# Execute this code with python_repl_tool. You MAY fix bugs, adjust parameters (column names,
# thresholds, plot aesthetics) to match the dataset. If unsure about a parameter, ask the user.
    """
    return dedent(code)





# ------------------ 20260130 Start implement new tools ------------------

@tool
def cell_type_annotation_guide() -> str:
    """
    Provides flexible guidance for cell type annotation using marker gene analysis.
    
    Workflow: Inspect data → Cluster → Find markers → Interpret → Annotate → Validate
    
    Works with any spatial transcriptomics dataset. Agent should ask user for 
    clarification when needed.
    
    Note: Use python_repl_tool to execute code iteratively.
    """
    code = f"""
    <cell_type_annotation_workflow>

    <step_1_inspect>
    import scanpy as sc
    import anndata as ad

    # Ask user for data path if unclear
    adata = ad.read_h5ad('path_to_data.h5ad')

    print(f"Cells: {{adata.n_obs}}, Genes: {{adata.n_vars}}")
    print(f"\\nObs columns: {{list(adata.obs.columns)}}")
    print(f"Embeddings: {{list(adata.obsm.keys())}}")

    # Examine above and ask user:
    # - Which clustering column to use (if exists)?
    # - Need to subset data (by sample, species, etc.)?
    # - What resolution for new clustering?
    </step_1_inspect>

    <step_2_cluster>
    # Use existing cluster_key or create new one
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata)

    resolution = 1.0  # Ask user to adjust if needed
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden_clusters')
    cluster_key = 'leiden_clusters'

    print(f"Clusters: {{adata.obs[cluster_key].nunique()}}")

    # Visualize (check which embedding exists)
    embedding = 'umap' if 'X_umap' in adata.obsm else 'tsne'
    sc.pl.embedding(adata, basis=embedding, color=cluster_key, legend_loc='on data')
    </step_2_cluster>

    <step_3_markers>
    # Find marker genes
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    for group in result['names'].dtype.names:
        print(f"Cluster {{group}}: {{', '.join(result['names'][group][:5])}}")
    # Use google_scholar_search for unfamiliar markers
    </step_3_markers>

    <step_4_annotate>
    # Based on markers above, create mapping
    # Ask user for help with ambiguous clusters
    cluster_to_celltype = {{
        # '0': 'cell_type_name',
    }}

    adata.obs['annotated_cell_type'] = adata.obs[cluster_key].astype(str).map(cluster_to_celltype)
    adata.obs['annotated_cell_type'].fillna('unknown', inplace=True)

    print(adata.obs['annotated_cell_type'].value_counts())
    sc.pl.embedding(adata, basis=embedding, color=['leiden_clusters', 'annotated_cell_type'])
    </step_4_annotate>

    <step_5_validate>
    # Check obs columns and ask user which to compare with
    print("\\nAvailable columns:", list(adata.obs.columns))

    # If reference annotation exists, compare:
    # import pandas as pd
    # import seaborn as sns
    # contingency = pd.crosstab(adata.obs['annotated_cell_type'], adata.obs[ref_col])
    # row_norm = contingency.div(contingency.sum(axis=1), axis=0)
    # sns.heatmap(row_norm, annot=True, fmt='.2f', cmap='viridis')
    # plt.show()
    </step_5_validate>

    <notes>
    - Ask user for: paths, column names, resolution, cell type labels when unclear
    - Iterate: adjust resolution if too many/few clusters
    - Merge similar clusters if needed
    - Explain biological reasoning for assignments
    </notes>

    </cell_type_annotation_workflow>

    <<<Execute with python_repl_tool step by step>>>
    <<<Ask user rather than assuming>>>
    """
    
    return dedent(code)


@tool
def spatial_domain_identification_staligner(
    data_path: str,
    output_dir: str,
    slice_col: str = "slice_name",
    selected_slices: str = "",
    use_subgraph: bool = True,
) -> str:
    """
    Generate STAligner execution code for spatial domain identification.

    IMPORTANT workflow rules:
    - This tool returns Python code and MUST be executed via `python_repl_tool`.
    - The returned code is marked for external subprocess execution in
      `STAgent_gpusub` to isolate dependencies from the main session.
    - You MUST confirm `slice_col` and (if provided) `selected_slices` with the user
      before executing the code.

    Args:
        data_path: User-provided AnnData path (.h5ad)
        output_dir: User-provided output directory for STAligner results
        slice_col: Column containing slice identifiers (default: "slice_name")
        selected_slices: Optional comma-separated subset of slice values
        use_subgraph: Whether to prefer memory-safe subgraph training (default: True)

    Returns:
        Python code string for `python_repl_tool` execution.
    """
    repo_root = Path(__file__).resolve().parent.parent
    selected_slices_csv = selected_slices or ""
    use_subgraph_flag = "1" if use_subgraph else "0"

    code = f"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
from datetime import datetime

import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import torch

REPO_ROOT = Path(r"{str(repo_root)}")
PACKAGES_AVAILABLE = REPO_ROOT / "packages_available"
if str(PACKAGES_AVAILABLE) not in sys.path:
    # Ensure we import the local patched STAligner package.
    sys.path.insert(0, str(PACKAGES_AVAILABLE))

import STAligner.STAligner as STAligner

DATA_PATH = r"{data_path}"
OUTPUT_DIR = r"{output_dir}"
SLICE_COL = r"{slice_col}"
SELECTED_SLICES_CSV = r"{selected_slices_csv}"
USE_SUBGRAPH = "{use_subgraph_flag}" == "1"

# Safe profile toggles (override by env vars if needed)
KNN_NEIGH = int(os.getenv("STALIGNER_KNN_NEIGH", "30"))
N_EPOCHS = int(os.getenv("STALIGNER_N_EPOCHS", "800"))
RAD_CUTOFF = int(os.getenv("STALIGNER_RAD_CUTOFF", "150"))
MARGIN = float(os.getenv("STALIGNER_MARGIN", "2.5"))

OUT_CONCAT = os.path.join(OUTPUT_DIR, "adata_concat.h5ad")
OUT_TEMP = os.path.join(OUTPUT_DIR, "ad_st_STAligner_temp_before_clustering.h5ad")
OUT_FINAL = os.path.join(OUTPUT_DIR, "ad_st_STAligner.h5ad")
OUT_LOG = os.path.join(OUTPUT_DIR, "staligner_run_debug.log")

def _log(msg: str) -> None:
    ts = datetime.now().isoformat()
    line = f"[{{ts}}][STAlignerPipeline] {{msg}}"
    print(line, flush=True)
    try:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\\n")
    except Exception:
        pass

os.makedirs(OUTPUT_DIR, exist_ok=True)
used_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_log(f"Debug log path: {{OUT_LOG}}")
_log(f"Input data_path: {{DATA_PATH}}")
_log(f"Output dir: {{OUTPUT_DIR}}")
_log(f"Device: {{used_device}}")
_log(
    f"Config: use_subgraph={{USE_SUBGRAPH}}, knn_neigh={{KNN_NEIGH}}, "
    f"n_epochs={{N_EPOCHS}}, rad_cutoff={{RAD_CUTOFF}}, margin={{MARGIN}}"
)

ad_st = ad.read_h5ad(DATA_PATH)
_log(f"Loaded adata shape: {{ad_st.shape}}")
if SLICE_COL not in ad_st.obs.columns:
    raise ValueError(
        f"slice_col '{{SLICE_COL}}' not found in adata.obs. "
        f"Available columns: {{list(ad_st.obs.columns)}}"
    )

available_slices = ad_st.obs[SLICE_COL].astype(str).value_counts().index.to_list()
_log(f"Available slices in '{{SLICE_COL}}': {{available_slices}}")

selected_slices = []
if SELECTED_SLICES_CSV.strip():
    selected_slices = [s.strip() for s in SELECTED_SLICES_CSV.split(",") if s.strip()]
    missing = sorted(set(selected_slices) - set(available_slices))
    if missing:
        raise ValueError(f"selected_slices not found in data: {{missing}}")
    section_ids = selected_slices
else:
    section_ids = available_slices

if len(section_ids) < 2:
    raise ValueError(
        f"Need >=2 slices for STAligner alignment. Got {{len(section_ids)}} from '{{SLICE_COL}}'."
    )

_log(f"Using slices: {{section_ids}}")
Batch_list = []
adj_list = []

for section_id in section_ids:
    _log(f"Preparing slice: {{section_id}}")
    adata = ad_st[ad_st.obs[SLICE_COL].astype(str) == str(section_id)].copy()
    adata.obs_names = [f"{{x}}_{{section_id}}" for x in adata.obs_names]
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=RAD_CUTOFF)
    adj_list.append(adata.uns["adj"])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label=SLICE_COL, keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs[SLICE_COL].astype("category")
_log(f"adata_concat shape: {{adata_concat.shape}}")
adata_concat.write_h5ad(OUT_CONCAT)
_log(f"Saved concat object: {{OUT_CONCAT}}")

adj_concat = adj_list[0]
for batch_id in range(1, len(section_ids)):
    adj_concat = sp.block_diag([adj_concat, adj_list[batch_id]], format="csr")
adata_concat.uns["edgeList"] = adj_concat.nonzero()
_log(f"edgeList size: {{len(adata_concat.uns['edgeList'][0])}} edges")

# Fix first slice as reference and align others to it.
iter_comb = [(i, 0) for i in range(1, len(section_ids))]
_log(f"iter_comb: {{iter_comb}}")

if USE_SUBGRAPH:
    _log("Running memory-safe subgraph training path.")
    adata_concat = STAligner.train_STAligner_subgraph(
        adata_concat,
        Batch_list=Batch_list,
        verbose=True,
        knn_neigh=KNN_NEIGH,
        n_epochs=N_EPOCHS,
        iter_comb=iter_comb,
        margin=MARGIN,
        device=used_device,
    )
else:
    _log("Running full-graph training path (higher memory usage).")
    adata_concat = STAligner.train_STAligner(
        adata_concat,
        verbose=True,
        knn_neigh=KNN_NEIGH,
        n_epochs=N_EPOCHS,
        iter_comb=iter_comb,
        margin=MARGIN,
        device=used_device,
    )

# Compatibility for h5ad serialization
adata_concat.uns["edgeList"] = list(adata_concat.uns["edgeList"])
adata_concat.write_h5ad(OUT_TEMP)
_log(f"Saved intermediate output: {{OUT_TEMP}}")

_log("Running neighbors/louvain/umap/leiden post-processing...")
sc.pp.neighbors(adata_concat, use_rep="STAligner", random_state=2025)
sc.tl.louvain(adata_concat, random_state=2025, key_added="louvain_STAligner", resolution=0.5)
sc.tl.umap(adata_concat, random_state=2025)
sc.tl.leiden(adata_concat, resolution=0.5, key_added="leiden_STAligner")

adata_concat.write_h5ad(OUT_FINAL)
_log(f"Saved final output: {{OUT_FINAL}}")
_log("STAligner finished.")
"""
    return prepend_external_exec_directives(
        dedent(code),
        exec_cwd=str(repo_root),
        exec_timeout=7200,
    )




















@tool
def gene_imputation_tangram(
    sc_raw_path: str,
    st_raw_path: str,
    output_dir: str,
    sc_processed_path: str = "",
    cluster_col: str = "seurat_clusters",
    selected_clusters: str = "",
    mapping_mode: str = "cells",
    cluster_label: str = "",
    num_epochs: int = 200,
    density_prior: str = "uniform",
    device: str = "cpu",
) -> str:
    """
    Generate Tangram execution code for gene imputation.

    IMPORTANT workflow rules:
    - This tool returns Python code and MUST be executed via `python_repl_tool`.
    - The returned code is marked for external subprocess execution in
      `STAgent_gpusub` to isolate dependencies from the main session.
    - You MUST confirm paths/cluster settings with the user before execution.

    Args:
        sc_raw_path: User-provided single-cell raw/reference AnnData path (.h5ad)
        st_raw_path: User-provided spatial transcriptomics AnnData path (.h5ad)
        output_dir: User-provided output directory for Tangram results
        sc_processed_path: Optional processed sc AnnData for marker derivation
        cluster_col: Cluster column for marker derivation/filtering
        selected_clusters: Optional comma-separated subset of clusters
        mapping_mode: Tangram mapping mode: "cells" or "clusters"
        cluster_label: obs column for cluster-mode mapping (defaults to cluster_col)
        num_epochs: Tangram training epochs
        density_prior: Tangram density prior (e.g. "uniform", "rna_count_based")
        device: "cpu" (default), "cuda", or "auto"

    Returns:
        Python code string for `python_repl_tool` execution.
    """
    repo_root = Path(__file__).resolve().parent.parent
    sc_processed_path_safe = sc_processed_path or ""
    selected_clusters_csv = selected_clusters or ""

    code = f"""
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tangram as tg

SC_RAW_PATH = r"{sc_raw_path}"
ST_RAW_PATH = r"{st_raw_path}"
OUTPUT_DIR = r"{output_dir}"
SC_PROCESSED_PATH = r"{sc_processed_path_safe}"
CLUSTER_COL = r"{cluster_col}"
SELECTED_CLUSTERS_CSV = r"{selected_clusters_csv}"
MAPPING_MODE = r"{mapping_mode}".strip().lower()
CLUSTER_LABEL = r"{cluster_label}".strip()
NUM_EPOCHS = int(r"{int(num_epochs)}")
DENSITY_PRIOR = r"{density_prior}".strip()
DEVICE_REQ = r"{device}".strip().lower()

OUT_LOG = os.path.join(OUTPUT_DIR, "tangram_run_debug.log")
OUT_AD_MAP = os.path.join(OUTPUT_DIR, "ad_map_tangram.h5ad")
OUT_AD_ST_IMPUTED = os.path.join(OUTPUT_DIR, "ad_st_tangram_imputed.h5ad")

def _log(msg: str) -> None:
    ts = datetime.now().isoformat()
    line = f"[{{ts}}][TangramPipeline] {{msg}}"
    print(line, flush=True)
    try:
        with open(OUT_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\\n")
    except Exception:
        pass

os.makedirs(OUTPUT_DIR, exist_ok=True)
_log(f"Input sc_raw_path: {{SC_RAW_PATH}}")
_log(f"Input st_raw_path: {{ST_RAW_PATH}}")
_log(f"Input sc_processed_path: {{SC_PROCESSED_PATH or 'None'}}")
_log(
    f"Config: mapping_mode={{MAPPING_MODE}}, cluster_col={{CLUSTER_COL}}, "
    f"selected_clusters={{SELECTED_CLUSTERS_CSV or 'ALL'}}, num_epochs={{NUM_EPOCHS}}, "
    f"density_prior={{DENSITY_PRIOR}}, requested_device={{DEVICE_REQ}}"
)

if MAPPING_MODE not in {{"cells", "clusters"}}:
    raise ValueError("mapping_mode must be one of: 'cells', 'clusters'")

ad_sc = ad.read_h5ad(SC_RAW_PATH)
ad_st = ad.read_h5ad(ST_RAW_PATH)
_log(f"Loaded ad_sc shape={{ad_sc.shape}}, ad_st shape={{ad_st.shape}}")

selected_cluster_list = []
if SELECTED_CLUSTERS_CSV.strip():
    selected_cluster_list = [x.strip() for x in SELECTED_CLUSTERS_CSV.split(",") if x.strip()]
    if CLUSTER_COL not in ad_sc.obs.columns:
        raise ValueError(
            f"selected_clusters provided but cluster_col '{{CLUSTER_COL}}' not found in sc_raw.obs. "
            f"Available columns: {{list(ad_sc.obs.columns)}}"
        )
    available_clusters = set(ad_sc.obs[CLUSTER_COL].astype(str).unique().tolist())
    missing = sorted(set(selected_cluster_list) - available_clusters)
    if missing:
        raise ValueError(f"selected_clusters not found in sc_raw data: {{missing}}")
    ad_sc = ad_sc[ad_sc.obs[CLUSTER_COL].astype(str).isin(selected_cluster_list)].copy()
    _log(f"Filtered ad_sc by selected clusters; new shape={{ad_sc.shape}}")

if MAPPING_MODE == "clusters":
    effective_cluster_label = CLUSTER_LABEL if CLUSTER_LABEL else CLUSTER_COL
    if not effective_cluster_label:
        raise ValueError("cluster_label (or cluster_col) is required for mapping_mode='clusters'")
    if effective_cluster_label not in ad_sc.obs.columns:
        raise ValueError(
            f"cluster label '{{effective_cluster_label}}' not found in sc_raw.obs. "
            f"Available columns: {{list(ad_sc.obs.columns)}}"
        )
else:
    effective_cluster_label = ""

markers = []
if SC_PROCESSED_PATH and Path(SC_PROCESSED_PATH).exists():
    ad_sc_proc = ad.read_h5ad(SC_PROCESSED_PATH)
    _log(f"Loaded ad_sc_proc shape={{ad_sc_proc.shape}} for marker derivation")
    if selected_cluster_list:
        if CLUSTER_COL not in ad_sc_proc.obs.columns:
            raise ValueError(
                f"cluster_col '{{CLUSTER_COL}}' not found in sc_processed.obs for selected_clusters filtering."
            )
        ad_sc_proc = ad_sc_proc[ad_sc_proc.obs[CLUSTER_COL].astype(str).isin(selected_cluster_list)].copy()
        _log(f"Filtered ad_sc_proc by selected clusters; new shape={{ad_sc_proc.shape}}")
    if CLUSTER_COL not in ad_sc_proc.obs.columns:
        raise ValueError(
            f"cluster_col '{{CLUSTER_COL}}' not found in sc_processed.obs. "
            f"Available columns: {{list(ad_sc_proc.obs.columns)}}"
        )
    ad_sc_proc.obs[CLUSTER_COL] = ad_sc_proc.obs[CLUSTER_COL].astype("category")
    sc.tl.rank_genes_groups(ad_sc_proc, groupby=CLUSTER_COL, use_raw=False)
    top_markers = int(os.getenv("STAGENT_TANGRAM_TOP_MARKERS", "100"))
    marker_df = pd.DataFrame(ad_sc_proc.uns["rank_genes_groups"]["names"]).iloc[:top_markers, :]
    markers = list(np.unique(marker_df.melt().value.values))
    _log(f"Derived {{len(markers)}} markers from sc_processed")
else:
    shared = ad_sc.var_names.intersection(ad_st.var_names)
    max_shared = int(os.getenv("STAGENT_TANGRAM_MAX_SHARED_GENES", "4000"))
    markers = list(shared[:max_shared])
    _log(
        f"sc_processed_path missing/unavailable; fallback to shared genes "
        f"({{len(markers)}} genes, capped by STAGENT_TANGRAM_MAX_SHARED_GENES)"
    )

if len(markers) == 0:
    raise ValueError("No marker/shared genes available for Tangram preprocessing.")

sc.pp.normalize_total(ad_sc, target_sum=10000)
sc.pp.normalize_total(ad_st, target_sum=10000)
tg.pp_adatas(ad_sc, ad_st, genes=markers)
_log("Completed normalize_total and tg.pp_adatas preprocessing")

if DEVICE_REQ == "auto":
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
elif DEVICE_REQ == "cuda":
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    if runtime_device == "cpu":
        _log("Requested CUDA but cuda is unavailable; fallback to CPU")
else:
    runtime_device = "cpu"
_log(f"Runtime device={{runtime_device}}")

map_kwargs = {{
    "mode": MAPPING_MODE,
    "density_prior": DENSITY_PRIOR,
    "num_epochs": NUM_EPOCHS,
    "device": runtime_device,
}}
if MAPPING_MODE == "clusters":
    map_kwargs["cluster_label"] = effective_cluster_label

ad_map = tg.map_cells_to_space(ad_sc, ad_st, **map_kwargs)
ad_map.write_h5ad(OUT_AD_MAP)
_log(f"Saved ad_map to {{OUT_AD_MAP}}")

ad_st_tangram = tg.project_genes(ad_map, ad_sc)
if "spatial" in ad_st.obsm:
    ad_st_tangram.obsm["spatial"] = np.asarray(ad_st.obsm["spatial"])
elif {{"x", "y"}}.issubset(set(ad_st.obs.columns)):
    ad_st_tangram.obsm["spatial"] = ad_st.obs[["x", "y"]].to_numpy()
ad_st_tangram.write_h5ad(OUT_AD_ST_IMPUTED)
_log(f"Saved imputed ST object to {{OUT_AD_ST_IMPUTED}}")
_log("Tangram gene imputation finished.")
"""
    return prepend_external_exec_directives(
        dedent(code),
        exec_cwd=str(repo_root),
        exec_timeout=72000, # be very long as the imputation is time-consuming
    )


@tool
def report_tool(context_path: str, query: str = "", debug: bool | None = True) -> str:
    """
    Pure report generator.

    Strong enforcement: this tool ONLY accepts a `report_context.json` created by `results_aggregator_tool`.
    It MUST NOT run deeper research or conflict aggregation itself.
    """
    min_refs = int(20)

    dbg = bool(debug)
    if dbg:
        print(f"[report_tool] start context_path={context_path}")
    if not context_path:
        return (
            "Missing required argument `context_path`.\n"
            "Run results_aggregator_tool first to generate `report_context.json`, then call report_tool(context_path=...)."
        )
    if not os.path.exists(context_path):
        return f"Context file not found: {context_path}"
    try:
        with open(context_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        return f"Failed to read context JSON: {e}"
    if dbg:
        print("[report_tool] loaded context JSON, validating schema")

    # Validate minimal schema
    try:
        ctx = ReportContext.model_validate(raw)
    except Exception as e:
        return (
            "Invalid report context schema. Ensure it was produced by results_aggregator_tool.\n"
            f"Validation error: {e}"
        )
    if dbg:
        print(
            f"[report_tool] validated context: session_id={ctx.meta.session_id} "
            f"messages={len(ctx.analysis_digest.related_messages)} "
            f"conflicts={len(ctx.conflicts)} conflict_events={len(ctx.conflict_events)} "
            f"research_results={len(ctx.research_results)}"
        )
        try:
            full_events = (ctx.conflict_log or {}).get("events") if isinstance(ctx.conflict_log, dict) else None
            if isinstance(full_events, list):
                print(f"[report_tool] conflict_log full events={len(full_events)} (stored in context file)")
        except Exception:
            pass

    # Strong enforcement: require research results to exist (means aggregator executed deeper research)
    allow_no_research = str(os.getenv("ALLOW_REPORT_WITHOUT_RESEARCH", "")).lower() in {"1", "true", "yes"}
    if not ctx.research_results and not allow_no_research:
        return (
            "report_context.json contains no `research_results`.\n"
            "This strongly suggests results_aggregator_tool did not run deeper research. "
            "Re-run results_aggregator_tool and then call report_tool again."
        )
    if dbg and not ctx.research_results and allow_no_research:
        print("[report_tool] ALLOW_REPORT_WITHOUT_RESEARCH=1 (benchmark override)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output_report", exist_ok=True)
    md_filename = f"./output_report/spatial_transcriptomics_report_{ts}.md"
    if dbg:
        print(f"[report_tool] output md will be saved to {md_filename}")

    # Keep prompt bounded.
    # Prefer feeding the full trace via `related_messages` (bounded per-message, images omitted),
    # instead of a giant concatenated digest which can blow up prompt size.
    trace_messages_text = ""
    try:
        trace_messages_text = json.dumps(ctx.analysis_digest.related_messages, ensure_ascii=False, indent=2)
    except Exception:
        trace_messages_text = ""
    if len(trace_messages_text) > 50000:
        trace_messages_text = trace_messages_text[:50000] + "\n...[truncated]"
        if dbg:
            print("[report_tool] trace_messages_text truncated to 50k chars for prompt")

    # Keep a smaller digest_text as a fallback summary (optional).
    digest_text = (ctx.analysis_digest.digest_text or "")
    if len(digest_text) > 20000:
        digest_text = digest_text[:20000] + "\n...[truncated]"
        if dbg:
            print("[report_tool] digest_text truncated to 20k chars for prompt")

    # Conflicts (structured + events) bounded
    conflicts_text = json.dumps([c.model_dump(exclude_none=True) for c in ctx.conflicts[:30]], ensure_ascii=False, indent=2)
    if len(conflicts_text) > 6000:
        conflicts_text = conflicts_text[:6000] + "\n...[truncated]"

    conflict_events_text = json.dumps(ctx.conflict_events[-30:], ensure_ascii=False, indent=2)
    if len(conflict_events_text) > 8000:
        conflict_events_text = conflict_events_text[:8000] + "\n...[truncated]"

    # Research results bounded
    research_text = json.dumps([r.model_dump(exclude_none=True) for r in ctx.research_results[:20]], ensure_ascii=False, indent=2)
    if len(research_text) > 9000:
        research_text = research_text[:9000] + "\n...[truncated]"

    # Load deeper-research report files (bounded). This ensures the report is grounded
    # in the actual deeper research content, not only file paths.
    research_files_blocks: List[str] = []
    total_chars = 0
    for r in ctx.research_results:
        p = r.saved_report_path
        if not p:
            continue
        try:
            if os.path.exists(p):
                path = p
            else:
                # Try relative to repo root/cwd
                path = os.path.join(os.getcwd(), p)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            # Bound each file and total included content
            excerpt = content[:5000] + ("\n...[truncated]" if len(content) > 5000 else "")
            block = f"FILE: {p}\nQUERY: {r.query}\n\n{excerpt}"
            if total_chars + len(block) > 20000:
                break
            research_files_blocks.append(block)
            total_chars += len(block)
        except Exception:
            continue
    research_files_text = "\n\n---\n\n".join(research_files_blocks)
    if dbg:
        print(f"[report_tool] loaded deeper research files={len(research_files_blocks)}")

    # Extract Google Scholar outputs from the full trace (bounded).
    scholar_blocks: List[str] = []
    scholar_chars = 0
    for m in (ctx.analysis_digest.related_messages or []):
        try:
            if (m.get("role") == "tool") and (m.get("tool") == "google_scholar_search"):
                content = str(m.get("content") or "")
                if not content.strip():
                    continue
                excerpt = content[:4000] + ("\n...[truncated]" if len(content) > 4000 else "")
                block = f"TOOL: google_scholar_search\n\n{excerpt}"
                if scholar_chars + len(block) > 12000:
                    break
                scholar_blocks.append(block)
                scholar_chars += len(block)
        except Exception:
            continue
    scholar_text = "\n\n---\n\n".join(scholar_blocks)
    if dbg:
        print(f"[report_tool] extracted google_scholar_search blocks={len(scholar_blocks)}")

    # Prefer Scholar results generated by results_aggregator_tool (more targeted).
    agg_scholar_text = ""
    try:
        if ctx.scholar_results:
            agg_scholar_text = json.dumps(ctx.scholar_results, ensure_ascii=False, indent=2)
            if len(agg_scholar_text) > 12000:
                agg_scholar_text = agg_scholar_text[:12000] + "\n...[truncated]"
    except Exception:
        agg_scholar_text = ""

    # Build a large reference candidate list from Scholar outputs + deeper research reports.
    # We will provide this as a numbered list and enforce a minimum number of citations.
    import re

    def _extract_scholar_items(text: str) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        if not text:
            return items
        # Scholar outputs are formatted as repeated blocks containing Title/Authors/Summary/Link.
        titles = re.findall(r"(?im)^Title:\\s*(.+)\\s*$", text)
        links = re.findall(r"(?im)^Link:\\s*(.+)\\s*$", text)
        # Pair titles and links conservatively by order; if mismatch, keep link empty.
        for i, t in enumerate(titles):
            items.append(
                {
                    "title": t.strip()[:260],
                    "link": (links[i].strip()[:400] if i < len(links) else ""),
                    "source": "google_scholar_search",
                }
            )
        return items

    def _extract_scholar_items_from_results(results: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
        """
        Extract scholar (title/link) items from `ctx.scholar_results` produced by `results_aggregator_tool`.

        Note: we intentionally parse the *raw* `result_excerpt` strings rather than `json.dumps(...)`
        output, because JSON escaping of newlines can break the line-anchored regex patterns.
        """
        items: List[Dict[str, str]] = []
        if not results:
            return items
        for r in results:
            try:
                excerpt = str((r or {}).get("result_excerpt") or "")
            except Exception:
                excerpt = ""
            items.extend(_extract_scholar_items(excerpt))
        return items

    def _extract_md_links(text: str) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        if not text:
            return items
        # Capture markdown links [text](url)
        try:
            # Example: [Paper title](https://example.com)
            for title, url in re.findall(r"\[([^\]]{5,200})\]\((https?://[^\s)]+)\)", text):
                items.append({"title": title.strip()[:260], "link": url.strip()[:400], "source": "deeper_research"})
        except re.error:
            # Never crash report generation due to regex issues.
            return items
        return items

    # Start with explicit scholar outputs (trace) and aggregator scholar results (JSON string).
    ref_candidates: List[Dict[str, str]] = []
    ref_candidates.extend(_extract_scholar_items(scholar_text))
    ref_candidates.extend(_extract_scholar_items_from_results(ctx.scholar_results))
    # Add any markdown links found inside deeper research excerpts.
    ref_candidates.extend(_extract_md_links(research_files_text))

    # De-duplicate by (title, link) while preserving order.
    seen = set()
    deduped: List[Dict[str, str]] = []
    for it in ref_candidates:
        key = ((it.get("title") or "").strip().lower(), (it.get("link") or "").strip().lower())
        if not key[0]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
        if len(deduped) >= 40:
            break

    min_refs = int(os.getenv("REPORT_MIN_REFS", "10"))
    ref_list_lines = []
    for i, it in enumerate(deduped, start=1):
        title = it.get("title", "").strip()
        link = it.get("link", "").strip()
        if link:
            ref_list_lines.append(f"[{i}] {title} — {link}")
        else:
            ref_list_lines.append(f"[{i}] {title}")
    ref_candidates_text = "\n".join(ref_list_lines)
    if dbg:
        print(f"[report_tool] reference candidates extracted={len(deduped)} min_refs_required={min_refs}")

    report_prompt = f"""
# Research Paper Draft (Spatial Transcriptomics)

<context>
You MUST write like a scientific research paper (not a technical log).
Use ONLY the evidence provided below; do not invent citations, numbers, or papers.
Every biological claim that relies on literature must have an inline citation in numeric form (e.g., [1], [2]).
All citations must correspond to entries in the References section (built ONLY from the provided sources).

Tone and focus requirements:
- Keep the narrative biology-forward: tissue architecture, cell-state/function, niches, interactions, mechanisms.
- Mention computational details briefly in Methods only; avoid repeating tool names throughout Results/Discussion.
- Convert technical artifacts into biological uncertainty statements (e.g., "annotation may reflect tissue contamination") and propose validation.

SESSION_ID: {ctx.meta.session_id}

TRACE_MESSAGES_JSON (bounded; full trace, images omitted):
{trace_messages_text}

TRACE_DIGEST_TEXT (smaller bounded fallback):
{digest_text}

CONFLICTS (structured, bounded):
{conflicts_text}

CONFLICT_EVENTS (raw/bounded):
{conflict_events_text}

DEEPER_RESEARCH_RESULTS (structured excerpts, bounded):
{research_text}

DEEPER_RESEARCH_REPORT_FILES (bounded excerpts):
{research_files_text}

GOOGLE_SCHOLAR_SEARCH_OUTPUTS (bounded excerpts):
{scholar_text}

AGGREGATOR_GOOGLE_SCHOLAR_RESULTS (structured excerpts, bounded):
{agg_scholar_text}

REFERENCE_CANDIDATES (use these numeric IDs for citations):
{ref_candidates_text}
</context>

<objective>
Write a comprehensive research-paper-style report (minimum 2000 words) in Markdown.
Ground all claims in the provided analysis trace, tool outputs, and literature excerpts.
If evidence is missing, explicitly label it as a limitation instead of guessing.
</objective>

<required_sections>
## Title
## Abstract (150–250 words)
## Introduction
- Biological background and motivation (cite literature [n] when used)
- Study questions / hypotheses (derived from the trace)
## Methods
- Data overview (what is in the dataset per trace)
- Computational workflow actually performed (brief; focus on what was measured/compared, not tool narration)
- Any assumptions and limitations
## Results
- Describe results like a paper: what you observed biologically (cell types, spatial domains, neighborhood patterns), with dataset-grounded statements
- Refer to generated artifacts by filename when relevant, but interpret biologically
## Discussion
- Interpret findings, connect to literature with citations [n]
- Discuss alternative explanations and confounders
## Conflicts & Validation
- Present the conflict checks as a formal “validation” section
- Summarize conflicts by severity and by conflict_type (internal_knowledge vs literature vs both)
- For each high/medium conflict:
  - Quote the conflict claim
  - State what deeper research + scholar outputs say (supported/contradicted/mixed/insufficient) with citations [n]
  - State how biological conclusions should be adjusted (caveats, revised interpretation, recommended validation experiments/analyses)
## Conclusion
## References
- Numeric list with AT LEAST {min_refs} entries.
- Each entry MUST be selected from REFERENCE_CANDIDATES above.
- Every in-text citation [n] must correspond to an entry in References.
</required_sections>

<extra_user_focus>
{query}
</extra_user_focus>

<important_instructions>
1) OUTPUT ONLY THE REPORT CONTENT, NO OTHER TEXT.
2) Do NOT call tools. Do NOT run deeper research. Do NOT fetch new literature.
3) Citation rules:
   - Use numeric citations: [1], [2], ...
   - Use ONLY IDs from REFERENCE_CANDIDATES.
   - You MUST include at least {min_refs} distinct citations across the paper (not all in one paragraph).
   - If you cannot support a claim with REFERENCE_CANDIDATES, mark it as “uncited / needs literature confirmation”.
4) Writing style rules:
   - Avoid “tool narrative” (“I ran tool X”) except briefly in Methods.
   - Prefer paper-like phrasing and logical flow.
</important_instructions>
"""

    report_model = os.getenv("REPORT_MODEL", "gpt-5.2")
    llm = ChatOpenAI(model=report_model, temperature=0.7, max_tokens=8000)
    try:
        if dbg:
            print("[report_tool] generating report via LLM (no tools)")
        report = llm.invoke([HumanMessage(content=report_prompt, name="report_tool")])
    except Exception as e:
        return f"Error generating report: {e}"

    # Some provider responses can contain empty content (e.g., refusal metadata).
    # Guard against writing an empty file silently.
    content = getattr(report, "content", None)
    if isinstance(content, list):
        content = "\n".join([str(x) for x in content if str(x).strip()])
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)

    if dbg:
        print(f"[report_tool] report content chars={len(content)}")

    # Enforce minimum citation coverage (best-effort): if under-cited, retry once.
    try:
        cited = set(re.findall(r"\\[(\\d{1,3})\\]", content))
        cited_n = len(cited)
    except Exception:
        cited_n = 0
    if content.strip() and cited_n < min_refs and len(deduped) >= min_refs:
        if dbg:
            print("[report_tool] WARNING: under-cited report; retrying once with explicit citation enforcement")
        revision_prompt = (
            "Revise the draft below to increase citation coverage and improve literature integration.\n"
            f"Hard requirements:\n"
            f"- Use ONLY the REFERENCE_CANDIDATES IDs provided.\n"
            f"- Include at least {min_refs} distinct citations across the paper.\n"
            f"- Ensure References has at least {min_refs} entries, all from REFERENCE_CANDIDATES.\n"
            "- Do NOT invent references.\n\n"
            "REFERENCE_CANDIDATES:\n"
            f"{ref_candidates_text}\n\n"
            "DRAFT:\n"
            f"{content}"
        )
        try:
            report_rev = llm.invoke([HumanMessage(content=revision_prompt, name="report_tool_revision")])
            content_rev = getattr(report_rev, "content", "") or ""
            if isinstance(content_rev, list):
                content_rev = "\n".join([str(x) for x in content_rev if str(x).strip()])
            if isinstance(content_rev, str) and content_rev.strip():
                content = content_rev
        except Exception as e:
            if dbg:
                print(f"[report_tool] revision retry failed: {e}")

    # If content is empty, retry once with a smaller prompt (bounded) to avoid silent failures.
    if not content.strip():
        if dbg:
            try:
                print(f"[report_tool] WARNING: empty report content returned; report repr={repr(report)}")
            except Exception:
                print("[report_tool] WARNING: empty report content returned; (repr unavailable)")
            print("[report_tool] retrying once with reduced context")

        reduced_trace = trace_messages_text[:20000] + ("\n...[truncated]" if len(trace_messages_text) > 20000 else "")
        reduced_research_files = research_files_text[:8000] + ("\n...[truncated]" if len(research_files_text) > 8000 else "")
        reduced_scholar = scholar_text[:6000] + ("\n...[truncated]" if len(scholar_text) > 6000 else "")
        reduced_prompt = report_prompt
        reduced_prompt = reduced_prompt.replace(trace_messages_text, reduced_trace)
        reduced_prompt = reduced_prompt.replace(research_files_text, reduced_research_files)
        reduced_prompt = reduced_prompt.replace(scholar_text, reduced_scholar)
        try:
            report2 = llm.invoke([HumanMessage(content=reduced_prompt, name="report_tool_retry")])
        except Exception as e:
            return f"Error generating report (retry failed): {e}"
        content2 = getattr(report2, "content", None)
        if isinstance(content2, list):
            content2 = "\n".join([str(x) for x in content2 if str(x).strip()])
        if content2 is None:
            content2 = ""
        if not isinstance(content2, str):
            content2 = str(content2)
        if dbg:
            print(f"[report_tool] retry report content chars={len(content2)}")
        content = content2
        report = report2

    try:
        with open(md_filename, "w", encoding="utf-8") as f:
            if content.strip():
                f.write(content)
            else:
                # Write diagnostics instead of an empty file.
                f.write(
                    "# ERROR: Empty report generated\n\n"
                    "The model returned an empty response body. This usually indicates refusal or a provider-side issue.\n\n"
                    f"- session_id: {getattr(ctx.meta, 'session_id', '')}\n"
                    f"- context_path: {context_path}\n"
                    f"- model: gpt-5\n"
                    "\n"
                    "## Debug\n"
                    f"- prompt_chars: {len(report_prompt)}\n"
                    f"- digest_chars_used: {len(digest_text)}\n"
                    f"- scholar_blocks: {len(scholar_blocks)}\n"
                    f"- deeper_research_files_loaded: {len(research_files_blocks)}\n"
                    "\n"
                    "## Raw response repr\n"
                    f"{repr(report)}\n"
                )
    except Exception as e:
        return f"Error saving markdown file: {e}"
    if dbg:
        print("[report_tool] report saved successfully")

    return f"Report has been saved as markdown file: {md_filename}"

@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")  # type: ignore[arg-type]
    locals: Optional[Dict] = None  # type: ignore[arg-type]

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.

        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)

        Args:
            query: The query to sanitize

        Returns:
            str: The sanitized query
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    @classmethod
    def worker(
        cls,
        command: str,
        globals: Optional[Dict],
        locals: Optional[Dict],
        queue: multiprocessing.Queue,
    ) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            cleaned_command = cls.sanitize_input(command)
            exec(cleaned_command, globals, locals)
            sys.stdout = old_stdout
            queue.put(mystdout.getvalue())
        except Exception as e:
            sys.stdout = old_stdout
            queue.put(repr(e))

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""

        # Warn against dangers of PythonREPL
        warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, self.locals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return "Execution timed out"
        else:
            self.worker(command, self.globals, self.locals, queue)
        # get the result from the worker function
        return queue.get()

    def run_external(
        self,
        command: str,
        *,
        python_bin: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
    ) -> str:
        """
        Run code in a separate Python interpreter (subprocess).

        This is useful for isolating dependency conflicts (e.g. NumPy 1.x vs 2.x).

        Notes:
        - No shared in-memory state with this process.
        - Matplotlib figure capture in the Streamlit process will NOT work; user code should save plots to disk.
        - `python_bin` may be a full command (e.g. "conda run -n env_STAligner python").
        """
        warn_once()
        cleaned_command = self.sanitize_input(command)

        # Write a small script file so multiline code works robustly.
        base_dir = Path(__file__).resolve().parent / "tmp" / "repl_scripts"
        base_dir.mkdir(parents=True, exist_ok=True)
        script_path = base_dir / f"repl_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py"
        script_path.write_text(cleaned_command + "\n", encoding="utf-8")

        try:
            cmd = shlex.split(python_bin) + [str(script_path)]
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            partial_out = (e.stdout or "") + (("\n" + e.stderr) if e.stderr else "")
            partial_out = partial_out.strip()
            details = [
                f"[external_python] timeout_after={timeout}s",
                f"[external_python] python_bin={python_bin}",
                f"[external_python] cwd={cwd or os.getcwd()}",
                f"[external_python] script={script_path}",
            ]
            if partial_out:
                details.append("[external_python] partial_output:")
                details.append(partial_out[:8000] + ("\n...[truncated]" if len(partial_out) > 8000 else ""))
            return "\n".join(details)
        except Exception as e:
            return f"External execution error: {e}"

        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        out = out.strip()
        if proc.returncode != 0:
            # Keep a compact signal for failures while still returning stderr.
            prefix = f"[external_python] exit_code={proc.returncode}"
            return f"{prefix}\n{out}" if out else prefix
        return out or "Executed code successfully with no output."
