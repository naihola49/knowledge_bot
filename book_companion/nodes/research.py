"""Node 3 orchestration: build research premise docs via Tavily"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from book_companion.premise_ingestion import build_premises_from_output_2
from book_companion.schema.validation import (
    validate_graph_state,
    validate_output_2,
    validate_output_3,
)
from book_companion.state import GraphState, Output3


def _research_docs_dir(state: dict, day: str) -> Path:
    run_id = str(state.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    return Path("artifacts") / "research" / day / run_id


def _render_research_markdown(day: str, docs: list) -> str:
    lines = [f"# Research Brief ({day})", ""]
    if not docs:
        lines.append("No research documents were retrieved from Tavily.")
        return "\n".join(lines) + "\n"

    for idx, doc in enumerate(docs, start=1):
        lines.extend(
            [
                f"## Doc {idx}: {doc.topic}",
                f"- intent: `{doc.intent_kind}`",
                f"- score: `{doc.source_score:.4f}`",
                f"- query: `{doc.query}`",
                f"- url: {doc.url}",
                f"- title: {doc.title}",
                "",
                "### Snippet",
                doc.snippet or "(empty snippet)",
                "",
                "### Raw Content (Preview)",
                (doc.raw_content[:2000] + "...") if len(doc.raw_content) > 2000 else doc.raw_content,
                "",
            ]
        )
    return "\n".join(lines)


def run_research_node(state: GraphState) -> GraphState:
    """Build premise docs from output_2 and persist a markdown brief artifact."""
    validated_state = validate_graph_state(state, context="research input")
    day = str(validated_state.get("day") or "unknown_day")
    output_2 = validate_output_2(validated_state.get("output_2") or {"topics": []})

    docs = build_premises_from_output_2(day=day, output_2=output_2)

    target_dir = _research_docs_dir(validated_state, day)
    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = target_dir / "research_brief.md"
    md_path.write_text(_render_research_markdown(day, docs), encoding="utf-8")

    output_3: Output3 = {
        "research_md_path": str(md_path),
        "prompt_user_retry": len(docs) == 0,
    }
    validated_output_3 = validate_output_3(output_3)

    return {**validated_state, "output_3": validated_output_3}

