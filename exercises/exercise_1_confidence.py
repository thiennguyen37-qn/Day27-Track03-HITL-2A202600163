"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse
import json as _json

from dotenv import load_dotenv
from json_repair import repair_json
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


def _fix_analysis_json(data: dict) -> PRAnalysis:
    """Normalize common LLM schema mistakes before Pydantic validation."""
    for alias in ("confidence_re", "confidence_rereasoning", "confidence_reason", "reasoning"):
        if alias in data and "confidence_reasoning" not in data:
            data["confidence_reasoning"] = data.pop(alias)
    if "confidence_reasoning" not in data:
        data["confidence_reasoning"] = ""
    data["comments"] = [
        c for c in data.get("comments", [])
        if isinstance(c.get("file"), str) and c.get("file")
    ]
    for comment in data["comments"]:
        if not isinstance(comment.get("line"), int):
            comment["line"] = None
    return PRAnalysis.model_validate(data)


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm()
    system = "You are a strict security-focused code reviewer. Output ONLY valid JSON — no markdown, no extra text."
    prompt = (
        f"Review this pull request for SAFETY TO MERGE.\n"
        f"Title: {state['pr_title']}\n"
        f"Files: {', '.join(state['pr_files'])}\n\n"
        f"Diff:\n{state['pr_diff']}\n\n"
        "confidence = probability this PR is SAFE TO MERGE without any human review.\n"
        "Lower confidence when you see: security issues, auth/crypto code, SQL queries, "
        "missing tests, schema migrations, hard-coded secrets, or anything unclear.\n"
        "confidence < 0.58 → escalate (serious risks, need human answers)\n"
        "confidence 0.58–0.72 → human approval needed\n"
        "confidence > 0.72 → safe to auto-approve (trivial/mechanical only)\n\n"
        "Return a JSON object with EXACTLY these fields (keep values SHORT):\n"
        '{"summary":"2 sentences max","risk_factors":["up to 3 items"],'
        '"comments":[{"file":"filename","line":null,"severity":"nit|suggestion|issue|blocker","body":"short"}],'
        '"confidence":0.0,"confidence_reasoning":"1 sentence","escalation_questions":[]}\n'
        "Rules:\n"
        "- comments: max 3 items\n"
        "- line: integer or null (never a string)\n"
        "- if confidence < 0.58 add specific escalation_questions\n"
        "- field name is confidence_reasoning (required, not optional)"
    )
    with console.status("[dim]LLM thinking...[/dim]"):
        msg = llm.invoke([("system", system), ("human", prompt)])
    content = msg.content.strip()
    data = repair_json(content, return_objects=True)
    analysis = _fix_analysis_json(data)
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}  risks={len(analysis.risk_factors)}")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision={decision}")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
