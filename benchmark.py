"""How many tokens (and dollars) does the check-in harness save on a task?

Every mode gets the same question, the same tools and, at some point, the same
clarification from you. What differs is WHEN the clarification arrives:

  baseline   Plain deep agent. It answers in full; if that answer was built on the
             wrong definition, you send the clarification and it redoes the work.
  ask_text   Plain deep agent told "never assume, ask first". It can only ask by
             ending its turn with a question (what Claude Code does today). You then
             reply with the clarification.
  steerable  This harness. It pauses with check_in mid-run, you answer, it finishes.

Usage:
  .venv/bin/python benchmark.py long 3     (task, runs per mode)
  .venv/bin/python benchmark.py toy 3
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from steerable import build_agent, resolve_model

load_dotenv()

MODEL = "anthropic:claude-sonnet-5-5"
# $ per 1M tokens for Claude Sonnet 5.5 (Anthropic pricing table, cached 2026-11-03).
PRICE = {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00}

ASK_FIRST_PROMPT = (
    "You are an analysis agent. Never assume. If a definition is ambiguous or data is "
    "missing, stop and ask the user before running further queries."
)


def load_task(name: str) -> SimpleNamespace:
    if name == "long":
        import long_task as t

        return SimpleNamespace(
            tools=t.TOOLS, question=t.QUESTION, clarification=t.CLARIFICATION,
            intended=t.INTENDED, leader=t.true_leader(),
        )
    import example_tools as t

    return SimpleNamespace(
        tools=t.TOOLS,
        question=(
            "Compare monthly referral vs paid_meta vs organic signups for Jul to Oct 2026, "
            "and the activation rate per channel. Which channel is best?"
        ),
        clarification=(
            "Leave September as a gap, don't interpolate it. Treat organic activation as "
            "unknown and rank only referral vs paid_meta."
        ),
        intended=None, leader="referral",
    )


def usage(messages: list) -> dict:
    """Sum billed tokens over every model call in the conversation."""
    t = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "calls": 0, "tool_calls": 0}
    for m in messages:
        if not isinstance(m, AIMessage) or not m.usage_metadata:
            continue
        u = m.usage_metadata
        d = u.get("input_token_details") or {}
        cr, cw = d.get("cache_read", 0) or 0, d.get("cache_creation", 0) or 0
        t["cache_read"] += cr
        t["cache_write"] += cw
        t["input"] += u["input_tokens"] - cr - cw  # input_tokens includes cached ones
        t["output"] += u["output_tokens"]  # includes thinking tokens
        t["calls"] += 1
        t["tool_calls"] += len(m.tool_calls)
    t["cost"] = sum(t[k] * PRICE[k] for k in PRICE) / 1e6
    return t


def definitions_queried(messages: list) -> Counter:
    return Counter(
        tc["args"].get("definition")
        for m in messages if isinstance(m, AIMessage)
        for tc in m.tool_calls if tc["name"] == "query_referral_activation"
    )


def finish(agent, cfg, task, extra: dict) -> dict:
    msgs = agent.get_state(cfg).values["messages"]
    t = usage(msgs)
    t.update(extra)
    t["defs"] = dict(definitions_queried(msgs))
    t["correct"] = task.leader.lower() in msgs[-1].text.lower()
    return t


def plain_agent(task, prompt: str):
    return create_deep_agent(
        model=resolve_model(MODEL), tools=task.tools,
        system_prompt=prompt, checkpointer=InMemorySaver(),
    )


def run_baseline(task, i: int) -> dict:
    agent = plain_agent(task, "You are an analysis agent.")
    cfg = {"configurable": {"thread_id": f"base-{i}"}}
    agent.invoke({"messages": [HumanMessage(task.question)]}, cfg)
    defs = definitions_queried(agent.get_state(cfg).values["messages"])
    # You only need to correct it if it didn't stick to the definition you meant.
    needs_fix = task.intended is None or set(defs) != {task.intended}
    if needs_fix:
        fix = f"Correction: {task.clarification} Please redo the answer."
        agent.invoke({"messages": [HumanMessage(fix)]}, cfg)
    return finish(agent, cfg, task, {"user_turns": 1 + needs_fix})


def run_ask_text(task, i: int) -> dict:
    agent = plain_agent(task, ASK_FIRST_PROMPT)
    cfg = {"configurable": {"thread_id": f"ask-{i}"}}
    agent.invoke({"messages": [HumanMessage(task.question)]}, cfg)
    agent.invoke({"messages": [HumanMessage(task.clarification)]}, cfg)
    return finish(agent, cfg, task, {"user_turns": 2})


def run_steerable(task, i: int) -> dict:
    # 5-minute cache like the plain agents, so costs compare like for like (no human pauses here)
    agent = build_agent(MODEL, task.tools, checkpointer=InMemorySaver(), cache_ttl="5m")
    cfg = {"configurable": {"thread_id": f"steer-{i}"}}
    payload = {"messages": [HumanMessage(task.question)]}
    check_ins = 0
    for _ in range(4):
        result = agent.invoke(payload, cfg)
        if not result.get("__interrupt__"):
            break
        check_ins += 1
        payload = Command(resume=task.clarification)
    return finish(agent, cfg, task, {"user_turns": 1, "check_ins": check_ins})


def summarize(name: str, runs: list[dict]) -> dict:
    keys = ["calls", "tool_calls", "input", "cache_write", "cache_read", "output", "cost"]
    med = {k: statistics.median(r[k] for r in runs) for k in keys}
    print(f"\n{name} (median of {len(runs)} runs)")
    print(f"  model calls / tool calls  {med['calls']:.0f} / {med['tool_calls']:.0f}")
    print(f"  uncached input   {med['input']:>10,.0f}   cache write {med['cache_write']:>8,.0f}")
    print(f"  cache read       {med['cache_read']:>10,.0f}   output      {med['output']:>8,.0f}")
    print(f"  cost             ${med['cost']:.4f}   (runs: {', '.join(f'${r['cost']:.4f}' for r in runs)})")
    print(f"  correct answer   {sum(r['correct'] for r in runs)}/{len(runs)}")
    for r in runs:
        extra = f", check-ins {r['check_ins']}" if "check_ins" in r else ""
        print(f"    run: defs queried {r['defs'] or '-'}, user turns {r['user_turns']}{extra}")
    return med


if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else "long"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    task = load_task(task_name)
    print(f"task: {task_name} | model: {MODEL} | correct leader: {task.leader}")

    results = {}
    for name, fn in [("baseline", run_baseline), ("ask_text", run_ask_text), ("steerable", run_steerable)]:
        runs = [fn(task, i) for i in range(n)]
        results[name] = summarize(name.upper(), runs)

    b = results["baseline"]["cost"]
    print("\nRESULT (cost per task, median)")
    for name, med in results.items():
        diff = b - med["cost"]
        print(f"  {name:<10} ${med['cost']:.4f}   saves ${diff:+.4f} vs baseline ({diff / b:+.0%})")
