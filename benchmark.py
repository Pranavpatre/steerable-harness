"""How many tokens (and dollars) does the check-in harness save on a task?

Every mode gets the same question, the same tools and, at some point, the same
clarification from you. What differs is WHEN the clarification arrives:

  baseline   Plain deep agent. It answers in full; if that answer was built on the
             wrong definition, you send the clarification and it redoes the work.
  ask_text   Plain deep agent told "never assume, ask first". It can only ask by
             ending its turn with a question (what Claude Code does today). You then
             reply with the clarification.
  steer      This harness in steer mode (Phases 1 to 3): it writes a ledger you confirm
             or correct before any query, and can pause with check_in mid-run.
  steer_routed  Same, plus the Phase 4 router sending routine steps (file reads,
             todo updates) to Claude Haiku 5.5.
  quick      This harness in quick mode: no ledger or check-in. Like baseline, you
             correct it afterwards if it used the wrong definition.

The simulated user gives the clarification at the first ledger or check-in pause,
answers any later check-in with it again, confirms later ledgers, and approves
every pre-tool approval.

Usage:
  .venv/bin/python benchmark.py long 3     (task, runs per mode)
  .venv/bin/python benchmark.py toy 3
  .venv/bin/python benchmark.py midrun 3

`midrun` plants a gap you can only spot once its data comes back (mid_run_task.py).
There, the simulated user has nothing to say up front ("use the data as it is"),
steers as soon as India's Sep or Oct result appears, and, in every mode, sends a
correction after the answer if the winner is still wrong.
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from steerable import ROUTINE_TOOLS, ModelRouterMiddleware, SteerInbox, build_agent, resolve_model, run_cost, steps_done
from steerable import usage as token_usage

load_dotenv()

MODEL = "anthropic:claude-sonnet-5-5"

ASK_FIRST_PROMPT = (
    "You are an analysis agent. Never assume. If a definition is ambiguous or data is "
    "missing, stop and ask the user before running further queries."
)


def load_task(name: str) -> SimpleNamespace:
    if name == "midrun":
        import mid_run_task as t

        return SimpleNamespace(
            tools=t.TOOLS, question=t.QUESTION, clarification=t.DIRECTION, intended=None,
            leader=t.true_leader(), midrun=True, no_pref=t.NO_PREFERENCE, is_gap=t.is_gap_result,
        )
    if name == "long":
        import long_task as t

        return SimpleNamespace(
            tools=t.TOOLS, question=t.QUESTION, clarification=t.CLARIFICATION,
            intended=t.INTENDED, leader=t.true_leader(), midrun=False,
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
        intended=None, leader="referral", midrun=False,
    )


def usage(messages: list) -> dict:
    """Sum billed tokens over every model call in the conversation."""
    t = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "calls": 0, "tool_calls": 0}
    for m in messages:
        if not isinstance(m, AIMessage) or not m.usage_metadata:
            continue
        # The harness's own parser: when Anthropic splits cache writes by lifetime, LangChain puts
        # them under ephemeral_5m/1h_input_tokens and sets cache_creation to 0.
        u = token_usage(m)
        for k in ("input", "cache_write", "cache_read", "output"):
            t[k] += u[k]  # output includes thinking tokens
        t["calls"] += 1
        t["tool_calls"] += len(m.tool_calls)
    t["cost"] = run_cost(messages)  # per model, so Haiku steps are priced as Haiku
    t["haiku_calls"] = sum(1 for m in messages if isinstance(m, AIMessage) and m.usage_metadata
                           and "haiku" in ((m.response_metadata or {}).get("harness_route") or {}).get("model", ""))
    return t


def definitions_queried(messages: list) -> Counter:
    return Counter(
        tc["args"].get("definition")
        for m in messages if isinstance(m, AIMessage)
        for tc in m.tool_calls if tc["name"] == "query_referral_activation"
    )


def winner(text: str) -> str | None:
    m = re.search(r"Winner:\W*([A-Z]{2})\b", text)
    return m.group(1) if m else None


def right(task, text: str) -> bool:
    return winner(text) == task.leader if task.midrun else task.leader.lower() in text.lower()


def last_text(agent, cfg) -> str:
    return agent.get_state(cfg).values["messages"][-1].text


def correct_if_wrong(agent, cfg, task) -> int:
    """midrun only: after the answer, send the correction if the winner is wrong. Returns turns sent."""
    if not task.midrun or right(task, last_text(agent, cfg)):
        return 0
    agent.invoke({"messages": [HumanMessage(f"Correction: {task.clarification} Please redo the answer.")]}, cfg)
    return 1


def finish(agent, cfg, task, extra: dict) -> dict:
    msgs = agent.get_state(cfg).values["messages"]
    t = usage(msgs)
    t.update(extra)
    t["defs"] = dict(definitions_queried(msgs))
    t["correct"] = right(task, msgs[-1].text)
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
    if task.midrun:
        first = right(task, last_text(agent, cfg))
        return finish(agent, cfg, task, {"user_turns": 1 + correct_if_wrong(agent, cfg, task), "first_try": first})
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
    if task.midrun:  # it can only ask up front, when you don't know about the gap yet
        agent.invoke({"messages": [HumanMessage(task.no_pref)]}, cfg)
        first = right(task, last_text(agent, cfg))
        return finish(agent, cfg, task, {"user_turns": 2 + correct_if_wrong(agent, cfg, task), "first_try": first})
    agent.invoke({"messages": [HumanMessage(task.clarification)]}, cfg)
    return finish(agent, cfg, task, {"user_turns": 2})


def harness_run(task, i: int, mode: str, routed: bool = False) -> dict:
    # 5-minute cache like the plain agents, so costs compare like for like (no human pauses here)
    router = ModelRouterMiddleware("anthropic:claude-haiku-5-5", routine_tools=ROUTINE_TOOLS) if routed else None
    inbox = SteerInbox()
    agent = build_agent(MODEL, task.tools, inbox=inbox, checkpointer=InMemorySaver(), cache_ttl="5m",
                        mode=mode, router=router)
    cfg = {"configurable": {"thread_id": f"{mode}-{routed}-{i}"}}
    payload = {"messages": [HumanMessage(task.question)]}
    pauses, clarified, steered = Counter(), False, None
    for _ in range(10):
        pending = None
        for chunk in agent.stream(payload, cfg, stream_mode="updates"):
            if "__interrupt__" in chunk:
                pending = chunk["__interrupt__"][0].value
                continue
            for update in chunk.values():  # you watch tool results come back, as in the live view
                for m in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                    if task.midrun and steered is None and getattr(m, "type", "") == "tool" and task.is_gap(m.text):
                        inbox.put(task.clarification)
                        steered = steps_done(agent.get_state(cfg).values["messages"])
        if pending is None:
            break
        kind = pending.get("type")
        pauses[kind] += 1
        if task.midrun and kind in {"ledger", "check_in"}:  # you only know about the gap once you've seen it
            answer = task.clarification if (kind == "check_in" and steered is not None) else \
                ("ok" if kind == "ledger" else task.no_pref)
        elif kind == "check_in" or (kind == "ledger" and not clarified):
            answer, clarified = task.clarification, True
        else:  # a later ledger, an approval, a budget or step pause: confirm / approve / continue
            answer = "1" if kind in {"budget", "step_pause"} else "ok"
        payload = Command(resume=answer)
    turns = 1
    if task.midrun:
        first = right(task, last_text(agent, cfg))
        turns += correct_if_wrong(agent, cfg, task)
        return finish(agent, cfg, task, {"user_turns": turns, "pauses": dict(pauses), "first_try": first,
                                         "steered_at": steered})
    if mode == "quick":  # no pauses: correct afterwards if it used the wrong definition, like baseline
        defs = definitions_queried(agent.get_state(cfg).values["messages"])
        if task.intended is None or set(defs) != {task.intended}:
            agent.invoke({"messages": [HumanMessage(f"Correction: {task.clarification} Please redo the answer.")]}, cfg)
            turns = 2
    return finish(agent, cfg, task, {"user_turns": turns, "pauses": dict(pauses)})


def run_steer(task, i): return harness_run(task, i, "steer")
def run_steer_routed(task, i): return harness_run(task, i, "steer", routed=True)
def run_quick(task, i): return harness_run(task, i, "quick")


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
        extra = f", pauses {r['pauses'] or '-'}" if "pauses" in r else ""
        extra += f", Haiku calls {r['haiku_calls']}" if r.get("haiku_calls") else ""
        extra += f", right first time {r['first_try']}" if "first_try" in r else ""
        extra += f", steered after step {r['steered_at']}" if r.get("steered_at") is not None else ""
        print(f"    run: defs queried {r['defs'] or '-'}, user turns {r['user_turns']}{extra}")
    return med


if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else "long"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    task = load_task(task_name)
    print(f"task: {task_name} | model: {MODEL} | correct leader: {task.leader}")

    results = {}
    for name, fn in [("baseline", run_baseline), ("ask_text", run_ask_text), ("steer", run_steer),
                     ("steer_routed", run_steer_routed), ("quick", run_quick)]:
        runs = [fn(task, i) for i in range(n)]
        results[name] = summarize(name.upper(), runs)

    b = results["baseline"]["cost"]
    print("\nRESULT (cost per task, median)")
    for name, med in results.items():
        diff = b - med["cost"]
        print(f"  {name:<12} ${med['cost']:.4f}   saves ${diff:+.4f} vs baseline ({diff / b:+.0%})")
