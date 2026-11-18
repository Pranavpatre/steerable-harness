"""A deep agent you can watch and steer while it works.

Phase 1 of the PRD (live view and steering, S1 to S6, S9 to S13). Everything is
middleware on top of Deep Agents, built on LangGraph checkpointers + threads.

1. check_in (agent-initiated). A tool that calls `interrupt()`. The run pauses,
   the checkpointer saves the whole state under the thread_id, and you see what
   the agent has so far, what is missing, and what it plans to do next. Your
   reply is returned to the agent as the tool result and the run resumes from
   the same point (`Command(resume=...)`).

2. Steering inbox (user-initiated). Anything you type while the agent runs goes
   into a SteerInbox and never blocks. Four modes:
     text            Steer: lands before the agent's next model call.
     /follow text    Follow-up: sent as its own turn after the current task.
     /collect text   Collect: held and batched; lands at the next pause, with the
                     next Steer, or as one turn when the task ends.
     /stop           Interrupt: cancels the run (mid-reply if it is generating),
                     keeps the last checkpoint, offers Resume or Resume with edit.
   Each message shows its state: queued, delivered at step N, or dropped.

3. Step summaries. Before every model call after a tool step, the harness builds
   a short summary from run state (tools called, files touched, output sizes,
   plan progress, tokens), with no extra model call. With `/pause N` (or
   pause_every=N) it also stops every N steps and waits for Continue, a caveat,
   or Interrupt, even when the model would have kept going.

4. Assumption ledger (Phase 2, A1 to A3, A5 to A7). Before any other tool the
   agent calls write_ledger (scope, inputs, constraints, done_when) and the run
   pauses for you to confirm or correct it. Every later change is recorded with
   update_ledger and shown in the step panel. `/ledger field: text` adds your own.
   ToolGuardMiddleware blocks tools until the ledger is confirmed and asks before
   subagent spawns, very large calls, or calls over a share of a token budget.
   Checkpoints go to SQLite, so `--thread ID` resumes a paused run after a restart.

5. SelfReviewMiddleware makes the agent audit its own last few steps every N
   tool calls and decide whether it needs to check in.

6. Token meter and budgets (Phase 3, T1, T2, T4, T6, T8). Every step panel
   shows that call's input, cache read, cache write and output tokens from the
   API's usage fields, and `/tokens` prints the table for the thread. Run and
   step budgets warn at 80% and pause at 100%. Prompt caching uses the 5-minute
   lifetime (1-hour with CACHE_TTL=1h), history is only appended to, and
   thinking settings are pinned.
"""

from __future__ import annotations

import logging
import operator
import os
import queue
import sqlite3
import sys
import threading
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, TodoListMiddleware, hook_config
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from deepagents import create_deep_agent

STOP = "/stop"

# ---------------------------------------------------------------------------
# 1. check_in: the agent pauses itself and hands you the wheel
# ---------------------------------------------------------------------------


@tool
def check_in(
    progress_so_far: str,
    gaps_or_concerns: list[str] | str,
    proposed_next_step: str,
    runtime: ToolRuntime,
    options: list[str] | None = None,
) -> Command:
    """Pause the run and show the user where you are before continuing.

    Call this INSTEAD of assuming whenever:
    - data you need is missing, null, incomplete, or not instrumented
    - you would otherwise have to invent a definition, threshold, or estimate
    - results contradict each other or the user's expectation
    - your plan has changed from what the user asked for
    - the next step is slow, costly, or hard to undo

    Args:
        progress_so_far: What you have done and found so far, with numbers.
        gaps_or_concerns: Each missing data point or doubt, one per item (a list of strings).
        proposed_next_step: What you will do next if the user just says "go".
        options: Optional 2-4 concrete alternatives the user can pick from.
    """
    if isinstance(gaps_or_concerns, str):  # models sometimes send one string, or a JSON list as text
        try:
            parsed = json.loads(gaps_or_concerns)
        except ValueError:
            parsed = None
        gaps_or_concerns = [str(g) for g in parsed] if isinstance(parsed, list) else [gaps_or_concerns]
    reply = interrupt(
        {
            "type": "check_in",
            "progress_so_far": progress_so_far,
            "gaps_or_concerns": gaps_or_concerns,
            "proposed_next_step": proposed_next_step,
            "options": options or [],
        }
    )
    text = (
        f"The user replied: {reply!r}\n"
        "Treat this as authoritative. If it changes the plan, follow the new plan. "
        "If it supplies missing data or a definition, use exactly that. "
        "If it says go / continue / ok, proceed with your proposed next step."
    )
    log = audit_entry(steps_done(runtime.state["messages"]), "pause", "check_in",
                      gaps=gaps_or_concerns, proposed=proposed_next_step, answer=str(reply))
    return Command(update={"audit": [log], "messages": [ToolMessage(text, tool_call_id=runtime.tool_call_id)]})


# ---------------------------------------------------------------------------
# 2. Steering inbox: never blocks, four modes, every message has a state (S2, S4, S5)
# ---------------------------------------------------------------------------

MODES = {"/follow": "follow_up", "/collect": "collect", "/ledger": "ledger", STOP: "interrupt"}


class StopRun(Exception):
    """Raised to cancel a run. The checkpoint before the cancelled step is kept."""


@dataclass
class Steer:
    id: int
    mode: str  # steer | follow_up | collect | ledger | interrupt
    text: str
    status: str = "queued"  # queued | delivered | dropped
    step: int | None = None  # the step it was delivered at
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])  # stable key for the audit log

    def label(self) -> str:
        where = f" at step {self.step}" if self.step is not None else ""
        return f"[#{self.id} {self.mode} · {self.status}{where}]"


class SteerInbox:
    """Thread-safe queue of user messages sent while the agent works.

    `on_change(steer)` is called whenever a message changes state, so a view
    can show queued / delivered at step N / dropped.
    """

    def __init__(self, on_change=None, pause_every: int | None = None) -> None:
        self._lock = threading.Lock()
        self.items: list[Steer] = []
        self.stop = threading.Event()
        self.pause_every = pause_every
        self.on_change = on_change or (lambda s: None)

    def put(self, line: str) -> Steer | None:
        cmd, _, rest = line.strip().partition(" ")
        if cmd == "/pause":
            self.pause_every = int(rest) if rest.strip().isdigit() and int(rest) > 0 else None
            return None
        mode = MODES.get(cmd)
        text = rest.strip() if mode else line.strip()
        if not text and mode != "interrupt":
            return None
        with self._lock:
            s = Steer(len(self.items) + 1, mode or "steer", text)
            self.items.append(s)
        if s.mode == "interrupt":
            self.stop.set()
        self.on_change(s)
        return s

    def _settle(self, items: list[Steer], status: str, step: int | None) -> list[Steer]:
        for s in items:
            s.status, s.step = status, step
            self.on_change(s)
        return items

    def _queued(self, *modes: str) -> list[Steer]:
        return [s for s in self.items if s.status == "queued" and s.mode in modes]

    def take(self, step: int, flush: bool = False, modes: tuple[str, ...] = ("steer", "collect", "ledger")) -> list[Steer]:
        """Messages to deliver before the model call at `step`.

        Steers always go. Collects wait for a pause (`flush`) or ride along with
        a Steer, listed first so the later Steer wins if they conflict.
        """
        with self._lock:
            due = self._queued(*modes)
            if not flush and not any(s.mode in ("steer", "ledger") for s in due):
                due = []
            return self._settle(due, "delivered", step)

    def take_interrupts(self, step: int | None) -> list[Steer]:
        with self._lock:
            return self._settle(self._queued("interrupt"), "delivered", step)

    def after_turn(self, step: int) -> list[str]:
        """Turns to send once the current task ends: leftover steers and collects
        as one turn, then each follow-up as its own turn."""
        with self._lock:
            turns = []
            leftover = self._settle(self._queued("steer", "collect"), "delivered", step)
            if leftover:
                turns.append("\n".join(s.text for s in leftover))
            for s in self._settle(self._queued("follow_up"), "delivered", step):
                turns.append(s.text)
            return turns

    def drop_pending(self) -> list[Steer]:
        with self._lock:
            return self._settle(self._queued("steer", "collect", "ledger", "follow_up", "interrupt"), "dropped", None)


# LangChain logs a warning before re-raising a callback error; /stop is not an error.
logging.getLogger("langchain_core.callbacks.manager").addFilter(lambda r: "StopRun" not in r.getMessage())


class StopOnRequest(BaseCallbackHandler):
    """Cancels a reply mid-generation when /stop arrives (T5: early abort).

    Pass it in the run config's callbacks. It raises from the token callback,
    which aborts the model call; LangGraph keeps the checkpoint from before it.
    """

    raise_error = True

    def __init__(self, inbox: SteerInbox) -> None:
        self.inbox = inbox

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.inbox.stop.is_set():
            raise StopRun("stopped mid-reply")


# ---------------------------------------------------------------------------
# 3. Assumption ledger (A1 to A3): what the agent believes, as an append-only log
# ---------------------------------------------------------------------------

LEDGER_FIELDS = ("scope", "inputs", "constraints", "done_when")
LedgerField = Literal["scope", "inputs", "constraints", "done_when"]
LEDGER_EXEMPT = {"write_ledger", "check_in"}  # the only tools allowed before the ledger is confirmed


class HarnessState(AgentState):
    # Every ledger change is one op. Ops are only ever appended (parallel tool calls
    # can't clash), the current ledger is folded from them, and the log doubles as
    # the record of who changed what at which step.
    ledger_ops: NotRequired[Annotated[list[dict[str, Any]], operator.add]]
    budget_limits: NotRequired[dict[str, Any]]  # T6: budgets you raised at a pause, for this run
    thinking_pins: NotRequired[dict[str, Any]]  # T4: thinking settings per model, fixed at first call
    audit: NotRequired[Annotated[list[dict[str, Any]], operator.add]]  # A8: steers, pauses, decisions


def audit_entry(step: int | None, kind: str, what: str, **detail: Any) -> dict[str, Any]:
    """A8: one audit record. kind is steer, pause, decision or ledger.

    Entries live in run state, so they are checkpointed with the run, survive a
    restart, and a node that re-runs after a pause doesn't write them twice.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {"step": step, "kind": kind, "what": what, "at": now, **{k: v for k, v in detail.items() if v not in (None, "", [])}}


def ledger_audit(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    written = [audit_entry(ops[0]["step"], "ledger", "written", by="agent")] if any(o["op"] == "set" for o in ops) else []
    return written + [audit_entry(o.get("step"), "ledger", o["op"], by=o.get("by"),
                                  change=format_op(o) if "field" in o or o["op"] == "edit" else None)
                      for o in ops if o["op"] != "set"]


def steer_audit(inbox: Any, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Steers that were delivered or dropped and aren't in the log yet."""
    have = {e.get("uid") for e in state.get("audit") or []}
    return [audit_entry(s.step, "steer", s.mode, status=s.status, text=s.text, uid=s.uid)
            for s in list(inbox.items) if s.status != "queued" and s.uid not in have]


def current_ledger(ops: list[dict[str, Any]]) -> tuple[dict[str, list[str]], bool]:
    """Fold the op log into (ledger, confirmed)."""
    ledger: dict[str, list[str]] = {f: [] for f in LEDGER_FIELDS}
    confirmed = False
    for o in ops:
        items = ledger.get(o.get("field"), [])
        if o["op"] == "set":
            ledger[o["field"]] = list(o["items"])
        elif o["op"] == "added":
            items.append(o["item"])
        elif o["op"] == "revised":
            if o["item"] in items:
                items[items.index(o["item"])] = o["new_item"]
            else:
                items.append(o["new_item"])
        elif o["op"] == "dropped" and o["item"] in items:
            items.remove(o["item"])
        elif o["op"] == "confirmed":
            confirmed = True
    return ledger, confirmed


def format_ledger(ledger: dict[str, list[str]]) -> list[str]:
    lines = []
    for f in LEDGER_FIELDS:
        lines.append(f"  {f}:")
        lines += [f"    - {x}" for x in ledger[f]] or ["    (none)"]
    return lines


def format_op(o: dict[str, Any]) -> str:
    who = "you" if o.get("by") == "user" else "agent"
    if o["op"] == "set":
        return f"{o['field']} written ({len(o['items'])})"
    if o["op"] == "edit":
        return f"edit from you: {o['item']}"
    if o["op"] == "revised":
        return f"~ {o['field']}: {o['item']} → {o['new_item']} ({who})"
    sign = {"added": "+", "dropped": "−"}.get(o["op"], o["op"])
    return f"{sign} {o['field']}: {o.get('item', '')} ({who})"


@tool
def write_ledger(
    scope: list[str],
    inputs: list[str],
    constraints: list[str],
    done_when: list[str],
    runtime: ToolRuntime,
) -> Command:
    """Write down your assumptions about the task BEFORE any other tool call.

    The run pauses so the user can confirm or correct them. Other tools are
    blocked until this is done. Be specific and short, one assumption per item.

    Args:
        scope: What is in and out of scope (e.g. "Jul to Oct 2026", "referral channel only").
        inputs: The data, tables, files or definitions you will rely on, and how you read them.
        constraints: Rules you will follow (e.g. "do not interpolate missing months").
        done_when: What the finished answer contains.
    """
    step = steps_done(runtime.state["messages"])
    fields = {"scope": scope, "inputs": inputs, "constraints": constraints, "done_when": done_when}
    ops = [{"op": "set", "field": f, "items": v, "by": "agent", "step": step} for f, v in fields.items()]
    reply = interrupt({"type": "ledger", "ledger": fields})  # the first checkpoint (A1)
    ops.append({"op": "confirmed", "by": "user", "step": step})
    reply = str(reply).strip()
    if reply.lower() in {"", "ok", "go", "yes", "y", "confirm", "1"}:
        text = "The user confirmed the ledger. Follow it. If a step changes any assumption, call update_ledger."
    else:
        ops.append({"op": "edit", "item": reply, "by": "user", "step": step})
        text = (
            f"The user corrected the ledger: {reply!r}\n"
            "Treat this as authoritative. Call update_ledger to record each change, then continue."
        )
    log = [*ledger_audit(ops), audit_entry(step, "pause", "ledger", answer=reply or "confirmed")]
    return Command(update={"ledger_ops": ops, "audit": log, "messages": [ToolMessage(text, tool_call_id=runtime.tool_call_id)]})


@tool
def update_ledger(
    change: Literal["added", "revised", "dropped"],
    field: LedgerField,
    item: str,
    runtime: ToolRuntime,
    new_item: str | None = None,
    reason: str = "",
) -> Command:
    """Record a change to one assumption in the ledger, in the same step it happens.

    Args:
        change: "added" for a new assumption, "revised" to replace `item` with `new_item`,
            "dropped" when an assumption no longer holds.
        field: Which part of the ledger: scope, inputs, constraints or done_when.
        item: The assumption, exactly as written in the ledger for revised/dropped.
        new_item: The replacement text, for "revised" only.
        reason: One short line on why it changed.
    """
    if change == "revised" and not new_item:
        return Command(update={"messages": [ToolMessage("revised needs new_item.", tool_call_id=runtime.tool_call_id, status="error")]})
    op = {"op": change, "field": field, "item": item, "new_item": new_item, "reason": reason,
          "by": "agent", "step": steps_done(runtime.state["messages"])}
    return Command(update={"ledger_ops": [op], "audit": ledger_audit([op]),
                           "messages": [ToolMessage(f"Ledger: {format_op(op)}", tool_call_id=runtime.tool_call_id)]})


# ---------------------------------------------------------------------------
# 4. Step summaries from run state, no model call (S10, S11)
# ---------------------------------------------------------------------------

FILE_WRITERS = {"write_file", "edit_file"}


def is_note(m: Any) -> bool:
    """A HumanMessage the harness added (steer, caveat), not a new question."""
    return isinstance(m, HumanMessage) and m.additional_kwargs.get("harness") == "note"


def note(text: str) -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs={"harness": "note"})


def steps_done(messages: list[Any]) -> int:
    """Model calls so far in the thread. Step N is the Nth model call."""
    return sum(isinstance(m, AIMessage) for m in messages)


def steps_in_turn(messages: list[Any]) -> int:
    n = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and not is_note(m):
            break
        n += isinstance(m, AIMessage)
    return n


def turn_start(messages: list[Any]) -> int:
    """Index of the question that started the current run (steering notes don't count)."""
    return max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and not is_note(m)), default=0)


# USD per 1M tokens. A 1-hour cache write bills at 2x input, a 5-minute write at 1.25x.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "cache_write_5m": 2.50, "cache_write_1h": 4.00, "cache_read": 0.10, "output": 10.00},
}
TOKEN_KEYS = ("input", "cache_read", "cache_write", "cache_write_1h", "output")


def usage(m: Any) -> dict[str, int]:
    """T8: one model call's tokens, from the usage fields the API returns.

    `input` is uncached input only. LangChain's `input_tokens` also counts cache
    reads and writes, so they are subtracted out here.
    """
    u = getattr(m, "usage_metadata", None) or {}
    d = u.get("input_token_details") or {}
    read = d.get("cache_read") or 0
    w1h = d.get("ephemeral_1h_input_tokens") or 0
    w5m = (d.get("ephemeral_5m_input_tokens") or 0) + (d.get("cache_creation") or 0)
    return {
        "input": max(u.get("input_tokens", 0) - read - w5m - w1h, 0),
        "cache_read": read,
        "cache_write": w5m + w1h,
        "cache_write_1h": w1h,
        "output": u.get("output_tokens", 0),
    }


def token_totals(messages: list[Any]) -> dict[str, int]:
    t = dict.fromkeys(TOKEN_KEYS, 0)
    for m in messages:
        if isinstance(m, AIMessage):
            for k, v in usage(m).items():
                t[k] += v
    return t


def budgeted(t: dict[str, int]) -> int:
    """What a budget counts: every token except cache reads, which bill at 1/20 of input."""
    return t["input"] + t["cache_write"] + t["output"]


def model_of(m: Any) -> str | None:
    return (getattr(m, "response_metadata", None) or {}).get("model_name")


def cost(t: dict[str, int], model: str | None) -> float | None:
    p = next((v for k, v in PRICES.items() if model and model.startswith(k)), None)
    if p is None:
        return None
    w5m = t["cache_write"] - t["cache_write_1h"]
    return (t["input"] * p["input"] + t["cache_read"] * p["cache_read"] + w5m * p["cache_write_5m"]
            + t["cache_write_1h"] * p["cache_write_1h"] + t["output"] * p["output"]) / 1e6


def run_cost(messages: list[Any]) -> float | None:
    costs = [cost(usage(m), model_of(m)) for m in messages if isinstance(m, AIMessage)]
    known = [c for c in costs if c is not None]
    return sum(known) if known else None


def token_table(messages: list[Any]) -> list[str]:
    """T8 dashboard: one row per model call in the thread (the /tokens command)."""
    rows = [f"   {'step':>4} {'input':>8} {'cache rd':>9} {'cache wr':>9} {'output':>7} {'cost':>8}  model"]
    n = 0
    for m in messages:
        if isinstance(m, AIMessage) and m.usage_metadata:
            n += 1
            u, c = usage(m), cost(usage(m), model_of(m))
            rows.append(f"   {n:>4} {u['input']:>8,} {u['cache_read']:>9,} {u['cache_write']:>9,} {u['output']:>7,} "
                        f"{'$%.4f' % c if c is not None else '?':>8}  {model_of(m) or ''}")
    t, c = token_totals(messages), run_cost(messages)
    rows.append(f"   {'all':>4} {t['input']:>8,} {t['cache_read']:>9,} {t['cache_write']:>9,} {t['output']:>7,} "
                f"{'$%.4f' % c if c is not None else '?':>8}")
    return rows


def step_summary(state: dict[str, Any]) -> dict[str, Any]:
    """What the last step did, what it changed, and progress against the plan."""
    msgs = state["messages"]
    i = max(j for j, m in enumerate(msgs) if isinstance(m, AIMessage))
    ai = msgs[i]
    results = [m for m in msgs[i + 1:] if isinstance(m, ToolMessage)]
    calls = Counter(tc["name"] for tc in ai.tool_calls)
    todos = state.get("todos") or []
    said = ai.text.strip().splitlines()[0][:140] if ai.text.strip() else ""
    run = msgs[turn_start(msgs):]
    return {
        "step": steps_done(msgs),
        "did": ", ".join(f"{n} ×{c}" if c > 1 else n for n, c in calls.items()) or "replied, no tools",
        "results": len(results),
        "output_chars": sum(len(str(m.content)) for m in results),
        "errors": sum(getattr(m, "status", None) == "error" for m in results),
        "files_changed": sorted({tc["args"].get("file_path") for tc in ai.tool_calls if tc["name"] in FILE_WRITERS} - {None}),
        "files_total": len(state.get("files") or {}),
        "plan_done": sum(t.get("status") == "completed" for t in todos),
        "plan_total": len(todos),
        "plan_now": next((t.get("content") for t in todos if t.get("status") == "in_progress"), None),
        "said": said,
        "model": model_of(ai),
        "step_tokens": usage(ai),
        "run_tokens": token_totals(run),
        "cost_step": cost(usage(ai), model_of(ai)),
        "cost_run": run_cost(run),
        "ledger_changes": ledger_changes(state.get("ledger_ops") or [], steps_done(msgs)),
    }


def ledger_changes(ops: list[dict[str, Any]], step: int) -> list[str]:
    """A3: the ledger changes made at `step`, one line each."""
    here = [o for o in ops if o.get("step") == step]
    out = ["ledger written"] if any(o["op"] == "set" for o in here) else []
    return out + [format_op(o) for o in here if o["op"] in {"added", "revised", "dropped", "edit"}]


def format_summary(s: dict[str, Any]) -> list[str]:
    st, run = s["step_tokens"], s["run_tokens"]
    head = f"── step {s['step']}"
    if s.get("model"):
        head += f" · {s['model']}"
    if s.get("cost_step") is not None:
        head += f" · ${s['cost_step']:.4f} this step"
    did = f"   did:   {s['did']}"
    if s["results"]:
        did += f" → {s['results']} result(s), {s['output_chars']:,} chars"
    if s["errors"]:
        did += f", {s['errors']} error(s)"
    files = ", ".join(s["files_changed"]) if s["files_changed"] else "none changed this step"
    used = budgeted(run)
    meter = f"   run:    {used:,} tokens"
    if s.get("run_limit"):
        meter += f" of {s['run_limit']:,} budget ({used / s['run_limit']:.0%})"
    meter += f", plus {run['cache_read']:,} cache reads"
    if s.get("cost_run") is not None:
        meter += f" · ${s['cost_run']:.3f} so far"
    lines = [
        head,
        f"   tokens: in {st['input']:,} · cache read {st['cache_read']:,} · "
        f"cache write {st['cache_write']:,} · out {st['output']:,}",
        meter,
        *(f"   ⚠ {w}" for w in s.get("warnings") or []),
        did,
        f"   files: {files} ({s['files_total']} in the run)",
    ]
    if s["plan_total"]:
        now = f" · now: {s['plan_now']}" if s["plan_now"] else ""
        lines.append(f"   plan:  {s['plan_done']}/{s['plan_total']} done{now}")
    if s.get("ledger_changes"):
        lines.append("   ledger: " + "; ".join(s["ledger_changes"]))
    if s.get("brief"):
        lines.append(f"   recap: {s['brief']}")
    if s["said"]:
        lines.append(f"   said:  {s['said']}")
    return lines


PAUSE_OPTIONS = ["Continue", "Steer with a caveat", "Interrupt"]
BUDGET_OPTIONS = ["Continue with more budget", "Set a new budget", "Stop"]


@dataclass
class Budget:
    """T6: token budgets, counted with `budgeted()` (cache reads excluded).

    `run` covers one task, from your question to the answer, follow-ups reset it.
    `step` covers one model call. A warning shows at `warn` (80%) and the run
    pauses at the next step boundary once a budget reaches 100%.
    """

    run: int | None = None
    step: int | None = None
    warn: float = 0.8


class SteeringMiddleware(AgentMiddleware):
    """Runs at every step boundary, before each model call:

    1. /stop pending → cancel the run (StopRun), keeping the last checkpoint.
    2. A tool step just finished → emit its summary to the live view; in pause
       mode, every N steps, wait for Continue / caveat / Interrupt.
    3. Drain the inbox and append due messages as one user turn at the END of
       the message list, so the cached prompt prefix stays valid.

    It uses a single interrupt() per node run (A6), so resuming is unambiguous.
    """

    state_schema = HarnessState

    def __init__(self, inbox: SteerInbox, summary_model: Any = None, budget: Budget | None = None) -> None:
        super().__init__()
        self.inbox = inbox
        self.summary_model = summary_model  # S11: optional cheap model for a one-line recap
        self.budget = budget or Budget()
        self._briefs: dict[int, str] = {}

    def _limits(self, state: dict[str, Any]) -> dict[str, Any]:
        """This run's budgets: the defaults, or what you raised them to at a budget pause."""
        msgs = state["messages"]
        run_id = msgs[turn_start(msgs)].id
        saved = state.get("budget_limits") or {}
        if saved.get("run_id") != run_id:
            saved = {}
        return {"run_id": run_id, "run": saved.get("run", self.budget.run), "step": saved.get("step", self.budget.step)}

    def _check_budget(self, s: dict[str, Any], limits: dict[str, Any]) -> list[dict[str, Any]]:
        """T6: add 80% warnings to the step panel; return the budgets at 100%."""
        run_used, step_used = budgeted(s["run_tokens"]), budgeted(s["step_tokens"])
        warn, over = [], []
        if lim := limits["run"]:
            if run_used >= lim:
                over.append({"kind": "run", "used": run_used, "limit": lim})
            elif run_used >= self.budget.warn * lim > run_used - step_used:  # crossed 80% this step
                warn.append(f"run has used {run_used / lim:.0%} of its {lim:,}-token budget")
        if lim := limits["step"]:
            if step_used >= lim:
                over.append({"kind": "step", "used": step_used, "limit": lim})
            elif step_used >= self.budget.warn * lim:
                warn.append(f"step {s['step']} used {step_used / lim:.0%} of the {lim:,}-token step budget")
        s["run_limit"], s["warnings"] = limits["run"], warn
        return over

    def _brief(self, s: dict[str, Any], state: dict[str, Any]) -> str | None:
        if self.summary_model is None:
            return None
        if s["step"] not in self._briefs:  # the node re-runs after a pause; call once
            facts = "\n".join(format_summary(s))
            last = state["messages"][-1]
            prompt = (
                "In one sentence for the user, say what this agent step found. "
                f"No preamble.\n{facts}\nLast tool result (truncated): {str(last.content)[:1500]}"
            )
            self._briefs[s["step"]] = self.summary_model.invoke([HumanMessage(prompt)]).text.strip()
        return self._briefs[s["step"]]

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        msgs = state["messages"]
        step = steps_done(msgs) + 1  # the model call about to happen
        if self.inbox.stop.is_set():
            self.inbox.take_interrupts(step - 1)
            raise StopRun(f"stopped before step {step}")

        paused, caveat, update, log = False, None, {}, []
        if isinstance(msgs[-1], ToolMessage):  # a tool step just finished
            s = step_summary(state)
            limits = self._limits(state)
            over = self._check_budget(s, limits)
            log += [audit_entry(s["step"], "decision", "budget_warning", detail=w) for w in s["warnings"]]
            s["brief"] = self._brief(s, state)
            get_stream_writer()({"type": "step_summary", **s})
            every = self.inbox.pause_every
            reply = None
            if over:  # T6: a budget hit 100%; this pause replaces the step pause
                reply = interrupt({"type": "budget", **s, "over": over, "extra": self.budget.run,
                                   "options": BUDGET_OPTIONS})
            elif every and steps_in_turn(msgs) % every == 0:
                reply = interrupt({"type": "step_pause", **s, "options": PAUSE_OPTIONS})
            if reply is not None:
                paused = True
                action = reply.get("action") if isinstance(reply, dict) else "continue"
                log.append(audit_entry(s["step"], "pause", "budget" if over else "step_pause", answer=action,
                                       text=reply.get("text") if isinstance(reply, dict) else None,
                                       tokens=reply.get("tokens") if isinstance(reply, dict) else None,
                                       over=[f"{o['kind']} {o['used']:,}/{o['limit']:,}" for o in over]))
                if action in {"interrupt", "stop"}:
                    return {
                        "messages": [AIMessage("Stopped at your request.", additional_kwargs={"harness": "stopped"})],
                        "jump_to": "end",
                        "audit": log + steer_audit(self.inbox, state),
                    }
                if action == "caveat":
                    caveat = reply["text"]
                if over:
                    for o in over:
                        if action == "raise":
                            limits[o["kind"]] = int(reply["tokens"])
                        elif o["kind"] == "run":  # Continue: one more run budget on top of what's used
                            limits["run"] = max(o["limit"], o["used"]) + (self.budget.run or 0)
                    update["budget_limits"] = limits

        due = self.inbox.take(step, flush=paused)
        log += steer_audit(self.inbox, state)  # this boundary's steers, plus any /stop or drop since the last one
        if log:
            update["audit"] = log
        if not due and not caveat:
            return update or None
        parts, ops = [], []
        notes = [s for s in due if s.mode != "ledger"]
        if notes:
            lines = "\n".join(f"- {s.text}" for s in notes)
            parts.append(f"[Note from the user, sent while you were working]\n{lines}")
            if len(notes) > 1:
                parts.append("If these conflict, the later line wins.")
        for s in due:
            if s.mode == "ledger":  # A2: a user edit to the ledger, delivered as a steer
                field, _, item = s.text.partition(":")
                ops.append({"op": "added", "field": field.strip(), "item": item.strip(), "by": "user", "step": step})
                parts.append(f"[Ledger edit from the user] {field.strip()}: + {item.strip()}")
        if caveat:  # S12: a caveat becomes a constraint in the ledger
            ops.append({"op": "added", "field": "constraints", "item": caveat, "by": "user", "step": step})
            parts.append(f"[Caveat from the user after reviewing step {step - 1}]\n- {caveat}")
        parts.append("Adjust your current plan to this before your next step.")
        update["messages"] = [note("\n".join(parts))]
        if ops:
            update["ledger_ops"] = ops
            update["audit"] = log + ledger_audit(ops)
        return update


# ---------------------------------------------------------------------------
# 5. Pre-tool guard (A1, A5, A6): ledger first, then ask before costly calls
# ---------------------------------------------------------------------------


class ToolGuardMiddleware(AgentMiddleware):
    """Runs after each model call, before its tools run.

    - Ledger first (A1): until the ledger is confirmed, any tool other than
      write_ledger / check_in gets a "blocked" result instead of running.
    - Ask first (A5): a subagent spawn, a call with very large arguments, or a
      call estimated to cost more than `budget_share` of `budget_tokens` pauses
      for approval, with the estimate shown.

    Blocked or declined calls get an artificial tool result, so LangGraph's
    conditional edge routes straight back to the model for a re-prompt. There is
    one interrupt() per node run and no loop around it (A6).
    """

    state_schema = HarnessState

    def __init__(
        self,
        require_ledger: bool = True,
        ask_tools: tuple[str, ...] = ("task",),
        large_args_chars: int = 20_000,
        budget_tokens: int | None = None,
        budget_share: float = 0.25,
        subagent_calls: int = 5,
    ) -> None:
        super().__init__()
        self.require_ledger = require_ledger
        self.ask_tools = set(ask_tools)
        self.large_args_chars = large_args_chars
        self.budget_tokens = budget_tokens
        self.budget_share = budget_share
        self.subagent_calls = subagent_calls

    def estimate(self, tc: dict[str, Any], msgs: list[Any]) -> int:
        """Rough tokens this call will cost. A subagent is priced as `subagent_calls`
        model calls at this run's average size so far; other calls by argument size."""
        if tc["name"] == "task":
            n = max(steps_done(msgs), 1)
            return budgeted(token_totals(msgs)) // n * self.subagent_calls
        return len(str(tc["args"])) // 4

    def reasons(self, tc: dict[str, Any], msgs: list[Any]) -> list[str]:
        why = []
        if tc["name"] in self.ask_tools:
            why.append("spawns a subagent" if tc["name"] == "task" else f"{tc['name']} needs approval")
        if len(str(tc["args"])) > self.large_args_chars:
            why.append(f"large call ({len(str(tc['args'])):,} chars of arguments)")
        if self.budget_tokens and self.estimate(tc, msgs) > self.budget_share * self.budget_tokens:
            why.append(f"over {self.budget_share:.0%} of the {self.budget_tokens:,}-token budget")
        return why

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        msgs = state["messages"]
        ai = msgs[-1]
        if not isinstance(ai, AIMessage) or not ai.tool_calls:
            return None
        _, confirmed = current_ledger(state.get("ledger_ops") or [])
        out: list[ToolMessage] = []
        review = []
        step, log = steps_done(msgs), []
        for tc in ai.tool_calls:
            if self.require_ledger and not confirmed and tc["name"] not in LEDGER_EXEMPT:
                log.append(audit_entry(step, "decision", "blocked", tool=tc["name"], why="ledger not confirmed"))
                out.append(ToolMessage(
                    "Blocked: call write_ledger with your assumptions first. The user confirms it, "
                    "then you can run this call.", tool_call_id=tc["id"], name=tc["name"], status="error"))
            elif why := self.reasons(tc, msgs):
                review.append({"id": tc["id"], "name": tc["name"], "args": str(tc["args"])[:300],
                               "reasons": why, "est_tokens": self.estimate(tc, msgs)})
        if review:
            reply = interrupt({"type": "approval", "calls": review})
            approved = str(reply).strip().lower() in {"", "y", "yes", "ok", "go", "approve", "1"}
            log.append(audit_entry(step, "pause", "approval", answer="approved" if approved else str(reply),
                                   calls=[f"{r['name']} ({'; '.join(r['reasons'])}, ~{r['est_tokens']:,} tokens)" for r in review]))
            if not approved:
                for r in review:
                    out.append(ToolMessage(f"The user declined this call: {reply!r}. Adjust and continue.",
                                           tool_call_id=r["id"], name=r["name"], status="error"))
        update = {"messages": out, "audit": log}
        return {k: v for k, v in update.items() if v} or None


# ---------------------------------------------------------------------------
# 6. Self-review: every N tool calls, audit the trajectory
# ---------------------------------------------------------------------------

SELF_REVIEW = (
    "[Self-review checkpoint] Before your next step:\n"
    "1. Start your reply with ONE line, under 30 words, in the form\n"
    "   `Progress: <what you have found so far, with numbers> | Next: <your next step>`\n"
    "   The user reads this line live and may redirect you.\n"
    "2. Look back at the recent tool results. Do they still answer what the user asked? "
    "Is any data missing, null, or suspicious? Are you about to assume something? "
    "If yes to any, call check_in now. Otherwise continue with your next step."
)
PROGRESS_PREFIX = "Progress:"


class SelfReviewMiddleware(AgentMiddleware):
    """Every `every` tool results, asks for a one-line progress note and a self-audit.

    The prompt is saved to history as a harness note, like a steer. Sending it
    once and dropping it would change the prompt prefix on the next call and
    cost a cache miss (T1); kept, it is about 100 tokens read from cache.
    """

    state_schema = HarnessState

    def __init__(self, every: int = 4) -> None:
        super().__init__()
        self.every = every

    def _tool_results_since_last_user_turn(self, messages: list[Any]) -> int:
        count = 0
        for m in reversed(messages):
            if isinstance(m, HumanMessage) and not is_note(m):  # a steer doesn't reset the clock
                break
            if isinstance(m, ToolMessage):
                if m.name == "check_in":  # a check-in resets the clock
                    break
                count += 1
        return count

    def due(self, messages: list[Any]) -> bool:
        """True when the latest batch of tool results crossed a multiple of `every`.

        Models often call several tools at once, so the count can jump from 3 to 5
        and never land exactly on 4. Comparing before and after the batch catches it.
        """
        n = self._tool_results_since_last_user_turn(messages)
        batch = 0
        for m in reversed(messages):
            if is_note(m):  # a steer that just landed at this boundary
                continue
            if not isinstance(m, ToolMessage):
                break
            batch += 1
        return n > 0 and n // self.every > (n - batch) // self.every

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        if self.due(state["messages"]):
            log = audit_entry(steps_done(state["messages"]) + 1, "decision", "self_review")  # sent with that step
            return {"messages": [note(SELF_REVIEW)], "audit": [log]}
        return None


class CacheUsageMiddleware(AgentMiddleware):
    """T8: label cache writes with the lifetime that was asked for.

    A streamed reply's usage carries the total cache-write count but not the
    split into 5-minute and 1-hour writes, and 1-hour writes bill at 2x input
    against 1.25x. When the split is missing, the writes came from this
    harness's own breakpoints, so they are recorded at its TTL.
    """

    def __init__(self, ttl: str) -> None:
        super().__init__()
        self.key = f"ephemeral_{ttl}_input_tokens"

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        response = handler(request)
        if isinstance(request.model, ChatAnthropic):
            for m in getattr(response, "result", None) or []:
                d = (getattr(m, "usage_metadata", None) or {}).get("input_token_details")
                if d and d.get("cache_creation") and not (d.get("ephemeral_5m_input_tokens") or d.get("ephemeral_1h_input_tokens")):
                    d[self.key], d["cache_creation"] = d["cache_creation"], 0
        return response


class ThinkingPinMiddleware(AgentMiddleware):
    """T4: thinking settings never change mid-run.

    The first call to each model in a thread fixes its thinking settings in state.
    If a later request asks for different ones (a middleware, a router), the
    original settings are put back and the live view says so, because a change
    would invalidate the cached message prefix.
    """

    state_schema = HarnessState

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        model = request.model
        name = getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__
        now = request.model_settings.get("thinking", getattr(model, "thinking", None))
        pins = request.state.get("thinking_pins") or {}
        if name not in pins:
            pin = Command(update={"thinking_pins": {**pins, name: now}})
            return ExtendedModelResponse(model_response=handler(request), command=pin)
        if now != pins[name]:
            get_stream_writer()({"type": "thinking_pinned", "model": name, "asked": now, "kept": pins[name]})
            settings = {k: v for k, v in request.model_settings.items() if k != "thinking"}
            if pins[name] is not None:
                settings["thinking"] = pins[name]
            request = request.override(model_settings=settings)
            log = audit_entry(steps_done(request.messages) + 1, "decision", "thinking_pinned",
                              model=name, asked=json.dumps(now), kept=json.dumps(pins[name]))
            return ExtendedModelResponse(model_response=handler(request), command=Command(update={"audit": [log]}))
        return handler(request)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an analysis agent that works WITH the user, not for them in isolation.

Rules for working:
- Before any other tool, call `write_ledger` with your assumptions: scope, inputs (the data,
  definitions and files you will rely on), constraints, and what done looks like. The user
  confirms it at the first checkpoint. Other tools are blocked until then.
- Whenever a step changes an assumption, call `update_ledger` (added / revised / dropped)
  in that same step. The user watches the ledger to spot drift.
- Plan briefly, then execute step by step.
- Never fill a gap with an assumption. If data is missing or ambiguous, call `check_in`
  with what you have so far, the specific gaps, and your proposed next step.
- Messages tagged "[Note from the user, sent while you were working]" or "[Caveat from
  the user ...]" arrive mid-run. They override your plan. Acknowledge them in one line
  and adapt.
- When you finish, state any remaining caveats plainly."""


ANTHROPIC_API = "https://api.anthropic.com"


def resolve_model(model: Any) -> Any:
    """Turn a "provider:model" string into a chat model.

    Anthropic models always go to the real API. Other tools (the Claude desktop
    app, proxies) can set ANTHROPIC_BASE_URL in your shell, and the SDK would
    otherwise send your key there. To use a proxy on purpose, set
    HARNESS_ANTHROPIC_BASE_URL instead.
    """
    if isinstance(model, str) and model.startswith("anthropic:"):
        from langchain.chat_models import init_chat_model

        base_url = os.environ.get("HARNESS_ANTHROPIC_BASE_URL", ANTHROPIC_API)
        return init_chat_model(model, base_url=base_url)
    return model


CHECKPOINTS = Path(__file__).parent / ".harness" / "checkpoints.sqlite"


def sqlite_checkpointer(path: Path = CHECKPOINTS) -> SqliteSaver:
    """A checkpointer on disk, so a paused run survives a restart (A7)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def build_agent(
    model: Any,
    tools: list[Any],
    inbox: SteerInbox | None = None,
    review_every: int = 4,
    summary_model: Any = None,
    checkpointer: Any = None,
    guard: ToolGuardMiddleware | None = None,
    budget: Budget | None = None,
    cache_ttl: Literal["5m", "1h"] = "5m",
    middleware: Sequence[AgentMiddleware] = (),
):
    """`inbox.pause_every` turns on pause-every-N-steps mode (S13).

    `checkpointer` defaults to SQLite under .harness/. Pass InMemorySaver() for
    throwaway runs. `guard` sets the ledger rule and approval thresholds.
    `budget` sets run and step token budgets (T6). `cache_ttl` is the prompt
    cache lifetime (T2): 5 minutes by default. Use "1h" when you expect to pause
    for longer than 5 minutes; its writes cost 2x input against 1.25x. `middleware` runs outermost.
    """
    model = resolve_model(model)
    inbox = inbox if inbox is not None else SteerInbox()
    return create_deep_agent(
        model=model,
        tools=[check_in, write_ledger, update_ledger, *tools],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            *middleware,
            TodoListMiddleware(),  # write_todos; deepagents 0.7 no longer adds it, and the live view shows the plan
            SteeringMiddleware(inbox, summary_model=resolve_model(summary_model), budget=budget),
            guard or ToolGuardMiddleware(budget_tokens=budget.run if budget else None),
            SelfReviewMiddleware(review_every),
            ThinkingPinMiddleware(),
            CacheUsageMiddleware(cache_ttl),
            # Same name as deepagents' own, so it replaces it in place: breakpoints after
            # the system prompt, after the tools, and on the latest message, with our TTL.
            AnthropicPromptCachingMiddleware(ttl=cache_ttl, unsupported_model_behavior="ignore"),
        ],
        checkpointer=checkpointer if checkpointer is not None else sqlite_checkpointer(),
    )


# ---------------------------------------------------------------------------
# Terminal live view (S1, S9)
# ---------------------------------------------------------------------------

HINT = "   [type to steer · /follow · /collect · /ledger · /tokens · /audit · /stop · /pause N]"
AUDIT_DIR = Path(__file__).parent / ".harness" / "audit"


def audit_log(values: dict[str, Any], inbox: SteerInbox | None = None) -> list[dict[str, Any]]:
    """A8: the run's audit log, plus steers the graph hasn't logged yet (a /stop or
    a dropped message after the last step boundary)."""
    log = list(values.get("audit") or [])
    return log + (steer_audit(inbox, values) if inbox else [])


def format_audit(log: list[dict[str, Any]]) -> list[str]:
    rows = []
    for e in log:
        rest = " · ".join(f"{k}: {v if not isinstance(v, list) else '; '.join(map(str, v))}"
                          for k, v in e.items() if k not in {"step", "kind", "what", "at", "uid"})
        step = "-" if e.get("step") is None else e["step"]
        rows.append(f"   step {step:>3}  {e['kind']:<8} {e['what']:<15} {rest}"[:220])
    return rows or ["   (nothing logged yet)"]


def export_audit(values: dict[str, Any], thread_id: str, inbox: SteerInbox | None = None,
                 directory: Path = AUDIT_DIR) -> Path:
    """Write the whole log to .harness/audit/<thread>.jsonl, one JSON object per line."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{thread_id}.jsonl"
    path.write_text("".join(json.dumps(e, default=str) + "\n" for e in audit_log(values, inbox)))
    return path


def _print_check_in(payload: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("CHECK-IN: the agent paused before continuing")
    print("-" * 60)
    print("So far:\n  " + payload["progress_so_far"].replace("\n", "\n  "))
    if payload["gaps_or_concerns"]:
        print("Missing / concerns:")
        for g in payload["gaps_or_concerns"]:
            print(f"  - {g}")
    print(f"Next step if you say go:\n  {payload['proposed_next_step']}")
    for i, o in enumerate(payload["options"], 1):
        print(f"  [{i}] {o}")
    print("=" * 60)
    print("Your reply (number, 'go', or anything to add/change):")


def _print_update(update: dict[str, Any], show_text: bool = True) -> None:
    for node, data in update.items():
        if not isinstance(data, dict):
            continue
        for m in data.get("messages", []) or []:
            if isinstance(m, AIMessage):
                text = m.text.strip()  # text blocks only; skips thinking + tool_use blocks
                if show_text and text.startswith(PROGRESS_PREFIX):
                    line, _, rest = text.partition("\n")
                    print(f"\n>>> {line}  (type a note to redirect, /stop to halt)")
                    text = rest.strip()
                if show_text and text:
                    print(f"\nagent> {text}")
                for tc in m.tool_calls:
                    if tc["name"] != "check_in":
                        print(f"  -> {tc['name']}({str(tc['args'])[:120]})")
            elif isinstance(m, ToolMessage) and m.name != "check_in":
                print(f"  <- {m.name}: {str(m.content)[:160]}")


class LiveView:
    """Prints tokens as they stream, tool calls, step panels and steer states."""

    def __init__(self) -> None:
        self._msg_id = None
        self._shown_steps: set[int] = set()

    def token(self, chunk: Any, meta: dict[str, Any]) -> None:
        if not isinstance(chunk, (AIMessageChunk, AIMessage)):
            return
        text = chunk.text
        if not text:
            return
        if chunk.id != self._msg_id:
            self._msg_id = chunk.id
            print("\nagent> ", end="")
        print(text, end="", flush=True)

    def event(self, e: dict[str, Any]) -> None:
        if e.get("type") == "step_summary" and e["step"] not in self._shown_steps:
            self._shown_steps.add(e["step"])
            print("\n" + "\n".join(format_summary(e)) + "\n" + HINT)
        elif e.get("type") == "thinking_pinned":
            print(f"\n   kept {e['model']}'s thinking settings as they were at the start "
                  "(changing them mid-run resets the prompt cache)")

    def steer(self, s: Steer) -> None:
        print(f"\n   {s.label()} {s.text[:80]}")


def stream_run(agent: Any, payload: Any, config: dict[str, Any], inbox: SteerInbox, view: LiveView):
    """Run until the turn ends, pauses or is cancelled. Returns (outcome, interrupt_value)."""
    inbox.stop.clear()
    try:
        for mode, chunk in agent.stream(payload, config, stream_mode=["messages", "updates", "custom"]):
            if mode == "messages":
                view.token(*chunk)
            elif mode == "custom":
                view.event(chunk)
            elif "__interrupt__" in chunk:
                return "paused", chunk["__interrupt__"][0].value
            else:
                _print_update(chunk, show_text=False)
    except StopRun:
        inbox.take_interrupts(steps_done(agent.get_state(config).values.get("messages") or []) + 1)
        return "cancelled", None
    last = agent.get_state(config).values["messages"][-1]
    if last.additional_kwargs.get("harness") == "stopped":
        return "stopped", None
    return "done", None


def _pause_reply(answer: str, ask) -> dict[str, str]:
    a = answer.strip()
    if a.lower() in {"", "1", "go", "c", "continue", "ok"}:
        return {"action": "continue"}
    if a.lower() in {"3", "interrupt", STOP}:
        return {"action": "interrupt"}
    if a == "2":
        a = ask("caveat> ")
    return {"action": "caveat", "text": a}


def _budget_reply(answer: str, ask) -> dict[str, Any]:
    a = answer.strip().replace(",", "")
    if a.lower() in {"", "1", "go", "c", "continue", "ok"}:
        return {"action": "continue"}
    if a.lower() in {"3", "stop", STOP}:
        return {"action": "stop"}
    if a == "2":
        a = ask("new budget in tokens> ").strip().replace(",", "")
    if a.isdigit():
        return {"action": "raise", "tokens": int(a)}
    return {"action": "caveat", "text": a}  # keep going, with this note (e.g. "wrap up now")


def _print_ledger_checkpoint(payload: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("LEDGER: the agent's assumptions, before it runs any tool")
    print("-" * 60)
    print("\n".join(format_ledger(payload["ledger"])))
    print("=" * 60)
    print("[Enter] confirm · or type what to change:")


def _print_approval(payload: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("APPROVAL: these calls are costly, so the agent asks first")
    for c in payload["calls"]:
        print(f"  - {c['name']}({c['args'][:120]})")
        print(f"    why: {'; '.join(c['reasons'])} · est. ~{c['est_tokens']:,} tokens")
    print("=" * 60)
    print("[Enter] approve · or type why not / what to do instead:")


def run_terminal(agent: Any, inbox: SteerInbox, thread_id: str | None = None) -> None:
    """Chat loop. While the agent runs, every line you type goes to the inbox.

    When the agent pauses (ledger, check-in, approval, step pause, or after
    /stop), the next line you type is your answer. Pass `thread_id` to pick up
    a run from an earlier session, including one that was paused.
    """
    view = LiveView()
    inbox.on_change = view.steer
    lines: queue.Queue[str] = queue.Queue()
    waiting = threading.Event()
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [StopOnRequest(inbox)]}

    def show_ledger() -> None:
        values = agent.get_state(config).values
        ledger, confirmed = current_ledger(values.get("ledger_ops") or [])
        print(f"\n   ledger ({'confirmed' if confirmed else 'not confirmed yet'}):")
        print("\n".join("   " + x for x in format_ledger(ledger)))

    def read_stdin() -> None:
        for line in sys.stdin:
            line = line.rstrip("\n")
            cmd, _, rest = line.strip().partition(" ")
            if cmd == "/tokens":  # works at any prompt, and while the agent runs
                print("\n" + "\n".join(token_table(agent.get_state(config).values.get("messages") or [])))
            elif cmd == "/audit":
                print("\n" + "\n".join(format_audit(audit_log(agent.get_state(config).values, inbox))))
            elif waiting.is_set():
                lines.put(line)
            elif cmd == "/pause":
                inbox.put(line)
                print(f"\n   pause mode: {'every ' + str(inbox.pause_every) + ' steps' if inbox.pause_every else 'off'}")
            elif cmd == "/ledger" and not rest:
                show_ledger()
            elif cmd == "/ledger" and rest.partition(":")[0].strip() not in LEDGER_FIELDS:
                print(f"\n   usage: /ledger <{'|'.join(LEDGER_FIELDS)}>: <text>   (or /ledger to show it)")
            else:
                inbox.put(line)

    def ask(prompt: str) -> str:
        waiting.set()
        print(prompt, end="", flush=True)
        answer = lines.get()
        waiting.clear()
        return answer

    def answer(pending: dict[str, Any], step: int) -> Command:
        """Show a pause and turn the user's reply into the resume value."""
        kind = pending.get("type")
        if kind == "step_pause":
            print(f"\n   Paused after step {pending['step']}: [1] Continue  [2] Steer with a caveat  [3] Interrupt")
            return Command(resume=_pause_reply(ask("> "), ask))
        if kind == "budget":
            for o in pending["over"]:
                print(f"\n   BUDGET: the {o['kind']} used {o['used']:,} of its {o['limit']:,}-token budget")
            more = f" (+{pending['extra']:,} tokens)" if pending.get("extra") else ""
            print(f"   [1] Continue{more}  [2] Set a new budget  [3] Stop  · or type a note to continue with")
            return Command(resume=_budget_reply(ask("> "), ask))
        if kind == "ledger":
            _print_ledger_checkpoint(pending)
        elif kind == "approval":
            _print_approval(pending)
        elif kind == "check_in":
            _print_check_in(pending)
        else:
            print(f"\nINTERRUPT: {pending}")
        # Messages typed during the run but not yet delivered belong to this answer.
        early = [s.text for s in inbox.take(step + 1, flush=True, modes=("steer", "collect"))]
        reply = "\n".join([*early, ask("> ")]).strip()
        opts = pending.get("options") or []
        if reply.isdigit() and 0 < int(reply) <= len(opts):
            reply = opts[int(reply) - 1]
        return Command(resume=reply)  # resume from the saved checkpoint

    threading.Thread(target=read_stdin, daemon=True).start()
    print(f"thread {thread_id}  (resume later: python steerable.py --thread {thread_id})")
    print(f"Ask a question. While it runs:\n{HINT.strip()}")

    turns: list[str] = []
    snapshot = agent.get_state(config)
    payload: Any = None
    if snapshot.interrupts:  # A7: a run paused in an earlier session picks up here
        print(f"\nThis thread was paused at step {steps_done(snapshot.values['messages'])}. Resuming.")
        payload = answer(snapshot.interrupts[0].value, steps_done(snapshot.values["messages"]))
    while True:
        if payload is None:
            if turns:
                question = turns.pop(0)
                print(f"\nyou (queued)> {question}")
            else:
                question = ask("\nyou> ")
                if question.strip() in {"exit", "quit"}:
                    return
            payload = {"messages": [HumanMessage(content=question)]}
        while True:  # one turn = possibly several pause/resume cycles
            outcome, pending = stream_run(agent, payload, config, inbox, view)
            values = agent.get_state(config).values
            export_audit(values, thread_id, inbox)  # A8: the file is current at every pause and stop
            step = steps_done(values["messages"])
            if outcome == "done":
                turns.extend(inbox.after_turn(step))
                payload = None
                break
            if outcome in {"cancelled", "stopped"}:
                print(f"\n■ Stopped after step {step}. The last checkpoint is kept.")
                reply = ask("  [Enter] resume · type an edit to resume with it · /new to drop it\n> ")
                if reply.strip() == "/new":
                    inbox.drop_pending()
                    payload = None
                    break
                if reply.strip():
                    payload = {"messages": [note(f"[Note from the user]\n- {reply.strip()}")]}
                elif outcome == "cancelled":
                    payload = None  # stream(None) re-runs the cancelled step from the checkpoint
                else:
                    payload = {"messages": [note("Continue from where you stopped.")]}
                continue
            payload = answer(pending, step)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # picks up ANTHROPIC_API_KEY etc. from .env

    if os.environ.get("HARNESS_TASK") == "drift":  # 8 countries x 4 months, planted NG anomaly
        from drift_task import TOOLS
    else:  # the short signups task with a missing month and a NULL column
        from example_tools import TOOLS

    model = os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5")
    env = {k: int(os.environ[k]) for k in ("PAUSE_EVERY", "RUN_BUDGET", "STEP_BUDGET") if os.environ.get(k)}
    box = SteerInbox(pause_every=env.get("PAUSE_EVERY"))
    budget = Budget(run=env.get("RUN_BUDGET"), step=env.get("STEP_BUDGET"))
    thread = sys.argv[sys.argv.index("--thread") + 1] if "--thread" in sys.argv else None
    ttl = os.environ.get("CACHE_TTL", "5m")  # 1h if you expect to step away mid-run
    run_terminal(build_agent(model, TOOLS, box, review_every=4, budget=budget, cache_ttl=ttl), box, thread_id=thread)
