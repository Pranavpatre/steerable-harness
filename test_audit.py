"""Offline tests for A8: every steer, pause and decision goes to an audit log with its step."""

import json
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from example_tools import TOOLS
from scripted_model import ScriptedModel, call, used
from steerable import (Budget, StopRun, SteerInbox, ToolGuardMiddleware, audit_log, build_agent,
                       export_audit, format_audit, sqlite_checkpointer)

LEDGER = {"scope": ["Jul to Oct 2026"], "inputs": ["query_signups"],
          "constraints": ["no interpolation"], "done_when": ["monthly table"]}


def run(agent, payload, cfg, on_tools=None):
    pending = None
    for mode, chunk in agent.stream(payload, cfg, stream_mode=["updates", "custom"]):
        if mode == "updates" and "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value
        elif mode == "updates" and "tools" in chunk and on_tools:
            on_tools()
    return pending


def find(log, kind, what):
    return [e for e in log if e["kind"] == kind and e["what"] == what]


# --- one run that hits every kind of entry ------------------------------------

db = Path(tempfile.mkdtemp()) / "checkpoints.sqlite"
inbox = SteerInbox()
agent = build_agent(ScriptedModel(responses=[
    call("query_signups", {"month": "2026-07"}, 1),  # before the ledger: blocked
    call("write_ledger", LEDGER, 2),
    call("query_signups", {"month": "2026-08"}, 3),
    call("task", {"description": "every country", "subagent_type": "general-purpose"}, 4),  # needs approval
    call("query_signups", {"month": "2026-10"}, 5),
    AIMessage("Done."),
]), TOOLS, inbox, review_every=2, checkpointer=sqlite_checkpointer(db))
cfg = {"configurable": {"thread_id": "audit"}}

assert run(agent, {"messages": [HumanMessage("trend?")]}, cfg)["type"] == "ledger"
sent = []
p = run(agent, Command(resume="skip July"), cfg,
        on_tools=lambda: sent or sent.append(inbox.put("referral only")))
assert p["type"] == "approval"
assert run(agent, Command(resume="no, do it inline"), cfg) is None

log = audit_log(agent.get_state(cfg).values)
blocked = find(log, "decision", "blocked")
assert blocked and blocked[0]["step"] == 1 and blocked[0]["tool"] == "query_signups", blocked
assert [e["what"] for e in log if e["kind"] == "ledger"] == ["written", "confirmed", "edit"]
(ledger_pause,) = find(log, "pause", "ledger")
assert ledger_pause["step"] == 2 and ledger_pause["answer"] == "skip July"
(steer,) = find(log, "steer", "steer")
assert steer["text"] == "referral only" and steer["status"] == "delivered" and steer["step"] == 3, steer
(approval,) = find(log, "pause", "approval")
assert approval["step"] == 4 and approval["answer"] == "no, do it inline" and "task (spawns a subagent" in approval["calls"][0]
assert find(log, "decision", "self_review")
assert all(isinstance(e["step"], int) and e["at"] for e in log), "every entry has a step and a time"
print("PASS one run logs the blocked call, the ledger and its correction, the steer, the approval and the self-review:")
print("\n".join(format_audit(log)))

# The ledger and approval nodes re-ran on resume; each pause is still logged once.
assert len(find(log, "pause", "ledger")) == 1 and len(find(log, "pause", "approval")) == 1
print("PASS a node that re-runs after a pause doesn't log it twice")

# It lives in the checkpoint, so a new process reads the same log.
again = build_agent(ScriptedModel(responses=[]), TOOLS, checkpointer=sqlite_checkpointer(db))
assert audit_log(again.get_state(cfg).values) == log
path = export_audit(again.get_state(cfg).values, "audit", directory=db.parent)
rows = [json.loads(line) for line in path.read_text().splitlines()]
assert rows == log, "the JSONL file holds the same entries"
print(f"PASS the log survives a restart and exports to {path.name}, one JSON object per line")

# --- step pause, budget pause, /stop, dropped messages ------------------------

inbox = SteerInbox(pause_every=1)
agent = build_agent(ScriptedModel(responses=[
    used(call("query_signups", {"month": "2026-07"}, 1), input=900),
    used(call("query_signups", {"month": "2026-08"}, 2), input=900),
    AIMessage("Done."),
]), TOOLS, inbox, guard=ToolGuardMiddleware(require_ledger=False), budget=Budget(run=1100),
    checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "pauses"}}
assert run(agent, {"messages": [HumanMessage("trend?")]}, cfg)["type"] == "step_pause"
assert run(agent, Command(resume={"action": "caveat", "text": "exclude bots"}), cfg)["type"] == "budget"
assert run(agent, Command(resume={"action": "raise", "tokens": 4000}), cfg) is None
log = audit_log(agent.get_state(cfg).values)
(sp,) = find(log, "pause", "step_pause")
assert sp["step"] == 1 and sp["answer"] == "caveat" and sp["text"] == "exclude bots"
(bp,) = find(log, "pause", "budget")
assert bp["step"] == 2 and bp["tokens"] == 4000 and bp["over"] == ["run 1,800/1,100"], bp
assert find(log, "decision", "budget_warning") and find(log, "ledger", "added")
print("PASS step pauses and budget pauses are logged with the answer you gave")

inbox = SteerInbox()
agent = build_agent(ScriptedModel(responses=[call("list_tables", {}, 1), AIMessage("after stop")]),
                    TOOLS, inbox, guard=ToolGuardMiddleware(require_ledger=False), checkpointer=InMemorySaver())
cfg = {"configurable": {"thread_id": "stop"}}
try:
    run(agent, {"messages": [HumanMessage("hi")]}, cfg, on_tools=lambda: inbox.put("/stop"))
    raise AssertionError("should have stopped")
except StopRun:
    pass
inbox.put("late note")
inbox.drop_pending()
seen = audit_log(agent.get_state(cfg).values, inbox)
assert [(e["what"], e["status"], e["step"]) for e in seen if e["kind"] == "steer"] == [
    ("interrupt", "delivered", 1), ("steer", "dropped", None)], seen
print("PASS a /stop and a dropped message are in the log even before the graph runs again")

inbox.stop.clear()
run(agent, None, cfg)  # resume: the next boundary writes them into the checkpointed log
saved = audit_log(agent.get_state(cfg).values)
assert len(find(saved, "steer", "interrupt")) == 1 and len(find(saved, "steer", "steer")) == 1
assert audit_log(agent.get_state(cfg).values, inbox) == saved, "no duplicates once they're saved"
print("PASS on resume they're saved into the run's log, once")

print("ALL-AUDIT-TESTS-OK")
