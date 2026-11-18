"""Offline tests with a scripted model: no API key, no tokens.

Covers check-in pause/resume, the steering inbox (modes and states), step
summaries, pause-every-N, /stop mid-reply and between steps, and self-review.
The Phase 1 gate is the first steering test: a steer lands before the next
model call and reports the step it landed on.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from example_tools import TOOLS
from scripted_model import SEEN, ScriptedModel, call
from steerable import (
    SelfReviewMiddleware,
    SteerInbox,
    StopOnRequest,
    StopRun,
    ToolGuardMiddleware,
    format_summary,
)
from steerable import build_agent as _build


def build_agent(model, tools, inbox=None, **kw):
    """Phase 1 tests: no ledger rule, in-memory checkpoints."""
    kw.setdefault("guard", ToolGuardMiddleware(require_ledger=False))
    return _build(model, tools, inbox, checkpointer=InMemorySaver(), **kw)

def run(agent, payload, cfg):
    """Stream like the terminal runner does; collect custom events and interrupts."""
    events, pending = [], None
    for mode, chunk in agent.stream(payload, cfg, stream_mode=["messages", "updates", "custom"]):
        if mode == "messages":
            continue
        if mode == "custom":
            events.append(chunk)
        elif "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value
    return events, pending


def saved(agent, cfg):
    return agent.get_state(cfg).values


# --- check_in pause/resume + steer gate ------------------------------------

script = [
    call("query_signups", {"month": "2026-08"}, 1),
    call("query_signups", {"month": "2026-09"}, 2),  # gap: no rows
    call("check_in", {
        "progress_so_far": "Aug referral signups 4,390.",
        "gaps_or_concerns": ["No rows for 2026-09"],
        "proposed_next_step": "Skip September",
        "options": ["Skip September", "Use Oct instead"],
    }, 3),
    AIMessage(content="Done: using October as instructed."),
]
inbox = SteerInbox()
agent = build_agent(ScriptedModel(responses=script), TOOLS, inbox, review_every=2)
cfg = {"configurable": {"thread_id": "t1"}}

pending = None
for mode, chunk in agent.stream({"messages": [HumanMessage("Signups trend?")]}, cfg, stream_mode=["updates", "custom"]):
    if mode == "updates" and "__interrupt__" in chunk:
        pending = chunk["__interrupt__"][0].value
    if mode == "updates" and "tools" in chunk and not inbox.items:  # typed while step 1's tool ran
        s = inbox.put("focus on referral only")
        assert s.status == "queued"

assert pending and pending["type"] == "check_in", pending
assert pending["gaps_or_concerns"] == ["No rows for 2026-09"]
print("PASS check_in paused the run with the gap:", pending["gaps_or_concerns"])
assert agent.get_state(cfg).next, "run should be paused, not finished"
print("PASS checkpoint saved, paused at:", agent.get_state(cfg).next)

# GATE: the steer landed before the next model call and records which step.
steer = inbox.items[0]
assert steer.status == "delivered" and steer.step == 2, steer
got = [m for m in SEEN[1] if "focus on referral" in str(m.content)]
assert got, "steering note never reached step 2's model call"
assert got[0] is SEEN[1][-1] or "focus on referral" in str(SEEN[1][-2].content), "note must be appended at the end"
print(f"PASS GATE steer landed before the next model call: {steer.label()}")

review = [m for m in SEEN[2] if "Self-review checkpoint" in str(m.content)]
assert review, "self-review prompt missing after 2 tool calls"
kept = [m for m in saved(agent, cfg)["messages"] if "Self-review checkpoint" in str(m.content)]
assert len(kept) == 1 and kept[0].additional_kwargs.get("harness") == "note", "saved once, as a harness note"
print("PASS self-review prompt sent after 2 tool calls and saved as a note, so the cached prefix holds")

for _ in agent.stream(Command(resume="Use Oct instead"), cfg, stream_mode="updates"):
    pass
msgs = saved(agent, cfg)["messages"]
assert "Use Oct instead" in next(m for m in msgs if isinstance(m, ToolMessage) and m.name == "check_in").content
assert msgs[-1].content == "Done: using October as instructed."
print("PASS resumed from checkpoint; agent received the answer")

# --- step summaries (S10, S11) ---------------------------------------------

SEEN.clear()
agent = build_agent(ScriptedModel(responses=[
    call("query_signups", {"month": "2026-08"}, 1),
    call("query_signups", {"month": "2026-10"}, 2),
    AIMessage("Signups rose."),
]), TOOLS, SteerInbox())
cfg = {"configurable": {"thread_id": "t-sum"}}
events, _ = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
sums = [e for e in events if e["type"] == "step_summary"]
assert [s["step"] for s in sums] == [1, 2], sums
assert sums[0]["did"] == "query_signups" and sums[0]["results"] == 1 and sums[0]["output_chars"] > 0
assert len(SEEN) == 3, "summaries must not add model calls"
print("PASS a summary after every tool step, built from state with no extra model call:")
print("     " + "\n     ".join(format_summary(sums[0])))

# --- pause every N steps + caveat (S12, S13) -------------------------------

inbox = SteerInbox(pause_every=1)
agent = build_agent(ScriptedModel(responses=[
    call("query_signups", {"month": "2026-08"}, 1),
    call("query_signups", {"month": "2026-10"}, 2),
    AIMessage("Done."),
]), TOOLS, inbox)
cfg = {"configurable": {"thread_id": "t-pause"}}
_, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
assert p and p["type"] == "step_pause" and p["step"] == 1, p
_, p = run(agent, Command(resume={"action": "caveat", "text": "exclude bots"}), cfg)
assert p and p["type"] == "step_pause" and p["step"] == 2, "should pause again after step 2"
state = saved(agent, cfg)
from steerable import current_ledger
assert current_ledger(state["ledger_ops"])[0]["constraints"] == ["exclude bots"], "a caveat is a new ledger constraint"
assert any("Caveat from the user after reviewing step 1" in str(m.content) for m in state["messages"])
_, p = run(agent, Command(resume={"action": "interrupt"}), cfg)
assert p is None and saved(agent, cfg)["messages"][-1].additional_kwargs.get("harness") == "stopped"
print("PASS pause-every-N stops at each boundary; a caveat lands as a note + recorded constraint; Interrupt ends the turn")

# --- message modes (S4) ----------------------------------------------------

box = SteerInbox()
box.put("/collect also split by week")
assert box.take(3) == [], "a lone collect waits"
box.put("use Oct only")
got = box.take(4)
assert [s.mode for s in got] == ["collect", "steer"], "a steer flushes collects, steer listed last"
box.put("/follow then email me the table")
box.put("/collect one more thing")
assert box.after_turn(9) == ["one more thing", "then email me the table"]
box.put("late")
assert [s.status for s in box.drop_pending()] == ["dropped"]
print("PASS modes: collect waits, steer flushes it and wins, follow-ups run after the task, drop marks dropped")

# --- /stop mid-reply and between steps (S6, T5) ----------------------------

inbox = SteerInbox()
model = ScriptedModel(responses=[AIMessage("word " * 50), AIMessage("Resumed answer.")])
model.on_token = lambda i: i == 3 and inbox.put("/stop")
agent = build_agent(model, TOOLS, inbox)
cfg = {"configurable": {"thread_id": "t-stop"}, "callbacks": [StopOnRequest(inbox)]}
try:
    run(agent, {"messages": [HumanMessage("hi")]}, cfg)
    raise AssertionError("should have been cancelled")
except StopRun:
    pass
assert not [m for m in saved(agent, cfg)["messages"] if isinstance(m, AIMessage)], "cancelled reply must not be saved"
inbox.stop.clear()
run(agent, None, cfg)  # Resume: re-run the cancelled step from the checkpoint
assert saved(agent, cfg)["messages"][-1].content == "Resumed answer."
print("PASS /stop cancels mid-reply after 4 of 50 words, keeps the checkpoint, Resume re-runs the step")

inbox = SteerInbox()
agent = build_agent(ScriptedModel(responses=[call("list_tables", {}, 9), AIMessage("after edit")]), TOOLS, inbox)
cfg = {"configurable": {"thread_id": "t-stop2"}}
try:
    for i, _ in enumerate(agent.stream({"messages": [HumanMessage("hi")]}, cfg, stream_mode="updates")):
        if i == 0:
            inbox.put("/stop")
    raise AssertionError("should have been cancelled")
except StopRun:
    pass
assert inbox.items[0].status == "delivered"
inbox.stop.clear()
from steerable import note
run(agent, {"messages": [note("[Note from the user]\n- only list tables")]}, cfg)  # Resume with edit
assert saved(agent, cfg)["messages"][-1].content == "after edit"
print("PASS /stop between steps cancels before the next model call; Resume with edit continues the thread")

# --- self-review batch logic -----------------------------------------------


def tools(k):
    return [ToolMessage(content="x", tool_call_id=f"t{i}", name="query_signups") for i in range(k)]


mw = SelfReviewMiddleware(every=4)
h = [HumanMessage("q")]
assert not mw.due(h + [AIMessage("")] + tools(3)), "3 results: not due"
assert mw.due(h + [AIMessage("")] + tools(3) + [AIMessage("")] + tools(2)), "3 -> 5 crosses 4"
assert not mw.due(h + [AIMessage("")] + tools(5) + [AIMessage("")] + tools(1)), "5 -> 6 does not"
assert mw.due(h + [AIMessage("")] + tools(8)), "0 -> 8 in one batch"
print("PASS self-review catches parallel batches that skip past a multiple")

# --- check_in accepts one string for gaps (seen live: a model sent a string, the call failed) ---

agent = build_agent(ScriptedModel(responses=[
    call("check_in", {"progress_so_far": "8 countries pulled", "gaps_or_concerns": "NG Oct looks fake",
                      "proposed_next_step": "exclude NG Oct"}, 1),
    AIMessage("ok"),
]), TOOLS, SteerInbox())
cfg = {"configurable": {"thread_id": "t-str"}}
_, p = run(agent, {"messages": [HumanMessage("rank")]}, cfg)
assert p and p["type"] == "check_in" and p["gaps_or_concerns"] == ["NG Oct looks fake"], p
print("PASS a check_in with gaps as one string pauses normally instead of failing")

agent = build_agent(ScriptedModel(responses=[
    call("check_in", {"progress_so_far": "x", "gaps_or_concerns": '["NG Oct looks fake", "Aug and Sep not pulled"]',
                      "proposed_next_step": "y"}, 1),
    AIMessage("ok"),
]), TOOLS, SteerInbox())
_, p = run(agent, {"messages": [HumanMessage("rank")]}, {"configurable": {"thread_id": "t-json"}})
assert p["gaps_or_concerns"] == ["NG Oct looks fake", "Aug and Sep not pulled"], p
print("PASS a check_in with gaps as a JSON list in a string is split back into separate gaps")
