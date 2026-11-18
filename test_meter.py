"""Offline tests for Phase 3: token meter, budgets, cache-friendly requests.

The Phase 3 gate is the first test: the meter shows cache reads, and a budget
pauses the run at 100%. The cache tests build the exact request payloads a real
ChatAnthropic would send (no network, no key), so they check the bytes the
prompt cache sees.
"""

import json

from langchain.agents.middleware import AgentMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from example_tools import TOOLS
from scripted_model import BOUND, ScriptedModel, call, used
from steerable import (Budget, SteerInbox, ToolGuardMiddleware, build_agent, format_summary, run_cost,
                       token_table, token_totals)


def build(model, inbox=None, **kw):
    kw.setdefault("guard", ToolGuardMiddleware(require_ledger=False))
    return build_agent(model, TOOLS, inbox, checkpointer=InMemorySaver(), **kw)


def run(agent, payload, cfg):
    events, pending = [], None
    for mode, chunk in agent.stream(payload, cfg, stream_mode=["messages", "updates", "custom"]):
        if mode == "custom":
            events.append(chunk)
        elif mode == "updates" and "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value
    return events, pending


def panels(events):
    return {e["step"]: e for e in events if e["type"] == "step_summary"}


# --- GATE: the meter shows cache reads; a budget pauses the run at 100% (T6, T8) ---

steps = [
    used(call("query_signups", {"month": "2026-07"}, 1), input=1000, cache_write=3000, output=100),  # 4,100
    used(call("query_signups", {"month": "2026-08"}, 2), input=200, cache_read=3000, cache_write=300, output=100),  # +600
    used(call("query_signups", {"month": "2026-10"}, 3), input=200, cache_read=3300, cache_write=200, output=100),  # +500
    used(AIMessage("Done."), input=100, cache_read=3500, output=50),
]
agent = build(ScriptedModel(responses=steps), budget=Budget(run=5000))
cfg = {"configurable": {"thread_id": "gate"}}
events, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
seen = panels(events)
assert seen[2]["step_tokens"] == {"input": 200, "cache_read": 3000, "cache_write": 300, "cache_write_1h": 0, "output": 100}
meter = "\n".join(format_summary(seen[2]))
assert "cache read 3,000" in meter and "4,700 tokens of 5,000 budget (94%)" in meter, meter
assert "$" in meter, "priced from the model name in the API response"
assert any("82% of its 5,000-token budget" in w for w in seen[1]["warnings"]), "warns when it crosses 80%"
assert not seen[2]["warnings"], "warns once, not on every step after"
assert p and p["type"] == "budget" and p["step"] == 3, "pauses at the step that reaches 100%, not before"
assert p["over"] == [{"kind": "run", "used": 5200, "limit": 5000}], p["over"]
print("PASS GATE the meter shows cache reads, warns at 80%, and the budget pauses the run at 100%:")
print("     " + "\n     ".join(format_summary(seen[2])[:3]))

_, p = run(agent, Command(resume={"action": "continue"}), cfg)
v = agent.get_state(cfg).values
assert p is None and v["messages"][-1].content == "Done." and v["budget_limits"]["run"] == 10_200
print("PASS Continue adds one more run budget (5,200 used + 5,000) and the run finishes")

table = token_table(v["messages"])
assert len(table) == 6 and "3,000" in table[2] and table[-1].split()[:5] == ["all", "1,500", "9,800", "3,500", "350"], table
print("PASS /tokens shows input, cache read, cache write and output per step:")
print("\n".join("  " + r for r in table))

# A new question starts a new run: the budget and the raised limit both reset.
agent2_steps = [used(call("query_signups", {"month": "2026-07"}, 9), input=3900), AIMessage("Again.")]
agent = build(ScriptedModel(responses=steps[:3] + agent2_steps), budget=Budget(run=5000))
cfg = {"configurable": {"thread_id": "reset"}}
run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
run(agent, Command(resume={"action": "stop"}), cfg)
assert agent.get_state(cfg).values["messages"][-1].additional_kwargs.get("harness") == "stopped"
events, p = run(agent, {"messages": [HumanMessage("and July only?")]}, cfg)
(new,) = panels(events).values()
assert p is None and new["run_limit"] == 5000 and not new["warnings"], "new run, fresh budget"
print("PASS Stop at a budget pause ends the run; a new question gets a fresh budget")

# --- per-step budget; a typed number raises it ---------------------------------

agent = build(ScriptedModel(responses=[
    used(call("query_signups", {"month": "2026-07"}, 1), input=900),
    used(call("query_signups", {"month": "2026-08"}, 2), input=1500),
    used(call("query_signups", {"month": "2026-09"}, 3), input=1800),
    AIMessage("Done."),
]), budget=Budget(step=1000))
cfg = {"configurable": {"thread_id": "step"}}
events, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
assert any("90% of the 1,000-token step budget" in w for w in panels(events)[1]["warnings"])
assert p["type"] == "budget" and p["over"][0]["kind"] == "step" and p["step"] == 2, p
_, p = run(agent, Command(resume={"action": "raise", "tokens": 2000}), cfg)
assert p is None and agent.get_state(cfg).values["budget_limits"]["step"] == 2000, "1,800 < the new 2,000"
print("PASS a step over the step budget pauses; typing a number raises it for the rest of the run")

# --- T4: thinking settings are pinned for the run -------------------------------


class FlipThinking(AgentMiddleware):
    """Stands in for anything that would change thinking mid-run."""

    def wrap_model_call(self, request, handler):
        if any(isinstance(m, AIMessage) for m in request.messages):
            request = request.override(model_settings={**request.model_settings,
                                                       "thinking": {"type": "enabled", "budget_tokens": 2000}})
        return handler(request)


BOUND.clear()
agent = build(ScriptedModel(responses=[call("query_signups", {"month": "2026-07"}, 1), AIMessage("Done.")]),
              middleware=[FlipThinking()])
cfg = {"configurable": {"thread_id": "pin"}}
events, _ = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
assert "thinking" not in BOUND[0] and "thinking" not in BOUND[1], BOUND
assert any(e["type"] == "thinking_pinned" for e in events)
assert agent.get_state(cfg).values["thinking_pins"] == {"ScriptedModel": None}
print("PASS a mid-run change to thinking is reverted to the first call's settings, and the live view says so")

# --- T1, T2: the real request payloads are cache-friendly -----------------------


class CaptureAnthropic(ChatAnthropic):
    """A real ChatAnthropic that records the request it would send instead of sending it."""

    replies: list = Field(default_factory=list)
    payloads: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.payloads.append(self._get_request_payload(messages, stop=stop, **kw))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def flat(payload):
    """Every content block in order, as JSON, without cache markers."""
    out = []
    for m in payload["messages"]:
        content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
        for b in content:
            out.append(m["role"] + json.dumps({k: v for k, v in b.items() if k != "cache_control"}, sort_keys=True))
    return out


model = CaptureAnthropic(model="claude-sonnet-5-5", api_key="offline", disable_streaming=True, replies=[
    # usage as a streamed reply reports it: a cache-write total with no 5-minute / 1-hour split
    used(call("query_signups", {"month": "2026-07"}, 1), input=10, cache_write=7000, output=50),
    used(call("query_signups", {"month": "2026-08"}, 2), input=5, cache_read=7000, cache_write=400, output=50),
    used(call("query_signups", {"month": "2026-10"}, 3), input=5, cache_read=7400, cache_write=300, output=50),
    used(AIMessage("Done."), input=5, cache_read=7700, output=20),
])
inbox = SteerInbox()
agent = build(model, inbox, review_every=2, cache_ttl="1h")
cfg = {"configurable": {"thread_id": "cache"}}
for chunk in agent.stream({"messages": [HumanMessage("trend?")]}, cfg, stream_mode="updates"):
    if "tools" in chunk and not inbox.items:
        inbox.put("referral only")  # a steer and, at step 3, a self-review prompt land mid-run

P = model.payloads
assert len(P) == 4
one_hour = {"type": "ephemeral", "ttl": "1h"}
for p in P:
    assert p["cache_control"] == one_hour, "breakpoint on the latest message, 1-hour lifetime"
    assert p["system"][-1]["cache_control"] == one_hour, "breakpoint after the system prompt"
    assert p["tools"][-1]["cache_control"] == one_hour, "breakpoint after the tool definitions"
    marks = sum("cache_control" in b for m in p["messages"] if isinstance(m["content"], list) for b in m["content"])
    assert marks + 3 <= 4, "at most 4 breakpoints per request"
print("PASS every request has 1-hour breakpoints after the system prompt, the tools, and the latest message")

for a, b in zip(P, P[1:]):
    assert flat(b)[:len(flat(a))] == flat(a), "each request must start with the previous one, byte for byte"
assert any("referral only" in x for x in flat(P[1])) and any("Self-review" in x for x in flat(P[2]))
print("PASS with a steer and a self-review prompt mid-run, every request extends the previous one, so the cache prefix holds")

t = token_totals(agent.get_state(cfg).values["messages"])
assert t["cache_write"] == t["cache_write_1h"] == 7700, t
assert abs(run_cost(agent.get_state(cfg).values["messages"]) - (25 * 2 + 22100 * 0.1 + 7700 * 4 + 170 * 10) / 1e6) < 1e-9
print("PASS cache writes with no lifetime split are counted as 1-hour writes and priced at $4 per million, not $2.50")

import inspect

assert inspect.signature(build_agent).parameters["cache_ttl"].default == "5m"
model = CaptureAnthropic(model="claude-sonnet-5-5", api_key="offline", disable_streaming=True,
                         replies=[used(AIMessage("hi"), input=5, cache_write=6000, output=3)])
agent = build(model)
cfg = {"configurable": {"thread_id": "default-ttl"}}
agent.invoke({"messages": [HumanMessage("hi")]}, cfg)
assert model.payloads[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
t = token_totals(agent.get_state(cfg).values["messages"])
assert t["cache_write"] == 6000 and t["cache_write_1h"] == 0, "default writes are 5-minute and priced at $2.50"
print("PASS the default cache is 5 minutes; CACHE_TTL=1h / cache_ttl='1h' opts into the 1-hour cache")

print("ALL-METER-TESTS-OK")
