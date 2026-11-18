"""Offline tests for Phase 4: model routing and free fallback (R1 to R10, T9).

Each tier is a scripted model that answers or raises the error a real API would
(429 rate limit, 400 "credit balance is too low", 529 overloaded). The R9 test
builds the real Anthropic and OpenRouter request payloads without sending them.
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from example_tools import TOOLS
from scripted_model import ScriptedModel, call, used
from steerable import (OPENROUTER_API, Budget, ModelRouterMiddleware, SteerInbox, ToolGuardMiddleware, audit_log,
                       build_agent, format_summary)


class APIError(Exception):
    """Shaped like the SDKs' errors: a message and a status_code."""

    def __init__(self, status, text):
        super().__init__(text)
        self.status_code = status


CREDITS = APIError(400, "Your credit balance is too low to access the Anthropic API.")
RATE = APIError(429, "rate_limit_error")
BUSY = APIError(529, "overloaded_error")


class Down(ScriptedModel):
    """A model whose every call fails with `error`."""

    error: object = None
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        raise self.error


def tier(name, *replies):
    return ScriptedModel(name=name, responses=list(replies))


def build(primary, router, budget=None, **kw):
    kw.setdefault("guard", ToolGuardMiddleware(require_ledger=False))
    return build_agent(primary, TOOLS, SteerInbox(), budget=budget, router=router, mode="quick",
                       checkpointer=InMemorySaver(), **kw)


def run(agent, payload, cfg):
    events, pending = [], None
    for mode, chunk in agent.stream(payload, cfg, stream_mode=["updates", "custom"]):
        if mode == "custom":
            events.append(chunk)
        elif "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0].value
    return events, pending


def of(events, kind):
    return [e for e in events if e["type"] == kind]


q = lambda month, i: call("query_signups", {"month": month}, i)  # noqa: E731
HAIKU, FREE = "claude-haiku-5-5", "meta-llama/llama-3.3-70b-instruct:free"

# --- R3, R7: out of credits → the cheap tier, told once, for the rest of the task ---

claude = Down(responses=[], name="claude-sonnet-5-5", error=CREDITS)
cheap = tier(HAIKU, used(q("2026-07", 1), input=1000, output=50), used(q("2026-08", 2), input=1100, output=50),
             AIMessage("Done."), AIMessage("Done again."))
agent = build(claude, ModelRouterMiddleware(cheap, check_tools=False))
cfg = {"configurable": {"thread_id": "credits"}}
events, _ = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
(switch,) = of(events, "model_switched")
assert switch["step"] == 1 and switch["to"] == HAIKU and "out of credits" in switch["why"], switch
assert claude.calls == 1, "after the switch, later steps don't retry the model that ran out"
panels = of(events, "step_summary")
assert [p["model"] for p in panels] == [HAIKU, HAIKU] and panels[0]["cost_step"] > 0, panels
assert format_summary(panels[0])[0].startswith(f"── step 1 · {HAIKU} · $"), "every panel names the model"
v = agent.get_state(cfg).values
assert v["route"]["tier"] == 1 and v["messages"][-1].content == "Done."
assert [e["what"] for e in audit_log(v) if e["what"] == "model_switched"] == ["model_switched"]
print("PASS GATE out of credits: the run moves to the cheap model at step 1, says so once, and stays there:")
print(f"     ⚠ step 1: switched from {switch['from']} to {switch['to']} ({switch['why']})")
print("     " + format_summary(panels[0])[0])

events, _ = run(agent, {"messages": [HumanMessage("and July?")]}, cfg)
assert claude.calls == 2 and len(of(events, "model_switched")) == 1, "a new question tries the main model again"
print("PASS a new question starts on the main model again, which is how it comes back once credits return")

# --- R3, R10: a chain of failures, then every tier down -------------------------

claude, cheap = Down(responses=[], error=RATE), Down(responses=[], name=HAIKU, error=BUSY)
free = tier(FREE, used(q("2026-07", 1), input=4000, output=80), AIMessage("Done on the free model."))
agent = build(claude, ModelRouterMiddleware(cheap, [free], check_tools=False))
cfg = {"configurable": {"thread_id": "chain"}}
events, _ = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
(switch,) = of(events, "model_switched")
assert switch["to"] == FREE and "rate limited" in switch["why"] and "overloaded" in switch["why"], switch
assert claude.calls == cheap.calls == 1, "each tier is tried once per call"
(panel,) = of(events, "step_summary")
assert panel["cost_step"] == 0.0 and panel["free_tokens"] == 4080, panel
print("PASS rate limited, then overloaded: it falls through to the free model, which costs $0")

claude, cheap, free = (Down(responses=[], error=e, name=n) for e, n in ((CREDITS, None), (RATE, HAIKU), (RATE, FREE)))
agent = build(claude, ModelRouterMiddleware(cheap, [free], check_tools=False))
cfg = {"configurable": {"thread_id": "all-down"}}
events, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
last = agent.get_state(cfg).values["messages"][-1]
assert p is None and last.content.startswith("Stopped: no model could take this step"), last.content
assert claude.calls == cheap.calls == free.calls == 1, "no retry loop that burns the free daily allowance"
assert [e["what"] for e in audit_log(agent.get_state(cfg).values)] == ["models_exhausted"]
print("PASS when every tier fails, each is tried once and the run stops with a message instead of looping")

claude = Down(responses=[], error=ValueError("bad tool schema"))
cheap = tier(HAIKU, AIMessage("should not run"))
agent = build(claude, ModelRouterMiddleware(cheap, check_tools=False))
try:
    run(agent, {"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": "bug"}})
    raise AssertionError("a bug must not be hidden by a fallback")
except ValueError:
    pass
print("PASS any other error is raised as it is, not papered over with a weaker model")

# --- R4: the run budget nearly spent → the free tier before the call -------------

claude = tier("claude-sonnet-5-5", used(q("2026-07", 1), input=950), AIMessage("unused"))
free = tier(FREE, used(q("2026-08", 2), input=5000, output=100), AIMessage("Done."))
agent = build(claude, ModelRouterMiddleware(cheap=None, free=[free], check_tools=False), budget=Budget(run=1000))
cfg = {"configurable": {"thread_id": "budget"}}
events, p = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
(switch,) = of(events, "model_switched")
assert switch["step"] == 2 and switch["to"] == FREE and "run budget at 95%" in switch["why"], switch
assert p is None, "free tokens don't count against the budget, so it never pauses"
meter = format_summary(of(events, "step_summary")[1])
assert "950 tokens of 1,000 budget (95%)" in meter[2] and "5,100 on free models" in meter[2], meter
print("PASS at 90% of the run budget the router moves to the free model before the call; no budget pause:")
print("     " + meter[2].strip())

# --- T9: a step that only listed tables goes to the cheap model ------------------

claude = tier("claude-sonnet-5-5", call("list_tables", {}, 1), AIMessage("Done."))
cheap = tier(HAIKU, q("2026-07", 2))
agent = build(claude, ModelRouterMiddleware(cheap, routine_tools=("list_tables",), check_tools=False))
cfg = {"configurable": {"thread_id": "routine"}}
events, _ = run(agent, {"messages": [HumanMessage("trend?")]}, cfg)
assert not of(events, "model_switched") and "route" not in agent.get_state(cfg).values
msgs = [m for m in agent.get_state(cfg).values["messages"] if isinstance(m, AIMessage)]
assert [m.response_metadata.get("harness_route", {}).get("model") for m in msgs] == [None, HAIKU, None]
assert "(routine step)" in format_summary(of(events, "step_summary")[1])[0]
print("PASS after a list_tables-only step the next call goes to the cheap model, then back to Claude, with no switch notice")

# --- R6: models without tool calling are dropped at startup ---------------------

good = ChatOpenAI(model="a/good:free", base_url=OPENROUTER_API, api_key="offline")
bad = ChatOpenAI(model="a/no-tools:free", base_url=OPENROUTER_API, api_key="offline")
r = ModelRouterMiddleware(FakeListChatModel(responses=["x"]), [bad, good], tool_models={"a/good:free"})
assert r.tiers() == ["the agent's model", "a/good:free"] and r.cheap is None, r.tiers()
assert any("a/no-tools:free" in d for d in r.dropped) and any("no tool calling" in d for d in r.dropped), r.dropped
print("PASS models with no tool calling are dropped before the run:", "; ".join(r.dropped))

# --- R8: a free model asks before it writes --------------------------------------

claude = Down(responses=[], error=CREDITS)
free = tier(FREE, call("write_file", {"file_path": "/report.md", "content": "hi"}, 1), AIMessage("Done."))
agent = build(claude, ModelRouterMiddleware(free=[free], check_tools=False))
cfg = {"configurable": {"thread_id": "write"}}
_, p = run(agent, {"messages": [HumanMessage("write it up")]}, cfg)
assert p["type"] == "approval" and "a free model" in p["calls"][0]["reasons"][0], p
_, p = run(agent, Command(resume="no"), cfg)
assert p is None and "/report.md" not in (agent.get_state(cfg).values.get("files") or {})
print("PASS on a free model, write_file pauses for approval; declining it writes nothing")

# --- R9: no Anthropic cache markers reach the other provider ---------------------


class CaptureAnthropic(ChatAnthropic):
    replies: list = Field(default_factory=list)
    payloads: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.payloads.append(self._get_request_payload(messages, stop=stop, **kw))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


class CaptureOpenRouter(ChatOpenAI):
    replies: list = Field(default_factory=list)
    payloads: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.payloads.append(self._get_request_payload(messages, stop=stop, **kw))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


first = AIMessage(content=[{"type": "text", "text": "Checking July."},
                           {"type": "tool_use", "id": "c1", "name": "query_signups", "input": {"month": "2026-07"}}],
                  tool_calls=[{"name": "query_signups", "args": {"month": "2026-07"}, "id": "c1"}])
claude = CaptureAnthropic(model="claude-sonnet-5-5", api_key="offline", disable_streaming=True,
                          replies=[used(first, input=10, cache_write=6000, output=40), CREDITS])
free = CaptureOpenRouter(model=FREE, base_url=OPENROUTER_API, api_key="offline", disable_streaming=True,
                         replies=[AIMessage("July had 120 signups.")])
agent = build(claude, ModelRouterMiddleware(free=[free], check_tools=False))
agent.invoke({"messages": [HumanMessage("July signups?")]}, {"configurable": {"thread_id": "markers"}})
assert all("cache_control" in json.dumps(p) for p in claude.payloads), "Claude's requests keep their breakpoints"
(sent,) = free.payloads
assert "cache_control" not in json.dumps(sent), "no Anthropic markers in the OpenRouter request"
roles = [m["role"] for m in sent["messages"]]
assert roles[-3:] == ["user", "assistant", "tool"] and sent["messages"][-2]["tool_calls"], roles
assert {t["function"]["name"] for t in sent["tools"]} >= {"query_signups"}
print("PASS after a switch mid-run, the OpenRouter request carries the history and tools with no cache markers")

print("ALL-ROUTER-TESTS-OK")
