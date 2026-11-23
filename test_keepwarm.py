"""Offline tests: the cache keep-warm that runs while a pause waits for you. No network."""

import time
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import steerable
from scripted_model import BOUND, ScriptedModel, used
from steerable import KeepWarm, build_agent

claude = ChatAnthropic(model="claude-sonnet-5-5", api_key="sk-ant-test")


class Req(SimpleNamespace):
    def override(self, **kw):
        return Req(**{**vars(self), **kw})


def request(model=claude):
    return Req(model=model, model_settings={"cache_control": {"type": "ephemeral", "ttl": "5m"}}, messages=[HumanMessage("q")])


def reply():  # what a 1-token refresh returns: the whole prompt read from cache
    return ModelResponse(result=[used(AIMessage("x"), input=5, cache_read=50_000, output=1)])


# --- a refresh replays the last request with max_tokens=1 and counts what it cost --
seen = []
w = KeepWarm(max_pings=3, interval=0.05)
w.remember(request(), lambda r: seen.append(r) or reply())
w.ping()
assert seen[0].model_settings == {"cache_control": {"type": "ephemeral", "ttl": "5m"}, "max_tokens": 1}
assert abs(w.spent - (5 * 2 + 50_000 * 0.10 + 1 * 10) / 1e6) < 1e-9, w.spent
print("PASS a refresh replays the last request with max_tokens=1; 50,000 cached tokens cost about $0.005")

# --- it refreshes on a timer while you're away, stops at the cap, and reports ----
w = KeepWarm(max_pings=2, interval=0.05)
w.remember(request(), lambda r: reply())
w.start(); time.sleep(0.4)
msg = w.stop()
assert w.pings == 2 and "2 refreshes" in msg and "$0.010" in msg, (w.pings, msg)
print("PASS refreshes run every interval, stop at the cap, and the user is told what they cost:", msg)

# --- answering before the first interval costs nothing and says nothing ----------
w = KeepWarm(max_pings=6, interval=60)
w.remember(request(), lambda r: reply())
w.start(); assert w.stop() is None and w.pings == 0
print("PASS answering before the first refresh is due costs nothing and prints nothing")

# --- a failing refresh stops the loop, never raises, and is reported -------------
def boom(r):
    raise RuntimeError("max_tokens must be greater than thinking.budget_tokens")
w = KeepWarm(max_pings=6, interval=0.05)
w.remember(request(), boom)
w.start(); time.sleep(0.3)
msg = w.stop()
assert w.pings == 0 and "keep-alive stopped" in msg and "thinking.budget_tokens" in msg, msg
print("PASS a refresh that errors stops the loop and says why, without breaking the run")

# --- only Anthropic requests are remembered; off means off -----------------------
w = KeepWarm(max_pings=6, interval=0.05)
w.remember(request(model=ScriptedModel(responses=[])), lambda r: reply())
w.start(); assert w.last is None and w.stop() is None
w = KeepWarm(max_pings=0, interval=0.05)
w.remember(request(), lambda r: reply()); w.start(); assert w._thread is None
print("PASS non-Anthropic models are never refreshed, and max_pings=0 turns it off")

# --- wired into the agent: the middleware sees real requests, and a replay reaches the model ---
steerable.ChatAnthropic = ScriptedModel  # let the scripted model stand in for Claude
try:
    BOUND.clear()
    inbox = steerable.SteerInbox()
    model = ScriptedModel(responses=[AIMessage("first answer"), AIMessage("x")])
    agent = build_agent(model, [], inbox=inbox, checkpointer=InMemorySaver(), mode="quick", keep_warm=2)
    agent.invoke({"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": "kw"}})
    warm = inbox.warm
    assert warm.last is not None, "the middleware never saw the request"
    n = len(BOUND)
    warm.ping()  # what the background loop does while a pause waits
    assert len(BOUND) == n + 1 and BOUND[-1].get("max_tokens") == 1, BOUND[-1]
    assert warm.pings == 1
finally:
    steerable.ChatAnthropic = ChatAnthropic
print("PASS inside a real agent, the middleware hands over each request and a refresh reaches the model with max_tokens=1")

print("ALL-KEEPWARM-TESTS-OK")
