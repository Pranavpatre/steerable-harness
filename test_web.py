"""Offline tests: Anthropic web search and fetch (server tools) in the harness. No network."""

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from scripted_model import TOOLS_SEEN, ScriptedModel, used
from steerable import (WEB_TOOLS, PauseTurnMiddleware, build_agent, cost, step_summary, usage,
                       without_server_tools)

SEARCH = [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "premier league 2024/25 table"}},
          {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
          {"type": "text", "text": "Liverpool won it."}]

# --- the tools reach the model only when web is on --------------------------
for web in (False, True):
    TOOLS_SEEN.clear()
    agent = build_agent(ScriptedModel(responses=[AIMessage("done")]), [], checkpointer=InMemorySaver(), mode="quick", web=web)
    agent.invoke({"messages": [HumanMessage("hi")]}, {"configurable": {"thread_id": f"w{web}"}})
    assert ({"web_search", "web_fetch"} <= set(TOOLS_SEEN[0])) == web, TOOLS_SEEN[0]
print("PASS web search and fetch are offered to the model only with web=True")

# --- meter: a search is counted and priced, and shows in the step panel ------
msg = used(AIMessage(SEARCH), input=15000, output=100)
u = usage(msg)
assert u["web_searches"] == 1 and abs(cost(u, "claude-sonnet-5-5") - (15000 * 2 + 100 * 10) / 1e6 - 0.01) < 1e-9
s = step_summary({"messages": [HumanMessage("q"), msg], "todos": []})
assert "web_search (on Anthropic's side)" in s["did"], s["did"]
print("PASS a web search is priced at $0.01 on top of its tokens, and the step panel names it")

# --- a non-Anthropic tier never sees the web tools or web blocks ------------
req = ModelRequest(model=None, messages=[HumanMessage("q"), AIMessage(SEARCH)], tools=[*WEB_TOOLS], system_message=None)
out = without_server_tools(req)
assert out.tools == [] and out.messages[1].content == "[web_search: {'query': 'premier league 2024/25 table'}]\nLiverpool won it."
assert without_server_tools(ModelRequest(model=None, messages=[HumanMessage("q")], tools=[], system_message=None)).tools == []
print("PASS for a non-Anthropic tier, web tools are dropped and past searches become plain text")

# --- pause_turn: the paused reply is sent back and the two parts are merged --
paused = AIMessage(SEARCH[:2], response_metadata={"stop_reason": "pause_turn"},
                   usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
final = AIMessage([SEARCH[2]], response_metadata={"stop_reason": "end_turn"},
                  usage_metadata={"input_tokens": 200, "output_tokens": 20, "total_tokens": 220})
calls = []
def handler(r):
    calls.append(r)
    return ModelResponse(result=[paused if len(calls) == 1 else final])
got = PauseTurnMiddleware().wrap_model_call(ModelRequest(model=None, messages=[HumanMessage("q")], tools=[], system_message=None), handler)
m = got.result[0]
assert len(calls) == 2 and calls[1].messages[-1] is paused, "the paused reply goes back so the API carries on"
assert [b["type"] for b in m.content] == ["server_tool_use", "web_search_tool_result", "text"]
assert m.usage_metadata["input_tokens"] == 300 and m.usage_metadata["output_tokens"] == 30
print("PASS a pause_turn reply is resumed, and the two parts come back as one reply with summed tokens")

print("ALL-WEB-TESTS-OK")
