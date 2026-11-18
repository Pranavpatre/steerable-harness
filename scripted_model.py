"""A scripted chat model for offline tests: fixed replies, real token callbacks."""

import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

SEEN: list[list] = []  # what the model received on each call
BOUND: list[dict] = []  # the settings each call was bound with (thinking, cache_control, ...)
TOOLS_SEEN: list[list] = []  # the tool names each call was given


def used(msg, input=0, cache_read=0, cache_write=0, output=0, model="claude-sonnet-5-5"):
    """Attach API-style usage to a scripted reply. input_tokens includes cached ones, as in LangChain."""
    msg.usage_metadata = {
        "input_tokens": input + cache_read + cache_write, "output_tokens": output,
        "total_tokens": input + cache_read + cache_write + output,
        "input_token_details": {"cache_read": cache_read, "cache_creation": cache_write},
    }
    msg.response_metadata = {"model_name": model}
    return msg


class ScriptedModel(FakeMessagesListChatModel):
    """Replies from a fixed script. Text replies stream word by word, so the
    token callbacks (live view, /stop mid-reply) fire like a real model."""

    on_token: object = None  # optional hook called after each streamed word

    def bind_tools(self, tools, **kwargs):
        BOUND.append({k: v for k, v in kwargs.items() if k != "tool_choice"})
        TOOLS_SEEN.append([getattr(t, "name", None) or t.get("name") for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        SEEN.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._generate(messages, stop, run_manager, **kwargs).generations[0].message
        words = msg.text.split(" ") if msg.text and not msg.tool_calls else [None]
        for i, w in enumerate(words):
            # usage and model name ride on the last chunk, as with a real streamed reply
            extra = {"usage_metadata": msg.usage_metadata, "response_metadata": msg.response_metadata} \
                if i == len(words) - 1 and msg.usage_metadata else {}
            if w is None:
                chunk = AIMessageChunk(content=msg.content, tool_call_chunks=[
                    {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": j}
                    for j, tc in enumerate(msg.tool_calls)
                ], id=msg.id, **extra)
            else:
                chunk = AIMessageChunk(content=(w if i == 0 else " " + w), id=msg.id, **extra)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=ChatGenerationChunk(message=chunk))
            yield ChatGenerationChunk(message=chunk)
            if self.on_token:
                self.on_token(i)


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


