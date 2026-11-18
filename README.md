# Steerable harness (on deepagents)

## The problem

A standard agent turn runs blind from question to answer. If it finds a gap halfway through, such as a missing month or a NULL column, it assumes something and keeps going. You only see the assumption at the end, after paying for all the tokens built on top of it.

## What this adds (PRD Phase 1: live view and steering)

| Piece | PRD | What happens |
|---|---|---|
| Live view | S1, S9 | Tokens stream as they're generated (`stream_mode="messages"`, the sync equivalent of `stream_events`). Tool calls and results print as they happen. After each tool step, a panel shows the step number, model, running tokens (with cached input), plan progress and files touched. |
| Steering inbox | S2 to S5 | Typing never blocks. Plain text is a **Steer** and lands before the next model call. `/follow text` is a **Follow-up**, sent as its own turn after the task. `/collect text` is a **Collect**: held until the next pause, the next Steer or the end of the task. A later Steer wins if they conflict. `/stop` is an **Interrupt**. Every message prints its state: queued, delivered at step N, or dropped. |
| Interrupt | S6, T5 | `/stop` cancels the reply mid-generation (or before the next model call if a tool is running) and keeps the last checkpoint. Then: Enter resumes, typed text resumes with that edit, `/new` drops the run. |
| Step summaries | S10, S11 | Built from run state (tools called, result sizes, errors, files written, todo progress, tokens) with no extra model call. `summary_model=` adds an optional one-line recap from a cheap model. |
| Pause every N steps | S12, S13 | `/pause N` (or `PAUSE_EVERY=N`) stops at every Nth step boundary, even when the model would keep going: [1] Continue, [2] Steer with a caveat, [3] Interrupt. A caveat is delivered as a steer and added to the ledger as a constraint. |
| `check_in` tool | | The agent pauses itself at a gap (`interrupt()`), shows progress, gaps and its next step, and resumes from the same checkpoint with your answer. |
| `SelfReviewMiddleware` | | Every N tool calls, adds a "do these steps still make sense?" prompt. It is saved as a harness note, like a steer, so the cached prefix never changes (see Phase 3). |

Steering notes and caveats are appended at the end of the message list, never edited into earlier turns, so the cached prompt prefix stays valid (PRD T1).

## Phase 2: assumption ledger and pre-tool pauses

| Piece | PRD | What happens |
|---|---|---|
| Ledger first | A1 | Before any other tool, the agent calls `write_ledger` (scope, inputs, constraints, done_when). The run pauses at this first checkpoint: Enter confirms, typed text corrects it. Until it's confirmed, `ToolGuardMiddleware` answers any other tool call with "Blocked" instead of running it, and the agent is re-prompted. |
| Editable ledger | A2 | `/ledger` shows it. `/ledger constraints: exclude bot signups` adds an item and lands as a steer before the next model call. A step-pause caveat becomes a new constraint. |
| Ledger changes | A3 | The agent records each change with `update_ledger` (added / revised / dropped). The step panel shows them, e.g. `ledger: ~ scope: Jul to Oct 2026 → Aug to Oct 2026 (agent)`. |
| Ask before costly calls | A5 | Pauses before a subagent spawn (`task`), a call with over 20,000 chars of arguments, or a call estimated over `budget_share` of `budget_tokens`, and shows the estimate. Enter approves; anything else declines and the agent adjusts. |
| One interrupt per node | A6 | Each pause is a single `interrupt()` in its node. Blocked or declined calls get an artificial tool result, and LangGraph's own conditional edge sends the run back to the model. No loops around `interrupt()`. |
| Survives a restart | A7 | Checkpoints go to SQLite at `.harness/checkpoints.sqlite`. `python steerable.py --thread <id>` picks up a paused run in a new process. |

The ledger is stored as an append-only list of ops (`state["ledger_ops"]`), and the current ledger is folded from it. Parallel tool calls can't overwrite each other, and the list already records who changed what at which step, which is the start of the A8 audit log.

## Phase 3: token meter and budgets

| Piece | PRD | What happens |
|---|---|---|
| Append-only history | T1 | Steers, caveats, ledger edits and the self-review prompt are all appended at the end and saved. Every request starts with the previous request, byte for byte, so repeated input is billed at the cache-read rate. |
| 1-hour cache | T2 | Breakpoints after the system prompt, after the tool definitions, and on the latest message, with the 1-hour lifetime, because a supervised run can pause for you longer than the 5-minute cache lasts. This replaces deepagents' own 5-minute caching middleware. `build_agent(cache_ttl="5m")` switches back. |
| Thinking pinned | T4 | The first call to each model in a thread fixes its thinking settings. If anything asks for different ones mid-run, the original settings are put back and the live view says so. |
| Budgets | T6 | `Budget(run=..., step=...)`, or `RUN_BUDGET` / `STEP_BUDGET` in the terminal. The step panel warns once the run crosses 80%, or when a step uses 80% of the step budget. At 100% the run pauses at the next step boundary: [1] Continue (adds one more run budget), [2] Set a new budget, [3] Stop, or type a note to continue with. A new question starts a fresh run budget. |
| Meter | T8 | Every step panel shows that call's input, cache read, cache write and output tokens from the usage fields the API returns, the run total against its budget, and the cost. `/tokens` prints the table for the whole thread, at any prompt. |

Budgets count every token except cache reads, which bill at 1/20 of the input price. Cache reads are shown next to the total instead.

```
── step 2 · claude-sonnet-5-5 · $0.0024 this step
   tokens: in 200 · cache read 3,000 · cache write 300 · out 100
   run:    4,700 tokens of 5,000 budget (94%), plus 3,000 cache reads · $0.013 so far
```

## Run it

```bash
uv venv && uv pip install deepagents langchain-anthropic
```

```bash
python test_steerable.py && python test_ledger.py && python test_meter.py
```

```bash
ANTHROPIC_API_KEY=... python steerable.py
```

```bash
RUN_BUDGET=50000 STEP_BUDGET=15000 python steerable.py
```

The tests run offline with a scripted model and cost no tokens. They include all three gates. Phase 1: a steer lands before the next model call and reports the step it landed on. Phase 2: a run pauses at the ledger in one process and is resumed and finished by another, from the SQLite file alone. Phase 3: the meter shows cache reads, and a budget pauses the run at 100%. The cache tests build the exact request a real `ChatAnthropic` would send, without sending it, and check the breakpoints and that each request extends the one before. For the live run, ask "Compare signups by channel Jul to Oct and activation per channel". The toy data has a missing September and a NULL organic activation rate, so the agent should check in.

## The build, step by step

1. **Base agent.** `create_deep_agent(model, tools, system_prompt)`. This gives you planning, files and sub-agents.
2. **Persistence.** Pass `checkpointer=InMemorySaver()` and call it with `config={"configurable": {"thread_id": ...}}`. Without this, a pause can't be resumed. Use `SqliteSaver` or `PostgresSaver` so paused runs survive a restart.
3. **Agent-initiated pause.** Add a tool whose body calls `interrupt(payload)`. The stream yields `__interrupt__`, and you resume with `agent.stream(Command(resume=answer), config)`. The tool's docstring sets when the model calls it, so list the triggers there (missing data, an assumption it would have to make, a contradiction, a plan change, a costly step).
4. **User-initiated steering.** Add middleware with `before_model` that drains a queue and returns `{"messages": [HumanMessage(...)]}`. To stop, return `{"jump_to": "end"}` and decorate the hook with `@hook_config(can_jump_to=["end"])`.
5. **Self-review cadence.** Use `wrap_model_call` plus `request.override(messages=[..., reminder])` to nudge the model without saving the nudge to history.
6. **UI loop.** Read input on a background thread. While the agent runs, each line is a steering note. During a check-in, the next line is your answer.
7. **Later.** Add `interrupt_on={"execute": True}` for tool approvals, which uses the same resume loop. Move the runner to a web UI or Slack by replacing `run_terminal`.

## Benchmark results (Claude Sonnet 5.5, 3 runs per mode, 2026-11-18)

`benchmark.py long 3`: "activation by country" with 3 possible definitions and no default. Measured before Phases 2 and 3. The ledger checkpoint and the saved self-review prompt change the steerable mode, so these numbers need a rerun. The benchmark keeps the 5-minute cache for every mode, so the comparison stays like for like.

| Mode | Tool calls | Cost per task | vs baseline |
|---|---|---|---|
| baseline (answer, then correct) | 98 | $0.223 | — |
| ask_text (plain agent told to ask first) | 34 | $0.083 | −63% |
| steerable (this harness) | 36 | $0.090 | −60% |

The saving comes from asking before doing the work: the plain agent ran the analysis for all 3 definitions. A plain agent with an "ask first" prompt saved just as much. `benchmark.py toy 3` (short task, obvious gaps) showed the harness costing about 2× a plain agent.

## Limits

- Steering takes effect at the next model call, not mid-tool. A 2-minute SQL query finishes before your note or `/stop` is read (PRD S7, later).
- `/stop` mid-reply relies on token streaming. A model or provider that doesn't stream can only be stopped at the next step boundary.
- deepagents 0.7 no longer adds `write_todos`, so `build_agent` adds LangChain's `TodoListMiddleware` itself.
- Whether the model calls `check_in` depends on the model and the prompt. The self-review prompt makes it more likely but doesn't guarantee it. For a guaranteed pause, use `/pause N`.
- The ledger adds one pause at the start of every task. That is the point of A1, but it costs one extra model turn on tasks with nothing to assume.
- A correction typed at the ledger checkpoint is free text. The agent is told to record it with `update_ledger`, but whether it does depends on the model.
- Cost estimates for approvals are rough: a subagent is priced as 5 model calls at the run's average size so far, any other call by its argument size.
- Budgets are checked at step boundaries. One very large step can overshoot before the pause, and its tool calls run first. The A5 approval catches calls it can see are large in advance.
- Tokens a subagent (`task`) spends aren't in the parent's messages, so the meter and budgets don't count them yet.
- Costs are only shown for models in `PRICES` (Sonnet 5.5 now). Other models show tokens without a cost.
- A 1-hour cache write costs 2x input against 1.25x for 5 minutes. It pays off when you pause for more than 5 minutes and the prefix would otherwise be written again.
- The cache tests check the request bytes, not real cache hits. Hit rates need a live run.
- `benchmark.py` and `live_test.py` use in-memory checkpoints because they reuse thread IDs.
- Anthropic models always call `https://api.anthropic.com`, even if your shell has `ANTHROPIC_BASE_URL` set (the Claude desktop app sets one). To use a proxy on purpose, set `HARNESS_ANTHROPIC_BASE_URL`.
