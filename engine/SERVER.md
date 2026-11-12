# Using Flash-Next from Claude Code

1. Plug in the 4T drive (the weights and the 32 GB n-gram table are read from it).
2. Start the server (loads in ~25 s, then listens on 127.0.0.1:8080):

       .venv/bin/python engine/server.py

3. In another terminal, in the project you want to work on:

       /path/to/engine/claude-local.sh

`claude-local.sh` points every Claude Code model slot (main, subagents, the small model used for
titles) at the server and disables non-essential traffic.

## What the server does
- Anthropic Messages API: `POST /v1/messages` (streaming or not), `POST /v1/messages/count_tokens`,
  `GET /v1/models`. Any model name is accepted; the reply echoes it.
- Tools: Claude Code's tool definitions go into the Qwen chat template; the model's
  `<tool_call><function=…><parameter=…>` replies come back as `tool_use` blocks, with parameter
  values typed from each tool's JSON schema. `tool_choice` any/tool is honored.
- Thinking: off unless the request enables it (Claude Code's thinking toggle). Budget below 8K →
  effort low, below 24K → medium, else xhigh. `--thinking on|off` overrides.
- Images in the conversation are replaced by a note (the engine loads the text model only).
- Prompt cache: the whole conversation is resent each turn, but only the new part is read. Up to 3
  conversations stay cached (`--sessions`), with rollback points every 4,096 tokens and at each
  prompt's end (`--snapshots`). The log line per request shows how much was cached.
- One request at a time; others queue. Closing the connection (Esc in Claude Code) stops
  generation.
- Context limit 98,304 tokens (`--max-context`); longer prompts get Anthropic's "prompt is too
  long" error. Claude Code is set to compact at its minimum window (100K) before that.

## Memory
The model keeps 68 GB of weights in memory, so the rest of the system has ~25 GB of the 96 GB.
When other apps use more than about 20 GB (browsers are the usual cause), macOS starts swapping
and writing drops from ~25 tok/s to 3–9 tok/s. Measured with 23 GB in use by other apps
(Chrome 7.6 GB): 3.4–9 tok/s, getting worse over time. Lowering the engine's own wired/cache limits
didn't help; freeing memory does.

## Speed to expect
Reading new prompt text ~320–375 tok/s (the first request with Claude Code's ~20K-token system
prompt takes about a minute; later turns only read what's new), writing ~25 tok/s.
