#!/bin/sh
# Run Claude Code against the local big2flash server (start it first: .venv/bin/python engine/server.py).
M=big2flash
export ANTHROPIC_BASE_URL="${BIG2FLASH_URL:-http://127.0.0.1:8080}"
export ANTHROPIC_AUTH_TOKEN=local
export ANTHROPIC_MODEL=$M ANTHROPIC_DEFAULT_OPUS_MODEL=$M ANTHROPIC_DEFAULT_SONNET_MODEL=$M
export ANTHROPIC_DEFAULT_HAIKU_MODEL=$M ANTHROPIC_SMALL_FAST_MODEL=$M CLAUDE_CODE_SUBAGENT_MODEL=$M
# no telemetry/update/side traffic; long timeout (a cold 20K-token prompt takes ~1 min to read)
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 API_TIMEOUT_MS=1800000
# the server holds up to 98K tokens: compact before that, and keep replies within the room left
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=${BIG2FLASH_COMPACT_WINDOW:-100000} CLAUDE_CODE_MAX_OUTPUT_TOKENS=16000
exec claude "$@"
