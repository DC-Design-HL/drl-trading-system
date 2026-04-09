---
name: telegram-response-protocol
description: Rules for responding to Chen via Telegram. Always send a completion reply after every task — never leave Chen guessing.
---

# Telegram Response Protocol

## Core Rule

**Always send a Telegram reply after finishing any task or request.** No exceptions.

Use `mcp__plugin_telegram_telegram__reply` with the originating `chat_id`.

## When to Reply

| Situation | Action |
|-----------|--------|
| Task complete (code change, restart, config, analysis) | Reply with brief summary of what was done |
| Task still running (long operation) | Send interim update so Chen knows it's in progress |
| Task failed or blocked | Reply immediately with what happened and why |
| No reply received after a request | This is a bug — fix the behavior |

## Format

- 2–5 lines max
- Lead with status (✅ Done / ⚠️ Issue / 🔄 In progress)
- State what was done and any key outcome (URL, metric, file changed)
- Skip filler words

## Why This Matters

Chen has no visibility into whether a task is running or complete unless explicitly told. Silence forces him to ask "are you done?" which wastes his time and is annoying.
