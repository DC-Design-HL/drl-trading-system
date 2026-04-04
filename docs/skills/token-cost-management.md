---
name: token-cost-management
description: Rules for managing token costs, context window compaction, and subagent delegation. Apply to EVERY interaction — check context size, choose cheapest execution path, remind user to compact.
---

# Token & Cost Management

## Context Compaction Alerts

Alert Chen in the group chat when context reaches these thresholds:

| Threshold | Action |
|-----------|--------|
| 100K tokens | "⚠️ Context at 100K — let's /compact now" — ask Chen immediately |

**Compact at 100K. Don't wait. Ask Chen proactively.**

Check context size at the start of each interaction using `session_status`.

## Subagent Model Rule

**All subagents MUST use Sonnet, not Opus.** Set model override on every `sessions_spawn`:

```
model: "anthropic/claude-sonnet-4-20250514"
```

CEO (this session) stays on Opus. Config is already set in openclaw.json but always verify when spawning.

## Direct Edit vs Subagent Decision

Do the work **directly** (no subagent) when:
- Single file edit or fix
- Simple bug fix with known root cause
- Reading logs / checking state
- Config changes
- Git operations

Spawn a **subagent** only when:
- Multi-file refactoring (5+ files)
- Deep codebase analysis requiring exploration
- Tasks that need iterative file discovery
- Complex architectural planning with research

**Default: do it yourself.** Only delegate when the task truly benefits from a separate agent context.

## No AI Crons

Never use cron jobs that fire into the AI session — they waste tokens by loading the full context on every invocation. System monitoring should be bash/Python scripts that only message the group when something needs attention.
