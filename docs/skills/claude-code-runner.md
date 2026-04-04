---
name: claude-code-runner
description: Run Claude Code CLI tasks on servers where the process runs as root. Claude Code refuses --permission-mode bypassPermissions under root/sudo. This skill handles creating a non-root user, copying the project, running Claude Code, and syncing results back. Use when delegating any coding task to Claude Code (analysis, planning, code changes, reviews) on this server.
---

# Claude Code Runner (Root Workaround)

Claude Code CLI rejects `--permission-mode bypassPermissions` when running as root. This skill solves that.

## Quick Use

```bash
bash /root/.openclaw/workspace-a2a-pipeline__ceo/skills/claude-code-runner/scripts/run-claude.sh <project-dir> "<prompt>"
```

## How It Works

1. Creates `claudeuser` (non-root) if missing
2. Copies project to `/home/claudeuser/<project-name>/`
3. Runs `claude --permission-mode bypassPermissions --print` as claudeuser with `ANTHROPIC_API_KEY`
4. Syncs modified files back to the original directory

## Requirements

- `ANTHROPIC_API_KEY` must be set as env var or in `/root/.bashrc`
- `claude` CLI must be installed globally (`/usr/bin/claude`)
- `rsync` must be available

## Usage Pattern (from CEO agent)

```bash
# Foreground (short tasks)
exec command:"bash skills/claude-code-runner/scripts/run-claude.sh /path/to/project 'Your prompt'" timeout:300

# Background (longer tasks)
exec command:"bash skills/claude-code-runner/scripts/run-claude.sh /path/to/project 'Your prompt'" background:true timeout:300
```

## Important

- Always run in background for tasks that may take >30 seconds
- Monitor with `process action:poll` and `process action:log`
- Push updates to the user proactively — never wait to be asked
- If the task fails, report the error immediately
