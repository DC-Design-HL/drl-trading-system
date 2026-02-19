---
description: Fix a bug reported in the dashboard or trading system
---

# Bug Fix Workflow

// turbo-all

## 1. Understand the Bug
- Read the error message carefully (screenshot, log, or user description)
- Check the HF Space runtime logs:
  ```
  curl -s -H "Authorization: Bearer $(grep HF_TOKEN .env | cut -d= -f2)" "https://huggingface.co/api/spaces/chen470/drl-trading-bot/logs/run" | head -c 5000
  ```
- Identify the failing file and line number

## 2. Reproduce Locally
- Run the relevant module locally using the venv:
  ```
  BINANCE_PROXY=$(grep BINANCE_PROXY .env | cut -d= -f2) ./venv/bin/python3 -c "..."
  ```
- Confirm the error reproduces before touching any code

## 3. Fix
- Make the minimal change needed to fix the bug
- Keep changes scoped — do NOT refactor unrelated code while fixing

## 4. Test Locally
- Run the fixed module again with the same test from Step 2
- Confirm the output is correct (no exception, expected values)

## 5. Deploy
```
git add <changed files>
git commit -m "Fix: <short description>"
git push origin main
```

## 6. Verify on HF
- Wait ~90s for build to complete
- Check the dashboard card that was broken
- Confirm the fix is live and no new errors appear

## Done Criteria
- The reported error is gone from the UI
- No new errors introduced
- Data shows expected values on the live dashboard
