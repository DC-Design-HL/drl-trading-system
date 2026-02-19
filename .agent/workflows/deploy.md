---
description: Deploy and verify changes to the Hugging Face Space
---

# Deploy Workflow

// turbo-all

## 1. Pre-deploy Check
- Confirm there are no uncommitted changes that shouldn't go out:
  ```
  git status
  git diff --stat
  ```
- Make sure `.env` is in `.gitignore` (it always should be — never commit secrets)

## 2. Commit & Push
```
git add <files>
git commit -m "<type>: <description>"
git push origin main
```

Commit message types: `Fix:`, `Feature:`, `Perf:`, `Refactor:`, `Docs:`

## 3. Monitor Build
- Check build logs (wait up to 3 minutes):
  ```
  curl -s -H "Authorization: Bearer $(grep HF_TOKEN .env | cut -d= -f2)" \
    "https://huggingface.co/api/spaces/chen470/drl-trading-bot/logs/build"
  ```
- Look for `DONE` and `Pushing image` — that means build succeeded

## 4. Verify Runtime
- Wait 30s after build for container to start
- Visit: https://huggingface.co/spaces/chen470/drl-trading-bot
- Check:
  - [ ] Dashboard loads without error
  - [ ] Bot status is correct (RUNNING / STOPPED)
  - [ ] Market Analysis cards show data (Whale, Funding, Order Flow)
  - [ ] No "Data Error" banners

## Done Criteria
- HF build completed with no errors
- Live dashboard shows expected state
- No runtime crashes in the first 2 minutes
