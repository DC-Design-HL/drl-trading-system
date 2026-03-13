# Dual Environment Setup

**Purpose:** Maintain separate production and development Hugging Face Spaces

---

## 🌍 Environments

### Production Environment
- **Space:** `Chen4700/drl-trading-bot`
- **Branch:** `main`
- **URL:** https://huggingface.co/spaces/Chen4700/drl-trading-bot
- **Purpose:** Live trading system (DO NOT TOUCH)
- **Git Remote:** `hf` (main)

### Development Environment
- **Space:** `Chen4700/drl-trading-bot-dev`
- **Branch:** `dev`
- **URL:** https://huggingface.co/spaces/Chen4700/drl-trading-bot-dev
- **Purpose:** Testing new features and changes
- **Git Remote:** `hf-dev`

---

## 🔄 Workflow

### Making Changes

1. **Always work on `dev` branch:**
   ```bash
   git checkout dev
   ```

2. **Make your changes, commit, and push to dev Space:**
   ```bash
   git add <files>
   git commit -m "Feature: Description"
   git push origin dev        # Push to GitHub dev branch
   git push hf-dev dev:main   # Deploy to dev Space
   ```

3. **Test on dev Space:**
   - Visit: https://huggingface.co/spaces/Chen4700/drl-trading-bot-dev
   - Check logs, verify functionality

4. **When ready for production:**
   ```bash
   git checkout main
   git merge dev
   git push origin main       # Push to GitHub main branch
   git push hf main:main      # Deploy to production Space
   ```

---

## 🔧 Git Remotes

### View Remotes
```bash
git remote -v
```

**Expected output:**
```
origin      https://huggingface.co/spaces/Chen4700/drl-trading-bot (fetch)
origin      https://huggingface.co/spaces/Chen4700/drl-trading-bot (push)
hf          https://huggingface.co/spaces/Chen4700/drl-trading-bot (fetch)
hf          https://huggingface.co/spaces/Chen4700/drl-trading-bot (push)
hf-dev      https://huggingface.co/spaces/Chen4700/drl-trading-bot-dev (fetch)
hf-dev      https://huggingface.co/spaces/Chen4700/drl-trading-bot-dev (push)
```

### Add Missing Remotes
```bash
# If hf-dev is missing
source .env
git remote add hf-dev https://user:${HF_TOKEN}@huggingface.co/spaces/Chen4700/drl-trading-bot-dev
```

---

## 📊 Monitoring Both Environments

### Production Logs
```bash
source .env
curl -N -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot/logs/run"
```

### Dev Logs
```bash
source .env
curl -N -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot-dev/logs/run"
```

---

## ⚙️ Creating/Recreating Dev Space

If you need to recreate the dev Space:

```bash
./venv/bin/python create_dev_space.py
```

This script will:
1. Duplicate the production Space
2. Configure secrets from `.env`
3. Set up the dev environment

Then push your dev branch:
```bash
git push hf-dev dev:main --force
```

---

## 🔒 Security

**Secrets Configuration:**
- Both Spaces share the same secrets (HF_TOKEN, MONGO_URI, etc.)
- Secrets are automatically added by `create_dev_space.py`
- Update secrets manually via HF UI if needed:
  - Production: https://huggingface.co/spaces/Chen4700/drl-trading-bot/settings
  - Dev: https://huggingface.co/spaces/Chen4700/drl-trading-bot-dev/settings

---

## 🎯 Best Practices

1. **Never push directly to production Space** - Always test on dev first
2. **Keep dev and main in sync** - Regularly merge dev to main
3. **Monitor both environments** - Check logs after deployments
4. **Use meaningful commit messages** - Helps track what's deployed where
5. **Test thoroughly on dev** - Run backtests, check all features

---

## 🚨 Emergency Rollback

If production breaks after deployment:

```bash
# Find the last working commit
git log main --oneline

# Reset to that commit
git checkout main
git reset --hard <commit-hash>
git push hf main:main --force

# Fix the issue on dev branch
git checkout dev
# ... make fixes ...
git push hf-dev dev:main
```

---

**Version:** 1.0
**Last Updated:** 2026-03-12
