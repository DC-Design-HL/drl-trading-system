# Production Monitoring Workflow

**Goal:** Monitor the live Hugging Face deployment and debug issues

---

## 🔍 Monitoring Commands

### Check Live Runtime Logs
```bash
# Real-time streaming logs (use Ctrl+C to stop)
curl -N -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot/logs/run"

# Last 100 lines
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot/logs/run" | tail -100

# Search for errors
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot/logs/run" | grep -i "error\|exception\|traceback"
```

### Check Build Logs
```bash
# Check deployment/build status
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot/logs/build"

# Look for build success
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot/logs/build" | grep -E "DONE|Pushing image|ERROR"
```

---

## 🐛 Common Issues & Fixes

**Out of Memory (Exit code 137):**
- Reduce memory usage in code
- Optimize model loading

**Import Errors:**
- Check requirements.txt
- Verify dependencies installed

**API Rate Limits:**
- Add caching
- Use proxy

---

**Version:** 1.0
