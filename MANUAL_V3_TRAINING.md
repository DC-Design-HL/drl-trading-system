# Manual V3 Model Retraining Guide

**Status:** Script ready, run manually when Quick Wins validated
**Priority:** MEDIUM (Quick Wins validation is higher priority)

---

## ⚠️ IMPORTANT: Run This AFTER Quick Wins Validation

**Wait 24-48 hours** to validate Quick Wins performance before retraining.

Why?
- Quick Wins already provide +124% P&L improvement
- Model retraining adds incremental gains (~+$300-400)
- Better to validate parameter fixes work first
- Retraining with improved baseline produces better results

---

## 🚀 How to Run V3 Retraining

### Step 1: Open Terminal

Navigate to project directory:
```bash
cd /Users/chenluigi/WebstormProjects/drl-trading-system
```

### Step 2: Activate Virtual Environment

```bash
source venv/bin/activate
```

Verify activation:
```bash
which python
# Should show: /Users/chenluigi/WebstormProjects/drl-trading-system/venv/bin/python
```

### Step 3: Run Training

**Option A: Full Training (2M timesteps, ~2-3 hours)**
```bash
python retrain_ultimate_v3.py --timesteps 2000000 --lr 3e-4
```

**Option B: Quick Test (100K timesteps, ~10 minutes)**
```bash
python retrain_ultimate_v3.py --timesteps 100000 --lr 3e-4
```

**Option C: With Logging**
```bash
python retrain_ultimate_v3.py --timesteps 2000000 2>&1 | tee logs/v3_training_manual.log
```

### Step 4: Monitor Progress

Training will show:
```
================================================================================
TRAINING ULTIMATE V3 - POST-QUICK WINS
================================================================================

1. Loading training data (BTC, ETH, SOL only)...
  Loading BTCUSDT data...
  Loading ETHUSDT data...
  Loading SOLUSDT data...

Total training data: 12966 candles across 3 assets

2. Creating training environments...
3. Initializing PPO model...
4. Setting up callbacks...
5. Starting training...

[Progress bar showing training progress]
```

**Validation checkpoints** every 50K steps:
```
============================================================
VALIDATION @ Step 50000
  Sharpe:   1.234
  Win Rate: 58.3%
  Trades:   42
============================================================
💾 New best model saved (Sharpe=1.234)
```

---

## 📊 What Happens During Training

### Data Loading (1-2 minutes)
- Fetches 180 days of BTC, ETH, SOL data
- **Excludes XRP** (75% loss rate)
- Splits 80/20 train/val (10,372 train / 2,594 val candles)

### Environment Creation (30 seconds)
- Creates QuickWinsEnv with:
  - Regime-aware rewards (2.0x HIGH_VOL bonus)
  - Time-based bonuses (holding >12h)
  - SL penalty adjustments

### Model Training (2-3 hours for 2M steps)
- PPO with 128x128 network (smaller than V1 to prevent overfitting)
- Validates every 50K steps
- Early stopping if no improvement for 5 checks
- Saves best model automatically

---

## ✅ Success Indicators

**Training is working if you see:**
1. ✅ Data loaded successfully (12,966 candles)
2. ✅ Environment created without errors
3. ✅ Progress bar advancing
4. ✅ Validation Sharpe improving over time
5. ✅ "💾 New best model saved" messages

**Training complete when you see:**
```
✅ Training complete!
  Duration: 2.3 hours
  Best Sharpe: 1.456

💾 Final model: ultimate_v3.zip
💾 Backup: ultimate_v3_20260315_1045.zip
📊 Training report: ultimate_v3_training_report.json

🎉 V3 TRAINING COMPLETE!
```

---

## 📁 Output Files

After training completes:

### Models
- `data/models/ultimate_v3.zip` - Final trained model
- `data/models/ultimate_v3_best.zip` - Best checkpoint (use this!)
- `data/models/ultimate_v3_vec_normalize.pkl` - Normalization stats
- `data/models/ultimate_v3_20260315_HHMM.zip` - Timestamped backup

### Reports
- `data/models/ultimate_v3_training_report.json` - Full training metrics

Example report:
```json
{
  "timestamp": "20260315_1045",
  "duration_hours": 2.3,
  "total_timesteps": 2000000,
  "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
  "train_candles": 10372,
  "val_candles": 2594,
  "best_sharpe": 1.456,
  "quick_wins_integrated": true,
  "xrp_excluded": true,
  "regime_aware": true
}
```

---

## 🧪 Validation After Training

### Step 1: Backtest V3 Model

```bash
python backtest_strategy.py --asset BTCUSDT --days 180 --model data/models/ultimate_v3_best.zip
```

### Step 2: Compare vs Quick Wins Only

| Metric | Quick Wins Only | Quick Wins + V3 | Improvement |
|--------|----------------|-----------------|-------------|
| P&L | +$1,114 | +$1,500+ | +$400 |
| Sharpe | 1.2 | >1.5 | +25% |
| Win Rate | ~50% | >55% | +10% |

### Step 3: A/B Test on Dev Space

**Deploy V3 to dev Space:**
```bash
# Copy V3 model to production location
cp data/models/ultimate_v3_best.zip data/models/ultimate_agent.zip
cp data/models/ultimate_v3_vec_normalize.pkl data/models/ultimate_agent_vec_normalize.pkl

# Commit and push
git add data/models/ultimate_agent*.zip data/models/ultimate_agent*.pkl
git commit -m "Model: Deploy V3 with Quick Wins integration"
git push origin dev
git push hf-dev dev:main
```

**Monitor for 24-48h**, compare vs Quick Wins only.

---

## ⚠️ Troubleshooting

### Error: ModuleNotFoundError

**Cause:** Virtual environment not activated

**Fix:**
```bash
source venv/bin/activate
which python  # Verify it's using venv
python retrain_ultimate_v3.py
```

### Error: No data loaded

**Cause:** Binance API issues or network problems

**Fix:**
- Check internet connection
- Try again (API may have rate limited)
- Check `data/historical/` for cached files

### Error: CUDA out of memory

**Cause:** GPU memory insufficient (if using GPU)

**Fix:**
```bash
# Force CPU training
CUDA_VISIBLE_DEVICES="" python retrain_ultimate_v3.py
```

### Training too slow

**Cause:** Running on CPU (normal, expected)

**Status:** 2M timesteps takes ~2-3 hours on CPU - this is normal

**Options:**
- Reduce timesteps: `--timesteps 1000000` (~1.5 hours)
- Run overnight
- Be patient :)

---

## 📈 Expected Timeline

| Task | Duration | Description |
|------|----------|-------------|
| **Data Loading** | 1-2 min | Fetch 180 days BTC/ETH/SOL |
| **Environment Setup** | 30 sec | Create training/val envs |
| **Training (2M steps)** | 2-3 hours | PPO with validation |
| **Validation (each 50K)** | 2-3 min | Evaluate on unseen data |
| **Total** | **2.5-3.5 hours** | End-to-end |

---

## 🎯 When to Run This

### ✅ Run V3 Retraining When:
1. Quick Wins validated (24-48h of good performance)
2. System showing improved profitability
3. You want additional +$300-400 gains
4. You have 3-4 hours for training

### ❌ Don't Run Yet If:
1. Quick Wins not validated (<24h deployed)
2. Dev Space showing errors/issues
3. Current system losing money (fix first)

---

## 💡 Pro Tips

1. **Run overnight** - 2-3 hours is perfect for overnight training
2. **Check logs** - Monitor `logs/v3_training_manual.log` for progress
3. **Save outputs** - Keep training reports for future reference
4. **Test first** - Use `--timesteps 100000` to verify everything works
5. **Backup** - Models are auto-backed up with timestamps

---

## 📞 Quick Reference

**Start Training:**
```bash
cd /Users/chenluigi/WebstormProjects/drl-trading-system
source venv/bin/activate
python retrain_ultimate_v3.py --timesteps 2000000
```

**Check Progress:**
```bash
tail -f logs/v3_training_manual.log
```

**After Training:**
```bash
# Backtest V3
python backtest_strategy.py --asset BTCUSDT --days 180 --model data/models/ultimate_v3_best.zip

# Deploy to dev Space
cp data/models/ultimate_v3_best.zip data/models/ultimate_agent.zip
git push hf-dev dev:main
```

---

**Priority:** Medium (after Quick Wins validation)
**Effort:** 3-4 hours (mostly automated)
**Expected Gain:** +$300-400 additional improvement
