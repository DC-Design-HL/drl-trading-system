---
description: Train or retrain the DRL trading model
---

# Model Training Workflow

// turbo-all

## 1. Clarify Training Goal
Before starting, confirm:
- Which asset(s)? (BTC, ETH, SOL, XRP, or all)
- Training from scratch or fine-tuning existing model?
- How many timesteps? (default: 500k for quick, 2M for full)

## 2. Prepare Data
- Fetch fresh historical data:
  ```
  ./venv/bin/python3 -c "
  from src.data.multi_asset_fetcher import MultiAssetDataFetcher
  f = MultiAssetDataFetcher()
  df = f.fetch_asset('BTCUSDT', '1h', days=365)
  print(f'Fetched {len(df)} rows')
  "
  ```

## 3. Run Training
- Quick training (local):
  ```
  ./venv/bin/python3 train_ultimate.py --timesteps 500000
  ```
- Full multi-asset:
  ```
  ./venv/bin/python3 train_ultimate.py --timesteps 2000000 --assets BTCUSDT ETHUSDT SOLUSDT XRPUSDT
  ```
- Monitor for: reward improvement, no NaN losses, stable entropy

## 4. Evaluate
- Check final reward vs baseline (buy-and-hold)
- Confirm model saved to `data/models/`

## 5. Deploy New Model
- Copy new model to correct path:
  ```
  cp data/models/latest_model.zip data/models/ultimate_agent.zip
  ```
- Commit and push:
  ```
  git add data/models/
  git commit -m "Model: Retrained on <date> - <asset> <timesteps>ts"
  git push origin main
  ```

## Done Criteria
- Model file exists in `data/models/`
- Dashboard shows updated "Model Date" in Active Model card
- Bot loads the new model without errors
