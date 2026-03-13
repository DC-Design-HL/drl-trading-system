# Database Separation (Prod vs Dev)

**Purpose:** Ensure production and development environments use separate databases

---

## 🔒 **Problem**

By default, both production and dev Spaces would use the same MongoDB database:
- Same `MONGO_URI` connection string
- Same database name: `"trading_system"`
- Same collections: `state` and `trades`
- **Result:** Dev testing would corrupt production data! ❌

---

## ✅ **Solution**

We use an `ENVIRONMENT` variable to automatically select different databases:

| Environment | Variable Value | Database Name | Collections |
|-------------|---------------|---------------|-------------|
| **Production** | `ENVIRONMENT=production` (default) | `trading_system` | `state`, `trades` |
| **Development** | `ENVIRONMENT=dev` | `trading_system_dev` | `state`, `trades` |

### How It Works

In `src/data/storage.py`:
```python
# Line 103-105
environment = os.getenv("ENVIRONMENT", "production").lower()
db_name = "trading_system" if environment == "production" else f"trading_system_{environment}"
self.db = self.client.get_database(db_name)
```

**Logic:**
- If `ENVIRONMENT` is not set → defaults to `"production"` → uses `trading_system`
- If `ENVIRONMENT="dev"` → uses `trading_system_dev`
- If `ENVIRONMENT="staging"` → uses `trading_system_staging`

---

## 🌍 **Space Configuration**

### Production Space
- **Space:** `Chen4700/drl-trading-bot`
- **ENVIRONMENT:** `production` (default, not explicitly set)
- **Database:** `trading_system`
- **State Document:** `{"_id": "current_state"}` in `trading_system.state`

### Dev Space
- **Space:** `Chen4700/drl-trading-bot-dev`
- **ENVIRONMENT:** `dev` (set as Space variable)
- **Database:** `trading_system_dev`
- **State Document:** `{"_id": "current_state"}` in `trading_system_dev.state`

---

## 🔧 **Setup Instructions**

### For Existing Dev Space
The `ENVIRONMENT=dev` variable is automatically added when you run:
```bash
./venv/bin/python create_dev_space.py
```

### Manual Setup (if needed)
If you need to manually add the variable:

```python
from huggingface_hub import HfApi
api = HfApi(token="your_hf_token")
api.add_space_variable(repo_id="Chen4700/drl-trading-bot-dev", key="ENVIRONMENT", value="dev")
```

Or via HF UI:
1. Go to: https://huggingface.co/spaces/Chen4700/drl-trading-bot-dev/settings
2. Navigate to "Variables and secrets"
3. Add variable: `ENVIRONMENT` = `dev`

---

## 📊 **Verify Separation**

### Check Production Logs
```bash
source .env
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot/logs/run" | \
  grep "Connected to MongoDB"
```

**Expected output:**
```
✅ Connected to MongoDB Atlas (database: trading_system, environment: production)
```

### Check Dev Logs
```bash
source .env
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot-dev/logs/run" | \
  grep "Connected to MongoDB"
```

**Expected output:**
```
✅ Connected to MongoDB Atlas (database: trading_system_dev, environment: dev)
```

---

## 🗄️ **MongoDB Atlas Structure**

Your MongoDB Atlas cluster now contains:

```
MongoDB Cluster (same MONGO_URI)
├── trading_system (production)
│   ├── state (collection)
│   │   └── _id: "current_state" (single document)
│   └── trades (collection)
│       └── [production trade documents]
│
└── trading_system_dev (development)
    ├── state (collection)
    │   └── _id: "current_state" (single document)
    └── trades (collection)
        └── [dev test trade documents]
```

**Key Points:**
- ✅ Same MongoDB cluster (cost-efficient)
- ✅ Separate databases (data isolation)
- ✅ No interference between prod and dev
- ✅ Can test freely on dev without affecting prod

---

## 🔄 **Data Migration (if needed)**

If you want to copy production data to dev for testing:

```python
from pymongo import MongoClient
import os

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))

# Copy state
prod_state = client["trading_system"]["state"].find_one({"_id": "current_state"})
if prod_state:
    client["trading_system_dev"]["state"].update_one(
        {"_id": "current_state"},
        {"$set": prod_state},
        upsert=True
    )

# Copy recent trades (last 100)
prod_trades = list(client["trading_system"]["trades"].find().sort("timestamp", -1).limit(100))
if prod_trades:
    client["trading_system_dev"]["trades"].insert_many(prod_trades)

print("✅ Data copied from production to dev")
```

---

## ⚠️ **Important Notes**

1. **Never set `ENVIRONMENT=production` on dev Space** - It would use prod database!
2. **Production Space defaults to production** - No variable needed
3. **Dev Space requires `ENVIRONMENT=dev`** - Set via Space variables
4. **Same MONGO_URI for both** - Cost-efficient, single cluster
5. **Different databases** - Complete data isolation

---

## 🚨 **Troubleshooting**

### Issue: Dev Space using production database
**Cause:** `ENVIRONMENT` variable not set or set incorrectly

**Fix:**
```bash
# Check current variable
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/Chen4700/drl-trading-bot-dev" | jq '.variables'

# Set it correctly
./venv/bin/python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv('HF_TOKEN'))
api.add_space_variable(repo_id='Chen4700/drl-trading-bot-dev', key='ENVIRONMENT', value='dev')
"
```

### Issue: Logs show wrong database
**Restart the Space to pick up variable changes:**
```bash
# Via API
./venv/bin/python -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv('HF_TOKEN'))
api.restart_space(repo_id='Chen4700/drl-trading-bot-dev')
"
```

---

**Version:** 1.0
**Last Updated:** 2026-03-12
