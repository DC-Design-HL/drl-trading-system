# 🔑 API Keys Setup Guide - Crypto News Aggregator

## 📋 Required API Keys/Tokens

Here are the **exact environment variables** you need to provide:

```bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRYPTO NEWS API KEYS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. CryptoPanic (OPTIONAL but recommended)
CRYPTOPANIC_TOKEN=

# 2. Reddit API (REQUIRED for social sentiment)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=

# 3. CryptoCompare (OPTIONAL but recommended)
CRYPTOCOMPARE_API_KEY=
```

---

## 🚀 Step-by-Step Setup Instructions

### **1. CryptoPanic API Token** (⚠️ OPTIONAL - Works without it!)

**Status:** OPTIONAL (public feed works without token)
**Free Tier:** Unlimited
**Cost:** FREE forever

**How to Get:**

1. Go to: https://cryptopanic.com/developers/api/
2. Click "Get Your Free API Token"
3. Sign up with email
4. Copy your token from the dashboard
5. Add to `.env`:

```bash
CRYPTOPANIC_TOKEN=your_token_here
```

**⚠️ NOTE:** The aggregator works WITHOUT this token using the public feed. Token gives you:
- Higher rate limits
- Access to more filters
- Priority support

**Skip this if you want - public feed is fine!**

---

### **2. Reddit API Credentials** (✅ REQUIRED)

**Status:** REQUIRED (for social sentiment)
**Free Tier:** Unlimited (60 requests/min)
**Cost:** FREE

**How to Get:**

#### Step 1: Create Reddit App

1. Go to: https://www.reddit.com/prefs/apps
2. Scroll to bottom, click **"create another app..."**
3. Fill in the form:
   - **name:** `DRL-Trading-Bot` (or any name)
   - **App type:** Select **"script"**
   - **description:** `Crypto news sentiment analysis`
   - **about url:** (leave blank)
   - **redirect uri:** `http://localhost:8080` (required but not used)
4. Click **"create app"**

#### Step 2: Get Credentials

After creating, you'll see:

```
┌─────────────────────────────────────┐
│ DRL-Trading-Bot                     │
│ personal use script                 │
│                                     │
│ abc123def456                        │  ← This is your CLIENT_ID
│                                     │
│ secret: xyz789abc123def456...       │  ← This is your CLIENT_SECRET
└─────────────────────────────────────┘
```

#### Step 3: Add to .env

```bash
REDDIT_CLIENT_ID=abc123def456
REDDIT_CLIENT_SECRET=xyz789abc123def456
```

**Example:**
```bash
REDDIT_CLIENT_ID=p-jcoLKBynTLew
REDDIT_CLIENT_SECRET=gko_LXELoV07ZBNUXrvWZfzE3aI
```

---

### **3. CryptoCompare API Key** (⚠️ OPTIONAL)

**Status:** OPTIONAL (but recommended for verified news)
**Free Tier:** 100,000 calls/month (3,333/day)
**Cost:** FREE

**How to Get:**

1. Go to: https://min-api.cryptocompare.com/
2. Click **"Get Your Free API Key"**
3. Sign up with email
4. Verify email
5. Go to dashboard: https://www.cryptocompare.com/cryptopian/api-keys
6. Copy your API key
7. Add to `.env`:

```bash
CRYPTOCOMPARE_API_KEY=your_key_here
```

**Example:**
```bash
CRYPTOCOMPARE_API_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

---

## ✅ Final .env File

After getting all keys, your `.env` should have:

```bash
# ... existing keys ...

# Crypto News Aggregator
CRYPTOPANIC_TOKEN=your_cryptopanic_token_here  # OPTIONAL
REDDIT_CLIENT_ID=your_reddit_client_id_here    # REQUIRED
REDDIT_CLIENT_SECRET=your_reddit_secret_here   # REQUIRED
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key   # OPTIONAL
```

---

## 🎯 What Works Without API Keys?

### **Minimum Setup (Only Reddit - 1 source):**

```bash
# Only these 2 are REQUIRED:
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```

**Result:**
- ✅ Reddit sentiment (30% weight)
- ⚠️ No CryptoPanic (50% weight lost)
- ⚠️ No CryptoCompare (20% weight lost)
- **Total coverage: 30%**

---

### **Recommended Setup (All 3 sources):**

```bash
CRYPTOPANIC_TOKEN=...          # FREE, unlimited
REDDIT_CLIENT_ID=...           # FREE, unlimited
REDDIT_CLIENT_SECRET=...       # FREE, unlimited
CRYPTOCOMPARE_API_KEY=...      # FREE, 100K/month
```

**Result:**
- ✅ CryptoPanic sentiment (50% weight)
- ✅ Reddit sentiment (30% weight)
- ✅ CryptoCompare sentiment (20% weight)
- **Total coverage: 100%**

---

## 🔍 Verification

After adding keys to `.env`, test them:

```bash
# Test the aggregator
python -c "from src.features.crypto_news_aggregator import CryptoNewsAggregator; agg = CryptoNewsAggregator('BTC'); print(agg.get_aggregated_sentiment())"
```

**Expected output:**
```
✅ Reddit client initialized
📰 CryptoPanic: 20 posts for BTC
🔴 Reddit: 15 posts for Bitcoin
📊 CryptoCompare: 30 articles for BTC
📰 Aggregated sentiment [BTC]: 🟢 +0.34 (confidence=0.72, sources=3/3, trend=improving)
```

**If you see errors:**
- `Reddit initialization failed` → Check REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
- `CryptoPanic error: 403` → Check CRYPTOPANIC_TOKEN (or remove it to use public feed)
- `CryptoCompare error: 401` → Check CRYPTOCOMPARE_API_KEY

---

## 📊 API Limits Summary

| Source | Free Tier Limit | Enough for Trading? | Cost if Exceeded |
|--------|----------------|---------------------|------------------|
| **CryptoPanic** | Unlimited | ✅ Yes | FREE forever |
| **Reddit** | 60 req/min | ✅ Yes (we use ~1/hour) | FREE forever |
| **CryptoCompare** | 100K/month (3,333/day) | ✅ Yes (we use ~24/day) | $17/month for 500K |

**For hourly trading:**
- CryptoPanic: ~24 calls/day (cache: 5 min)
- Reddit: ~24 calls/day (cache: 5 min)
- CryptoCompare: ~24 calls/day (cache: 5 min)

**Total:** ~72 calls/day across all sources

---

## 🚨 Troubleshooting

### Error: "praw.exceptions.ResponseException: received 401 HTTP response"

**Cause:** Reddit credentials are wrong

**Solution:**
1. Double-check `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`
2. Make sure you created a "script" app (not "web app")
3. Recreate the app if needed

---

### Error: "Reddit client not available"

**Cause:** `praw` library not installed

**Solution:**
```bash
pip install praw
```

---

### Error: "CryptoPanic error: 429"

**Cause:** Rate limit exceeded (unlikely with caching)

**Solution:**
1. Increase `cache_ttl` in `crypto_news_aggregator.py` to 600 (10 min)
2. Get API token to increase limits

---

### Warning: "Reddit credentials not set - using fallback"

**Cause:** Missing `REDDIT_CLIENT_ID` or `REDDIT_CLIENT_SECRET`

**Solution:**
1. Add both variables to `.env`
2. Restart the application

---

## ✅ Quick Test Checklist

Run these commands to verify everything works:

```bash
# 1. Check .env file
cat .env | grep -E "REDDIT|CRYPTOPANIC|CRYPTOCOMPARE"

# Should output:
# REDDIT_CLIENT_ID=...
# REDDIT_CLIENT_SECRET=...
# CRYPTOPANIC_TOKEN=...
# CRYPTOCOMPARE_API_KEY=...

# 2. Test Reddit
python -c "import praw; r = praw.Reddit(client_id='YOUR_ID', client_secret='YOUR_SECRET', user_agent='test'); print('✅ Reddit OK')"

# 3. Test CryptoPanic
curl "https://cryptopanic.com/api/v1/posts/?public=true&currencies=BTC" | head -20

# 4. Test CryptoCompare
curl "https://min-api.cryptocompare.com/data/v2/news/?categories=BTC" | head -20

# 5. Test full aggregator
python test_crypto_news.py
```

---

## 📝 Summary

**Exact variables you need to add to `.env`:**

```bash
# REQUIRED (Reddit)
REDDIT_CLIENT_ID=your_reddit_app_client_id
REDDIT_CLIENT_SECRET=your_reddit_app_client_secret

# OPTIONAL (CryptoPanic - works without it)
CRYPTOPANIC_TOKEN=your_cryptopanic_token

# OPTIONAL (CryptoCompare - recommended)
CRYPTOCOMPARE_API_KEY=your_cryptocompare_key
```

**Next step:** Get these keys and add them to `.env`, then run the test!

---

## 🔗 Quick Links

- **Reddit Apps:** https://www.reddit.com/prefs/apps
- **CryptoPanic API:** https://cryptopanic.com/developers/api/
- **CryptoCompare API:** https://min-api.cryptocompare.com/

---

**Questions?** Run the test script and check the output:

```bash
python test_crypto_news.py
```

The script will tell you exactly which keys are missing!
