# Cloudflare Workers Proxy for Binance Testnet

This proxy bypasses geo-restrictions (Error 451) when accessing Binance testnet from HuggingFace Spaces.

## 🎯 Quick Start

### 1. Deploy the Worker (15 minutes)

Follow the step-by-step guide in **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)**

**Summary:**
1. Sign up at [workers.cloudflare.com](https://workers.cloudflare.com) (FREE, no credit card)
2. Create a new Worker
3. Copy code from `binance-testnet-proxy.js`
4. Save and Deploy
5. Copy your Worker URL

### 2. Configure HuggingFace Space

Add your Worker URL to HuggingFace Secrets:

1. Go to your Space Settings → Repository Secrets
2. Add new secret:
   - **Name:** `BINANCE_TESTNET_PROXY_URL`
   - **Value:** `https://your-worker.workers.dev`
3. Restart the Space

### 3. Test Locally (Optional)

Before deploying to HuggingFace, test locally:

```bash
# Add to .env file
echo "BINANCE_TESTNET_PROXY_URL=https://your-worker.workers.dev" >> .env

# Run test script
python test_proxy.py
```

If all tests pass ✅, you're ready to deploy to HuggingFace!

---

## 📊 Free Tier Limits

- **Requests:** 100,000 per day (FREE)
- **CPU Time:** 50ms per request
- **Workers:** 30 deployed scripts
- **Storage:** 1GB total

**For a single trading bot:** These limits are more than enough!

---

## 🔒 Security

- ✅ You control the proxy code
- ✅ API keys are not logged or stored
- ✅ CORS protection limits access to your HuggingFace Spaces only
- ✅ All traffic is encrypted (HTTPS)

---

## 📁 Files in This Directory

- **`binance-testnet-proxy.js`** - The Cloudflare Worker code
- **`DEPLOYMENT_GUIDE.md`** - Detailed deployment instructions
- **`README.md`** - This file

---

## 🆘 Need Help?

1. Check the [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for troubleshooting
2. Review Cloudflare Workers docs: https://developers.cloudflare.com/workers/
3. Check if your Worker is deployed correctly in the Cloudflare dashboard

---

## ✅ Success Checklist

- [ ] Cloudflare account created
- [ ] Worker deployed with proxy code
- [ ] Worker URL copied
- [ ] `BINANCE_TESTNET_PROXY_URL` added to HuggingFace Secrets
- [ ] HuggingFace Space restarted
- [ ] Testnet tab working! 🎉
