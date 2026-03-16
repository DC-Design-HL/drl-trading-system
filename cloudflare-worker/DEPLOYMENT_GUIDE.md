# Cloudflare Worker Deployment Guide

This guide will help you deploy the Binance Testnet Proxy to Cloudflare Workers (100% FREE, no credit card required).

---

## 📋 Prerequisites

- None! Just an email address to create a Cloudflare account

---

## 🚀 Step-by-Step Deployment (15 minutes)

### Step 1: Create Cloudflare Account

1. Go to https://workers.cloudflare.com/
2. Click **"Sign Up"** (top right)
3. Enter your email and create a password
4. Verify your email (check inbox)
5. **No credit card required!**

---

### Step 2: Create a New Worker

1. After logging in, you'll see the Workers dashboard
2. Click **"Create a Service"** or **"Create Worker"**
3. You'll see a default worker with sample code

---

### Step 3: Deploy the Proxy Code

1. **Delete** all the existing code in the editor
2. **Copy** the entire content of `binance-testnet-proxy.js` (from this folder)
3. **Paste** it into the Cloudflare Worker editor
4. Click **"Save and Deploy"** (bottom right)

---

### Step 4: Get Your Worker URL

1. After deployment, you'll see a success message
2. Your Worker URL will be something like:
   ```
   https://binance-proxy.YOUR-SUBDOMAIN.workers.dev
   ```
3. **Copy this URL** - you'll need it for Step 5

Example:
- If your subdomain is `mybot`, your URL will be:
  ```
  https://binance-proxy.mybot.workers.dev
  ```

---

### Step 5: Test Your Proxy

1. Open a new browser tab
2. Visit your Worker URL + `/api/v3/time`:
   ```
   https://binance-proxy.YOUR-SUBDOMAIN.workers.dev/api/v3/time
   ```
3. You should see Binance server time:
   ```json
   {"serverTime":1773694234567}
   ```

✅ If you see this, your proxy is working!

---

### Step 6: Update Environment Variable

1. Go to your HuggingFace Space settings
2. Go to **Repository secrets**
3. Add a new secret:
   - **Name:** `BINANCE_TESTNET_PROXY_URL`
   - **Value:** Your Worker URL (e.g., `https://binance-proxy.mybot.workers.dev`)
4. Click **Save**
5. Restart your Space

---

## 🎯 After Deployment

Once you've completed these steps:

1. ✅ Your proxy is deployed and running
2. ✅ 100,000 free requests per day
3. ✅ No geo-restrictions
4. ✅ Fast global edge network
5. ✅ Your API keys stay secure

The Python code has been updated to automatically use the proxy when the `BINANCE_TESTNET_PROXY_URL` environment variable is set.

---

## 🔧 Troubleshooting

### Worker Not Responding
- Check if the Worker is deployed (green checkmark in dashboard)
- Try re-deploying by clicking "Save and Deploy" again

### 403 or CORS Errors
- Make sure you copied the entire Worker code
- Check that the Worker URL is correct
- Verify your HuggingFace Space URLs are in the `allowedOrigins` array

### Rate Limit Exceeded
- Free tier: 100,000 requests/day
- For a single trading bot, this should be plenty
- If exceeded, consider upgrading to Workers Paid (~$5/month for 10M requests)

---

## 📊 Monitoring Usage

1. Go to your Cloudflare Workers dashboard
2. Click on your Worker
3. Click **"Metrics"** tab
4. You'll see:
   - Requests per day
   - Success rate
   - CPU time used
   - Errors

---

## 🔒 Security Notes

- The proxy only forwards requests to `testnet.binance.vision`
- Your API keys are sent directly from HuggingFace to Binance (through the proxy)
- The Worker doesn't log or store your API keys
- Only your HuggingFace Spaces can use the proxy (CORS protection)

---

## 💡 Optional: Custom Domain

If you want a custom domain instead of `.workers.dev`:

1. Add your domain to Cloudflare (requires domain ownership)
2. Go to Worker settings → Triggers → Custom Domains
3. Add your custom domain
4. Update the `BINANCE_TESTNET_PROXY_URL` environment variable

---

## ✅ Quick Start Summary

1. Sign up at workers.cloudflare.com
2. Create new Worker
3. Paste the code from `binance-testnet-proxy.js`
4. Save and Deploy
5. Copy your Worker URL
6. Add `BINANCE_TESTNET_PROXY_URL` to HuggingFace Secrets
7. Restart Space
8. Done! 🎉

---

**Questions?** Check the Cloudflare Workers docs: https://developers.cloudflare.com/workers/
