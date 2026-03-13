#!/usr/bin/env python3
"""
Create a dev Space on Hugging Face for testing changes from the dev branch.

This script:
1. Duplicates the production Space
2. Configures it with the same secrets
3. Sets up tracking for the dev branch
"""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

# Initialize API
api = HfApi(token=HF_TOKEN)

# Configuration
PROD_SPACE = "Chen4700/drl-trading-bot"
DEV_SPACE = "Chen4700/drl-trading-bot-dev"

print(f"🚀 Creating dev Space: {DEV_SPACE}")
print(f"📦 Duplicating from: {PROD_SPACE}")
print()

# Step 1: Duplicate the production Space
print("Step 1: Duplicating Space...")
try:
    # Get current space secrets (we'll need to re-add them)
    # Note: We can't read secret VALUES, but we know what keys exist

    # Duplicate the space
    api.duplicate_space(
        from_id=PROD_SPACE,
        to_id=DEV_SPACE,
        private=True,  # Keep dev space private
        hardware="cpu-basic",  # Use free tier for dev
    )
    print(f"✅ Space duplicated successfully: https://huggingface.co/spaces/{DEV_SPACE}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"⚠️  Space {DEV_SPACE} already exists")
    else:
        print(f"❌ Error duplicating Space: {e}")
        raise

# Step 2: Add necessary secrets to dev Space
print("\nStep 2: Configuring secrets...")
secrets_to_add = {
    "HF_TOKEN": HF_TOKEN,
    "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY", ""),
    "BINANCE_SECRET": os.getenv("BINANCE_SECRET", ""),
    "MONGO_URI": os.getenv("MONGO_URI", ""),
    "HELIUS_API_KEY": os.getenv("HELIUS_API_KEY", ""),
}

for key, value in secrets_to_add.items():
    if value:
        try:
            api.add_space_secret(repo_id=DEV_SPACE, key=key, value=value)
            print(f"  ✅ Added secret: {key}")
        except Exception as e:
            print(f"  ⚠️  Could not add {key}: {e}")
    else:
        print(f"  ⚠️  Skipping {key} (not found in .env)")

print("\n" + "="*60)
print("🎉 Dev Space Setup Complete!")
print("="*60)
print()
print(f"Production Space: https://huggingface.co/spaces/{PROD_SPACE}")
print(f"Dev Space:        https://huggingface.co/spaces/{DEV_SPACE}")
print()
print("Next Steps:")
print("1. The dev Space is now running (it will auto-deploy from main branch initially)")
print("2. To switch it to dev branch, push your dev branch to the Space:")
print()
print("   cd /Users/chenluigi/WebstormProjects/drl-trading-system")
print(f"   git remote add hf-dev https://huggingface.co/spaces/{DEV_SPACE}")
print("   git push hf-dev dev:main --force")
print()
print("   This pushes your local 'dev' branch to the Space's 'main' branch")
print()
print("3. The Space will automatically rebuild with your dev branch code")
print()
print("⚠️  IMPORTANT: The production Space remains untouched!")
print()
