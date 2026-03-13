#!/usr/bin/env python3
"""
Debug: Why are ETH/SOL/XRP missing from the state P&L?
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
import json

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
try:
    import certifi
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
except ImportError:
    client = MongoClient(MONGO_URI)

db = client["trading_system"]
state_collection = db["state"]

print("=" * 100)
print("🔍 DEBUG: State Saving Logic")
print("=" * 100)

state = state_collection.find_one({"_id": "current_state"})

if state:
    state.pop('_id', None)
    print("\n📋 Full State Document:")
    print(json.dumps(state, indent=2))

    print("\n" + "=" * 100)
    print("🔍 Asset Analysis:")
    print("=" * 100)

    assets = state.get('assets', {})
    for asset_name, asset_data in sorted(assets.items()):
        print(f"\n{asset_name}:")
        print(json.dumps(asset_data, indent=2))
else:
    print("❌ No state found!")
