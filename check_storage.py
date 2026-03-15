#!/usr/bin/env python3
"""
Check which storage backend is being used and show current state.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from src.data.storage import get_storage

def check_storage():
    """Check storage backend and current state."""

    print("=" * 60)
    print("STORAGE DIAGNOSTICS")
    print("=" * 60)

    # Get storage instance
    storage = get_storage()
    storage_type = type(storage).__name__

    print(f"\n📊 Storage Backend: {storage_type}")

    if storage_type == "JsonFileStorage":
        print(f"   - State file: {storage.state_file}")
        print(f"   - Trade file: {storage.trade_file}")
        print(f"   - State exists: {storage.state_file.exists()}")
        print(f"   - Trades exist: {storage.trade_file.exists()}")

        if storage.state_file.exists():
            import json
            with open(storage.state_file) as f:
                state = json.load(f)
            print(f"\n📋 Current State (JSON):")
            print(f"   - Keys: {list(state.keys())}")
            if 'positions' in state:
                print(f"   - Positions: {state['positions']}")

    elif storage_type == "MongoStorage":
        env = os.getenv("ENVIRONMENT", "production").lower()
        db_name = "trading_system" if env == "production" else f"trading_system_{env}"
        print(f"   - Database: {db_name}")
        print(f"   - Environment: {env}")

        # Count documents
        state_count = storage.state_collection.count_documents({})
        trades_count = storage.trades_collection.count_documents({})

        print(f"\n📊 MongoDB State:")
        print(f"   - State documents: {state_count}")
        print(f"   - Trade documents: {trades_count}")

        if state_count > 0:
            state = storage.load_state()
            print(f"\n📋 Current State (MongoDB):")
            print(f"   - Keys: {list(state.keys())}")
            if 'positions' in state:
                print(f"   - Positions: {state['positions']}")

    # Load state via interface
    print("\n" + "=" * 60)
    print("CURRENT STATE VIA STORAGE INTERFACE")
    print("=" * 60)

    state = storage.load_state()

    if state:
        print(f"\n📋 Loaded state keys: {list(state.keys())}")

        if 'positions' in state:
            positions = state['positions']
            print(f"\n📊 Open Positions:")
            if positions:
                for asset, pos in positions.items():
                    if pos and isinstance(pos, dict) and pos.get('quantity', 0) != 0:
                        print(f"\n   {asset}:")
                        print(f"      Side: {pos.get('side', 'UNKNOWN')}")
                        print(f"      Quantity: {pos.get('quantity', 0)}")
                        print(f"      Entry Price: ${pos.get('entry_price', 0):.2f}")
                        print(f"      Entry Time: {pos.get('entry_time', 'UNKNOWN')}")
            else:
                print("   No open positions")

        if 'portfolio' in state:
            portfolio = state['portfolio']
            print(f"\n💰 Portfolio:")
            print(f"   - Balance: ${portfolio.get('balance', 0):.2f}")
            print(f"   - Equity: ${portfolio.get('equity', 0):.2f}")
    else:
        print("\n✅ State is empty - database clean!")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        check_storage()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
