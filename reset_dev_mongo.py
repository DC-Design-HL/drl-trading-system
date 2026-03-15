#!/usr/bin/env python3
"""
Reset Dev Environment - MongoDB Version
Clears all trades and positions from MongoDB for a fresh start.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import pymongo

# Load environment variables from .env file manually
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def reset_mongodb():
    """Reset MongoDB trading database to fresh state."""

    print("=" * 60)
    print("RESETTING DEV ENVIRONMENT - MONGODB")
    print("=" * 60)

    # Get MongoDB connection
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("❌ ERROR: MONGO_URI not found in environment variables")
        return False

    # Get environment
    environment = os.getenv("ENVIRONMENT", "production").lower()
    db_name = "trading_system" if environment == "production" else f"trading_system_{environment}"

    print(f"\n📊 Configuration:")
    print(f"   - Environment: {environment}")
    print(f"   - Database: {db_name}")
    print(f"   - URI: {mongo_uri[:50]}...")

    try:
        # Connect to MongoDB
        try:
            import certifi
            client = pymongo.MongoClient(mongo_uri, tlsCAFile=certifi.where())
        except ImportError:
            client = pymongo.MongoClient(mongo_uri)

        # Verify connection
        client.admin.command('ping')
        print(f"\n✅ Connected to MongoDB successfully")

        # Get database
        db = client.get_database(db_name)
        state_collection = db.get_collection("state")
        trades_collection = db.get_collection("trades")

        # Count existing records
        state_count = state_collection.count_documents({})
        trades_count = trades_collection.count_documents({})

        print(f"\n📊 Current state:")
        print(f"   - State documents: {state_count}")
        print(f"   - Trade documents: {trades_count}")

        if state_count == 0 and trades_count == 0:
            print("\nℹ️  Database is already empty - nothing to reset")
            return True

        # Get current state to show what positions will be cleared
        current_state = state_collection.find_one({"_id": "current_state"})
        if current_state:
            print(f"\n📋 Current positions to be cleared:")
            positions = current_state.get("positions", {})
            if positions:
                for asset, pos in positions.items():
                    if pos and pos.get("quantity", 0) != 0:
                        print(f"   - {asset}: {pos.get('side', 'UNKNOWN')} {pos.get('quantity', 0)} @ ${pos.get('entry_price', 0):.2f}")
            else:
                print("   - No open positions")

        # Backup current state (optional)
        backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = project_root / f"data/backups/mongodb_backup_{backup_time}.json"
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        if trades_count > 0:
            import json
            trades = list(trades_collection.find())
            # Remove MongoDB _id for JSON serialization
            for trade in trades:
                if '_id' in trade:
                    del trade['_id']
            with open(backup_file, 'w') as f:
                json.dump({
                    'state': current_state if current_state else {},
                    'trades': trades,
                    'backup_time': backup_time,
                    'environment': environment
                }, f, indent=2, default=str)
            print(f"\n💾 Backed up {trades_count} trades to: {backup_file}")

        # Clear collections
        print(f"\n🗑️  Clearing database...")

        # Clear trades
        if trades_count > 0:
            result = trades_collection.delete_many({})
            print(f"   ✅ Deleted {result.deleted_count} trades")

        # Clear state
        if state_count > 0:
            result = state_collection.delete_many({})
            print(f"   ✅ Deleted {result.deleted_count} state documents")

        # Verify deletion
        state_count_after = state_collection.count_documents({})
        trades_count_after = trades_collection.count_documents({})

        print("\n" + "=" * 60)
        print("RESET COMPLETE")
        print("=" * 60)
        print(f"📊 Final state:")
        print(f"   - State documents: {state_count_after}")
        print(f"   - Trade documents: {trades_count_after}")
        print(f"   - Open positions: 0 (FLAT)")
        print(f"   - Database: {db_name}")
        print("\n🎯 Dev environment ready for fresh testing!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error resetting MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = reset_mongodb()
    sys.exit(0 if success else 1)
