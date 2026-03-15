#!/usr/bin/env python3
"""
Universal Reset Script - Clears BOTH MongoDB AND JSON files
"""

import os
import sys
import json
from datetime import datetime
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

def reset_json_files():
    """Reset JSON file storage."""
    print("\n" + "=" * 60)
    print("RESETTING JSON FILE STORAGE")
    print("=" * 60)

    logs_dir = project_root / "logs"
    state_file = logs_dir / "multi_asset_state.json"
    trade_file = logs_dir / "trading_log.json"

    # Backup existing files
    backup_dir = project_root / "data/backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    backed_up = []

    if state_file.exists():
        backup_path = backup_dir / f"state_backup_{backup_time}.json"
        import shutil
        shutil.copy2(state_file, backup_path)
        backed_up.append(f"State: {backup_path}")

        # Delete state file
        state_file.unlink()
        print(f"✅ Deleted: {state_file}")

    if trade_file.exists():
        backup_path = backup_dir / f"trades_backup_{backup_time}.json"
        import shutil
        shutil.copy2(trade_file, backup_path)
        backed_up.append(f"Trades: {backup_path}")

        # Delete trade file
        trade_file.unlink()
        print(f"✅ Deleted: {trade_file}")

    if backed_up:
        print(f"\n💾 Backups saved:")
        for backup in backed_up:
            print(f"   - {backup}")
    else:
        print("\nℹ️  No JSON files to clear")

def reset_mongodb():
    """Reset MongoDB storage."""
    print("\n" + "=" * 60)
    print("RESETTING MONGODB STORAGE")
    print("=" * 60)

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("⚠️  MONGO_URI not set - skipping MongoDB reset")
        return

    try:
        import pymongo

        # Connect to MongoDB
        try:
            import certifi
            client = pymongo.MongoClient(mongo_uri, tlsCAFile=certifi.where())
        except ImportError:
            client = pymongo.MongoClient(mongo_uri)

        # Verify connection
        client.admin.command('ping')

        # Get environment and database
        environment = os.getenv("ENVIRONMENT", "production").lower()
        db_name = "trading_system" if environment == "production" else f"trading_system_{environment}"

        print(f"\n📊 MongoDB Config:")
        print(f"   - Environment: {environment}")
        print(f"   - Database: {db_name}")

        # Get database and collections
        db = client.get_database(db_name)
        state_collection = db.get_collection("state")
        trades_collection = db.get_collection("trades")

        # Count documents
        state_count = state_collection.count_documents({})
        trades_count = trades_collection.count_documents({})

        print(f"   - State documents: {state_count}")
        print(f"   - Trade documents: {trades_count}")

        if state_count > 0 or trades_count > 0:
            # Clear collections
            if state_count > 0:
                result = state_collection.delete_many({})
                print(f"\n✅ Deleted {result.deleted_count} state documents")

            if trades_count > 0:
                result = trades_collection.delete_many({})
                print(f"✅ Deleted {result.deleted_count} trade documents")
        else:
            print("\nℹ️  MongoDB already empty")

    except ImportError:
        print("⚠️  pymongo not installed - skipping MongoDB reset")
    except Exception as e:
        print(f"⚠️  MongoDB reset failed: {e}")

def main():
    """Run complete reset."""
    print("=" * 60)
    print("UNIVERSAL STORAGE RESET")
    print("=" * 60)
    print("\nThis will clear ALL trading data:")
    print("  - JSON files (logs/multi_asset_state.json, logs/trading_log.json)")
    print("  - MongoDB collections (state, trades)")
    print("\n💾 All data will be backed up before deletion")

    # Reset JSON files
    reset_json_files()

    # Reset MongoDB
    reset_mongodb()

    # Final summary
    print("\n" + "=" * 60)
    print("RESET COMPLETE")
    print("=" * 60)
    print("\n✅ All storage backends cleared")
    print("✅ Backups saved to data/backups/")
    print("✅ Database ready for fresh testing")
    print("\n🎯 Please restart/refresh your application!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
