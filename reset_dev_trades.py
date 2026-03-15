#!/usr/bin/env python3
"""
Reset Dev Environment Trading Database
Clears all trades for a fresh start while preserving schema.
"""

import sqlite3
import os
from datetime import datetime
import shutil

DB_PATH = "data/trading.db"
BACKUP_DIR = "data/backups"

def reset_trading_database():
    """Reset trading database to fresh state."""

    print("=" * 60)
    print("RESETTING DEV ENVIRONMENT TRADING DATABASE")
    print("=" * 60)

    # Create backup directory if it doesn't exist
    os.makedirs(BACKUP_DIR, exist_ok=True)

    # Backup existing database if it has data
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
        backup_name = f"trading_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = os.path.join(BACKUP_DIR, backup_name)
        shutil.copy2(DB_PATH, backup_path)
        print(f"✅ Backed up existing database to: {backup_path}")
    else:
        print("ℹ️  No existing data to backup")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    if tables:
        print(f"\n📊 Found {len(tables)} table(s): {', '.join(tables)}")

        # Count existing records before deletion
        total_records = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"   - {table}: {count} records")
                total_records += count

        if total_records > 0:
            print(f"\n🗑️  Clearing {total_records} total records...")

            # Clear all tables
            for table in tables:
                cursor.execute(f"DELETE FROM {table}")
                print(f"   ✅ Cleared {table}")

            # Reset auto-increment sequences
            cursor.execute("DELETE FROM sqlite_sequence")
            print("   ✅ Reset auto-increment sequences")

            conn.commit()
            print("\n✅ All trades cleared successfully!")
        else:
            print("\nℹ️  Database is already empty - no records to clear")
    else:
        print("\nℹ️  Database has no tables - creating fresh schema...")

        # Create trades table schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT,
                hold_time_bars INTEGER,
                entry_composite REAL,
                entry_confidence REAL,
                entry_whale_signal REAL,
                entry_regime TEXT,
                entry_tft_forecast REAL,
                entry_funding REAL,
                entry_order_flow REAL,
                status TEXT DEFAULT 'OPEN'
            )
        """)

        # Create positions table schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT UNIQUE NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_time TEXT NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                entry_composite REAL,
                entry_confidence REAL
            )
        """)

        conn.commit()
        print("✅ Created fresh database schema")

    # Verify database is now empty
    cursor.execute("SELECT COUNT(*) FROM trades") if 'trades' in tables or not tables else cursor.execute("SELECT 0")
    trade_count = cursor.fetchone()[0] if 'trades' in tables or not tables else 0

    conn.close()

    print("\n" + "=" * 60)
    print("RESET COMPLETE")
    print("=" * 60)
    print(f"📊 Current state:")
    print(f"   - Trades: {trade_count}")
    print(f"   - Positions: 0 (FLAT)")
    print(f"   - Database: {DB_PATH}")
    print("\n🎯 Dev environment ready for fresh testing!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        reset_trading_database()
    except Exception as e:
        print(f"\n❌ Error resetting database: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
