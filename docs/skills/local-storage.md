---
name: local-storage
description: Storage backend rules for the DRL trading system. Use when modifying, querying, or debugging trade storage, state persistence, or database connectivity. Covers SQLite (primary), MongoDB (cloud backup), and JSON (legacy fallback). Applies to any code touching src/data/storage.py, STORAGE_TYPE env var, or trade/state persistence.
---

# Local Storage — DRL Trading System

## Architecture

```
STORAGE_TYPE priority: sqlite (default) → mongo (cloud) → json (legacy)
```

- **SQLite** (`data/trading.db`): Primary. Zero network, WAL mode, thread-safe, indexed.
- **MongoDB Atlas**: Optional cloud sync. Set `STORAGE_TYPE=mongo` in `.env` when ready. Falls back to SQLite on connection failure.
- **JSON files** (`logs/`): Legacy. Only use if explicitly set `STORAGE_TYPE=json`.

## Storage Interface

All backends implement `StorageInterface` in `src/data/storage.py`:

```python
save_state(state: Dict)           # Upsert app state (single doc, key="current_state")
load_state() -> Dict              # Read current state
log_trade(trade: Dict)            # Append trade record
get_trades(limit=100) -> List     # Recent trades (oldest first)
```

SQLite extras (not on interface — cast to `SQLiteStorage` first):
```python
get_trades_by_symbol(symbol, limit=50)  # Filter by symbol
get_trade_stats() -> Dict               # Aggregate: total, wins, losses, pnl, avg, best, worst
```

## SQLite Schema

```sql
-- State: single key-value row
state(key TEXT PK, value TEXT, updated_at TEXT)

-- Trades: one row per open/close event
trades(id INTEGER PK, timestamp TEXT, symbol TEXT, action TEXT,
       price REAL, pnl REAL, confidence REAL, data TEXT, created_at TEXT)
-- Indexes: symbol, timestamp, action
```

## Rules

1. **Never use MongoDB as primary** until Atlas connectivity is verified and IP-whitelisted.
2. **Trade dict must include** at minimum: `timestamp`, `symbol`, `action`, `price`. Optional: `pnl`, `confidence`.
3. **`data` column** stores the full JSON blob — always pass the complete trade dict to `log_trade()`.
4. **Thread safety**: SQLite uses per-thread connections via `threading.local()`. Do not share connections across threads.
5. **WAL mode** is enabled — concurrent reads are safe. Writes serialize automatically.
6. **DB location**: `data/trading.db` (gitignored). Override with `SQLITE_DB_PATH` env var.
7. **Disk impact**: ~100 bytes per trade. 10K trades ≈ 1MB. Negligible vs model files (2.2GB).
8. **Querying from CLI**:
   ```bash
   sqlite3 data/trading.db "SELECT symbol, action, price, pnl FROM trades ORDER BY id DESC LIMIT 20"
   sqlite3 data/trading.db "SELECT symbol, COUNT(*), SUM(pnl), AVG(pnl) FROM trades WHERE pnl IS NOT NULL GROUP BY symbol"
   ```
9. **Migration to cloud**: When ready, set `STORAGE_TYPE=mongo` in `.env`, verify Atlas IP whitelist, restart bots. Mongo failure auto-falls back to SQLite.
10. **Backups**: `cp data/trading.db data/trading_backup_$(date +%Y%m%d).db`
