
import os
import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import pymongo

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class StorageInterface(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_state(self, state: Dict):
        """Save the current application state."""
        pass
    
    @abstractmethod
    def load_state(self) -> Dict:
        """Load the application state."""
        pass
    
    @abstractmethod
    def log_trade(self, trade: Dict):
        """Append a trade to the trade log."""
        pass
        
    @abstractmethod
    def get_trades(self, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        pass

class JsonFileStorage(StorageInterface):
    """Legacy storage using local JSON files."""
    
    def __init__(self, base_dir: Path = Path("logs")):
        self.base_dir = base_dir
        self.state_file = base_dir / "multi_asset_state.json"
        self.trade_file = base_dir / "trading_log.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_state(self, state: Dict):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state to file: {e}")
            
    def load_state(self) -> Dict:
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state from file: {e}")
            return {}
            
    def log_trade(self, trade: Dict):
        try:
            with open(self.trade_file, 'a') as f:
                f.write(json.dumps(trade) + '\n')
        except Exception as e:
            logger.error(f"Failed to log trade to file: {e}")
            
    def get_trades(self, limit: int = 100) -> List[Dict]:
        if not self.trade_file.exists():
            return []
        trades = []
        try:
            with open(self.trade_file, 'r') as f:
                for line in f:
                    if line.strip():
                        trades.append(json.loads(line))
            return trades[-limit:]
        except Exception as e:
            logger.error(f"Failed to load trades from file: {e}")
            return []

class SQLiteStorage(StorageInterface):
    """Local SQLite storage — zero network dependency, minimal RAM/disk."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("SQLITE_DB_PATH", "data/trading.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Init schema on main thread
        self._init_schema(self._get_conn())
        logger.info("💾 SQLite storage ready: %s", self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        """One connection per thread (SQLite is not thread-safe by default)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path), timeout=10,
                check_same_thread=False
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @staticmethod
    def _init_schema(conn: sqlite3.Connection):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trades (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                symbol     TEXT,
                action     TEXT,
                price      REAL,
                pnl        REAL,
                confidence REAL,
                data       TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_trades_symbol   ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_ts       ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_action   ON trades(action);
        """)

    def save_state(self, state: Dict):
        try:
            conn = self._get_conn()
            now = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO state (key, value, updated_at) VALUES (?, ?, ?)",
                ("current_state", json.dumps(state, default=str), now),
            )
            conn.commit()
        except Exception as e:
            logger.error("SQLite save_state failed: %s", e)

    def load_state(self) -> Dict:
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT value FROM state WHERE key = ?", ("current_state",)
            ).fetchone()
            return json.loads(row["value"]) if row else {}
        except Exception as e:
            logger.error("SQLite load_state failed: %s", e)
            return {}

    def log_trade(self, trade: Dict):
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO trades (timestamp, symbol, action, price, pnl, confidence, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(trade.get("timestamp", "")),
                    trade.get("symbol", ""),
                    trade.get("action", ""),
                    trade.get("price"),
                    trade.get("pnl") or trade.get("realized_pnl"),
                    trade.get("confidence"),
                    json.dumps(trade, default=str),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error("SQLite log_trade failed: %s", e)

    def get_trades(self, limit: int = 100) -> List[Dict]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT data FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            trades = [json.loads(r["data"]) for r in rows]
            return trades[::-1]  # oldest first
        except Exception as e:
            logger.error("SQLite get_trades failed: %s", e)
            return []

    # ── Extra query methods for the UI / API ──

    def get_trades_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict]:
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT data FROM trades WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
            return [json.loads(r["data"]) for r in rows][::-1]
        except Exception as e:
            logger.error("SQLite get_trades_by_symbol failed: %s", e)
            return []

    def get_trade_stats(self) -> Dict:
        """Quick aggregate stats for the dashboard."""
        try:
            conn = self._get_conn()
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades WHERE pnl IS NOT NULL
            """).fetchone()
            return dict(row) if row else {}
        except Exception as e:
            logger.error("SQLite get_trade_stats failed: %s", e)
            return {}


class MongoStorage(StorageInterface):
    """Cloud-native storage using MongoDB Atlas."""
    
    def __init__(self, connection_string: str = None):
        self.uri = connection_string or os.getenv("MONGO_URI")
        if not self.uri:
            raise ValueError("MONGO_URI environment variable is not set")

        # Create client with SSL certificate support
        try:
            import certifi
            self.client = pymongo.MongoClient(self.uri, tlsCAFile=certifi.where())
        except ImportError:
            self.client = pymongo.MongoClient(self.uri)

        # Use environment-specific database name to separate prod/dev data
        environment = os.getenv("ENVIRONMENT", "production").lower()
        db_name = "trading_system" if environment == "production" else f"trading_system_{environment}"

        try:
            self.db = self.client.get_database(db_name)
            self.state_collection = self.db.get_collection("state")
            self.trades_collection = self.db.get_collection("trades")
            # Verify connection
            self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB Atlas (database: {db_name}, environment: {environment})")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def save_state(self, state: Dict):
        try:
            # Upsert the single state document (ID='current_state')
            self.state_collection.update_one(
                {"_id": "current_state"},
                {"$set": state},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save state to MongoDB: {e}")
            
    def load_state(self) -> Dict:
        try:
            state = self.state_collection.find_one({"_id": "current_state"})
            if state:
                # Remove _id which is not part of the app state
                del state['_id']
                return state
            return {}
        except Exception as e:
            logger.error(f"Failed to load state from MongoDB: {e}")
            return {}
            
    def log_trade(self, trade: Dict):
        try:
            self.trades_collection.insert_one(trade)
        except Exception as e:
            logger.error(f"Failed to log trade to MongoDB: {e}")
            
    def get_trades(self, limit: int = 100) -> List[Dict]:
        try:
            cursor = self.trades_collection.find().sort("timestamp", -1).limit(limit)
            trades = list(cursor)
            # Remove _id and reverse to match file order (oldest first) ? 
            # Actually dashboard expects newest first usually but file reading was messy.
            # Let's clean up _id
            for t in trades:
                if '_id' in t:
                    del t['_id']
            return trades[::-1] # Return oldest first to match file behavior if needed, or check app usage
        except Exception as e:
            logger.error(f"Failed to fetch trades from MongoDB: {e}")
            return []

_storage_singleton: StorageInterface = None


def get_storage() -> StorageInterface:
    """Factory to get the configured storage backend (singleton).
    
    STORAGE_TYPE priority: sqlite > mongo > json
    - sqlite: local SQLite DB (default) — zero network, fast, queryable
    - mongo:  MongoDB Atlas — tries connect, falls back to sqlite
    - json:   legacy flat-file JSONL
    """
    global _storage_singleton
    if _storage_singleton is not None:
        return _storage_singleton

    storage_type = os.getenv("STORAGE_TYPE", "sqlite").lower()
    
    if storage_type == "sqlite":
        logger.info("💾 Using SQLite Storage")
        _storage_singleton = SQLiteStorage()
        return _storage_singleton
    elif storage_type == "mongo":
        try:
            logger.info("💽 Attempting MongoDB Storage...")
            _storage_singleton = MongoStorage()
            return _storage_singleton
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            logger.info("💾 Falling back to SQLite Storage")
            _storage_singleton = SQLiteStorage()
            return _storage_singleton
    else:
        logger.info("📁 Using Local JSON File Storage")
        _storage_singleton = JsonFileStorage()
        return _storage_singleton
