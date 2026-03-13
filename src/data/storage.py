
import os
import json
import logging
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

def get_storage() -> StorageInterface:
    """Factory to get the configured storage backend."""
    storage_type = os.getenv("STORAGE_TYPE", "json").lower()
    
    if storage_type == "mongo":
        try:
            logger.info("💽 Attempting MongoDB Storage...")
            return MongoStorage()
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
            logger.info("📁 Falling back to Local JSON File Storage")
            return JsonFileStorage()
    else:
        logger.info("📁 Using Local JSON File Storage")
        return JsonFileStorage()
