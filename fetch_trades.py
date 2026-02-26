import os
import pymongo
from dotenv import load_dotenv
import pandas as pd
import certifi

load_dotenv()
uri = os.getenv("MONGO_URI")

client = pymongo.MongoClient(uri, tlsCAFile=certifi.where())
db = client.get_database("trading_system")
trades_col = db.get_collection("trades")

trades = list(trades_col.find({"pnl": {"$exists": True}}).sort("timestamp", -1).limit(50))

if not trades:
    print("No trades found in MongoDB.")
else:
    df = pd.DataFrame(trades)
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    
    # Calculate win rate and metrics
    if 'pnl' in df.columns:
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        total_finished = len(wins) + len(losses)
        win_rate = len(wins) / total_finished if total_finished > 0 else 0
        total_pnl = df['pnl'].sum()
        
        print(f"Total Completed Trades: {total_finished}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total PnL (last 50): ${total_pnl:.2f}")
        
    print("\nRecent Trades:")
    # Print columns nicely
    cols_to_print = ['timestamp', 'symbol', 'action', 'price', 'pnl', 'reason']
    available_cols = [c for c in cols_to_print if c in df.columns]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df[available_cols].head(50).to_string())
