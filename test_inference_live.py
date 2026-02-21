import pandas as pd
import logging
from src.models.confidence_engine import ConfidenceEngine
from live_trading_multi import MultiAssetTradingBot

logging.basicConfig(level=logging.INFO)

def test_inference():
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    print("====================================")
    print(" LIVE AI INFERENCE DIAGNOSTICS")
    print("====================================\n")
    
    for symbol in assets:
        print(f"\n--- INITIATING DIAGNOSTIC FOR {symbol} ---")
        bot = MultiAssetTradingBot(symbol, dry_run=True, initial_balance=10000)
        
        print("Fetching data...")
        df = bot.fetch_data(days=14)
    
        if df.empty:
            print("Error: No data fetched")
            continue
            
        print(f"Data fetched: {len(df)} rows. Current price: {df.iloc[-1]['close']}")
        
        # Force the agent to make a prediction
        raw_action_int = bot.get_action(df)
        
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        raw_action_str = action_map.get(raw_action_int, "HOLD")
        
        print(f"\n--- INFERENCE RESULTS ---")
        print(f"Raw Action: {raw_action_str}")
        print(f"Confidence (0-1): {bot.last_confidence:.4f}")
        
        # Test Confidence Scaling limits
        print("\n--- CONFIDENCE SCALING SIMULATION ---")
        engine = ConfidenceEngine()
        
        base_trade_value = 10000 * 0.5  # $5000 base
        print(f"Base Position Size: ${base_trade_value:.2f}")
        
        # Try actual confidence
        actual_scaled = engine.apply_confidence(base_trade_value, bot.last_confidence)
        print(f"Scaled size for actual confidence ({bot.last_confidence:.4f}): ${actual_scaled:.2f}")
        
        # Try low confidence
        low_scaled = engine.apply_confidence(base_trade_value, 0.20)
        print(f"Scaled size for low confidence (0.20): ${low_scaled:.2f}")

        print("\n--- RUNNING FULL ITERATION ---")
        res = bot.run_iteration()
        print("Iteration result keys:", res.keys())
        print(f"Filtered Action: {res.get('filtered_action')}")
        print(f"Reason: {res.get('reason')}")
        print(f"Units Traded: {res.get('trade', {}).get('units', 0) if res.get('trade') else 0}")
    
if __name__ == "__main__":
    test_inference()
