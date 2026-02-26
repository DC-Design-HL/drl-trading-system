import re

with open("live_trading_multi.py", "r") as f:
    content = f.read()

# 1. Add tracking variables to TradingBot class
init_replacement = '''        self.sl_price = 0.0
        self.tp_price = 0.0
        self.highest_price = 0.0
        self.lowest_price = 0.0
        self.base_trailing_pct = 0.0'''

content = re.sub(
    r'        self\.sl_price = 0\.0\n        self\.tp_price = 0\.0',
    init_replacement,
    content
)

# 2. Add Stop Loss / Take profit tracking to execute_trade
execute_long_replacement = '''                # Calculate SL/TP
                try:
                    df = self.fetch_data(days=3) # Need some history for ATR
                    sl_pct, tp_pct = self.risk_manager.get_adaptive_sl_tp(df, "long")
                    self.sl_price = current_price * (1 - sl_pct)
                    self.tp_price = current_price * (1 + tp_pct)
                    self.highest_price = current_price
                    self.base_trailing_pct = sl_pct # Use SL distance as trailing distance
                    trade["sl"] = self.sl_price
                    trade["tp"] = self.tp_price
                    logger.info(f"🛡️ LONG SL: ${self.sl_price:.2f} (-{sl_pct:.2%}) | TP: ${self.tp_price:.2f} (+{tp_pct:.2%})")
                except Exception as e:
                    logger.error(f"Failed to calc SL/TP: {e}")'''

content = re.sub(
    r'                # Calculate SL/TP\n                try:.*?logger\.error\(f"Failed to calc SL/TP: {e}"\)',
    execute_long_replacement,
    content,
    flags=re.DOTALL,
    count=1
)

execute_short_replacement = '''                # Calculate SL/TP
                try:
                    df = self.fetch_data(days=3)
                    sl_pct, tp_pct = self.risk_manager.get_adaptive_sl_tp(df, "short")
                    self.sl_price = current_price * (1 + sl_pct)
                    self.tp_price = current_price * (1 - tp_pct)
                    self.lowest_price = current_price
                    self.base_trailing_pct = sl_pct
                    trade["sl"] = self.sl_price
                    trade["tp"] = self.tp_price
                    logger.info(f"🛡️ SHORT SL: ${self.sl_price:.2f} (-{sl_pct:.2%}) | TP: ${self.tp_price:.2f} (+{tp_pct:.2%})")
                except Exception as e:
                    logger.error(f"Failed to calc SL/TP: {e}")'''

content = re.sub(
    r'                # Calculate SL/TP\n                try:.*?logger\.error\(f"Failed to calc SL/TP: {e}"\)',
    execute_short_replacement,
    content,
    flags=re.DOTALL,
    count=1  # Only second match (which is actually first after the sub above)
)


# 3. Add trailing stop logic to run_iteration
run_iteration_replacement = '''        # Check SL/TP Hits first
        if self.position != 0:
            hit_sl = False
            hit_tp = False
            hit_trailing = False
            reason = ""
            
            if self.position == 1: # LONG
                self.highest_price = max(self.highest_price, current_price)
                
                # Check normal SL/TP
                if current_price <= self.sl_price and self.sl_price > 0:
                    hit_sl = True
                    reason = "STOP_LOSS"
                elif current_price >= self.tp_price and self.tp_price > 0:
                    hit_tp = True
                    reason = "TAKE_PROFIT"
                else:
                    # Check Trailing Stop
                    trailing_stop = self.highest_price * (1 - self.base_trailing_pct)
                    if trailing_stop > self.sl_price: # Trailing Stop only moves UP
                        if current_price <= trailing_stop:
                            hit_trailing = True
                            reason = "TRAILING_STOP"
                            
                    # Break-Even Stop (moved halfway to TP)
                    tp_distance = self.tp_price - self.position_price
                    if current_price >= self.position_price + (tp_distance * 0.5):
                        if self.sl_price < self.position_price:
                            self.sl_price = self.position_price # Move SL to Break-Even
                            logger.info(f"🛡️ Moved SL to Break-Even for {self.symbol} @ ${self.sl_price:.2f}")

                if hit_sl or hit_tp or hit_trailing:
                    trade = self.execute_trade(2, current_price) # Sell to close
                    if hit_sl:
                        self.last_loss_time = time.time()
                    logger.info(f"🛑 {reason} triggered for {self.symbol} @ ${current_price:.2f}")
                    
                    # Return result immediately
                    # Calculate unrealized (now 0)
                    total_equity = self.balance
                    self.last_equity = total_equity
                    
                    return {
                        "symbol": self.symbol,
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "raw_action": "HOLD",
                        "filtered_action": "CLOSE_LONG",
                        "reason": reason,
                        "position": 0,
                        "balance": self.balance,
                        "equity": total_equity,
                        "realized_pnl": self.realized_pnl,
                        "unrealized_pnl": 0,
                        "trade": trade,
                        "sl": 0,
                        "tp": 0
                    }

            elif self.position == -1: # SHORT
                self.lowest_price = min(self.lowest_price, current_price) if self.lowest_price > 0 else current_price
                
                if current_price >= self.sl_price and self.sl_price > 0:
                    hit_sl = True
                    reason = "STOP_LOSS"
                elif current_price <= self.tp_price and self.tp_price > 0:
                    hit_tp = True
                    reason = "TAKE_PROFIT"
                else:
                    # Check Trailing Stop
                    trailing_stop = self.lowest_price * (1 + self.base_trailing_pct)
                    if trailing_stop < self.sl_price and self.sl_price > 0: # Trailing Stop only moves DOWN
                        if current_price >= trailing_stop:
                            hit_trailing = True
                            reason = "TRAILING_STOP"
                            
                    # Break-Even Stop (moved halfway to TP)
                    tp_distance = self.position_price - self.tp_price
                    if current_price <= self.position_price - (tp_distance * 0.5):
                        if self.sl_price > self.position_price:
                            self.sl_price = self.position_price # Move SL to Break-Even
                            logger.info(f"🛡️ Moved SL to Break-Even for {self.symbol} @ ${self.sl_price:.2f}")
                    
                if hit_sl or hit_tp or hit_trailing:
                    trade = self.execute_trade(1, current_price) # Buy to close
                    if hit_sl:
                        self.last_loss_time = time.time()
                    logger.info(f"🛑 {reason} triggered for {self.symbol} @ ${current_price:.2f}")
                    
                    # Return result immediately
                    total_equity = self.balance
                    self.last_equity = total_equity
                    
                    return {
                        "symbol": self.symbol,
                        "timestamp": datetime.now().isoformat(),
                        "price": current_price,
                        "raw_action": "HOLD",
                        "filtered_action": "CLOSE_SHORT",
                        "reason": reason,
                        "position": 0,
                        "balance": self.balance,
                        "equity": total_equity,
                        "realized_pnl": self.realized_pnl,
                        "unrealized_pnl": 0,
                        "trade": trade,
                        "sl": 0,
                        "tp": 0
                    }'''

content = re.sub(
    r'        # Check SL/TP Hits first\n        if self\.position != 0:.*?                    }\n',
    run_iteration_replacement + '\n',
    content,
    flags=re.DOTALL
)

with open("live_trading_multi.py", "w") as f:
    f.write(content)

