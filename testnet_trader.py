#!/usr/bin/env python3
"""
Binance Testnet Trading Bot
Executes real trades on Binance testnet using the trained PPO model.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from src.api.binance import BinanceConnector
from src.features.ultimate_feature_engine import UltimateFeatureEngine
from src.features.regime_detector import MarketRegimeDetector
from src.features.risk_manager import AdaptiveRiskManager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/testnet_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestnetTradingBot:
    """
    Testnet trading bot that executes real trades using PPO model decisions.

    Features:
    - Real order execution on Binance testnet
    - PPO model for decision making
    - SL/TP management with trailing stops
    - Position tracking and P&L calculation
    - Portfolio risk management
    """

    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        initial_balance: float = 10000.0,
        model_path: str = 'data/models/ultimate_agent.zip',
        vec_normalize_path: str = 'data/models/ultimate_agent_vec_normalize.pkl',
    ):
        """Initialize testnet trading bot."""
        self.symbol = symbol
        self.symbol_ccxt = symbol.replace('/', '')  # BTCUSDT for binance
        self.initial_balance = initial_balance

        # Initialize Binance connector
        logger.info("🔗 Connecting to Binance testnet...")
        self.exchange = BinanceConnector(
            api_key=os.getenv('BINANCE_TESTNET_API_KEY'),
            api_secret=os.getenv('BINANCE_TESTNET_API_SECRET'),
            testnet=True,
        )

        # Test connectivity
        if not self.exchange.test_connectivity():
            raise ConnectionError("Failed to connect to Binance testnet")
        logger.info("✅ Connected to Binance testnet")

        # Initialize feature engine
        logger.info("🧠 Initializing feature engines...")
        self.feature_engine = UltimateFeatureEngine()
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = AdaptiveRiskManager()

        # Load PPO model
        logger.info(f"📦 Loading PPO model from {model_path}...")
        self.model = PPO.load(model_path)

        # Load VecNormalize stats
        if Path(vec_normalize_path).exists():
            logger.info(f"📊 Loading VecNormalize from {vec_normalize_path}...")
            self.vec_normalize = VecNormalize.load(vec_normalize_path, DummyVecEnv([lambda: None]))
        else:
            logger.warning("⚠️ VecNormalize not found - using raw observations")
            self.vec_normalize = None

        # Trading state
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.position_size = 0.0  # Amount in base currency
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.highest_price = 0.0
        self.lowest_price = 0.0
        self.entry_time = 0

        logger.info("✅ Testnet trading bot initialized")

    def get_account_balance(self) -> Dict[str, float]:
        """Get current account balance."""
        try:
            balances = self.exchange.get_all_balances()

            # Extract USDT and base currency
            base_currency = self.symbol.split('/')[0]

            return {
                'USDT': balances.get('USDT', {}).get('free', 0.0),
                base_currency: balances.get(base_currency, {}).get('free', 0.0),
                'total_balances': balances,
            }
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return {'USDT': 0.0, self.symbol.split('/')[0]: 0.0}

    def get_current_price(self) -> float:
        """Get current market price."""
        ticker = self.exchange.get_ticker(self.symbol)
        return float(ticker.get('last', 0))

    def calculate_position_size(self, current_price: float, balance: float) -> float:
        """Calculate position size based on available balance."""
        # Use 95% of balance for position (keep 5% for fees)
        usable_balance = balance * 0.95

        # Calculate amount in base currency
        amount = usable_balance / current_price

        # Round to appropriate precision (e.g., 6 decimals for BTC)
        amount = round(amount, 6)

        logger.info(f"💰 Position size: {amount} {self.symbol.split('/')[0]} (${usable_balance:.2f})")
        return amount

    def execute_trade(self, action: int, current_price: float) -> Optional[Dict]:
        """Execute trade based on PPO action."""
        try:
            base_currency = self.symbol.split('/')[0]

            if action == 1:  # BUY/LONG
                if self.position == 1:
                    logger.info("Already in LONG position - skipping")
                    return None

                # Close SHORT if exists
                if self.position == -1:
                    logger.info("Closing SHORT position before opening LONG")
                    self.exchange.place_market_order(
                        symbol=self.symbol,
                        side='buy',
                        amount=self.position_size,
                    )
                    self.position = 0

                # Get balance and calculate position size
                balance = self.get_account_balance()
                usdt_balance = float(balance['USDT'])

                if usdt_balance < 10:
                    logger.warning(f"Insufficient balance: ${usdt_balance:.2f}")
                    return None

                position_size = self.calculate_position_size(current_price, usdt_balance)

                # Place BUY order
                order = self.exchange.place_market_order(
                    symbol=self.symbol,
                    side='buy',
                    amount=position_size,
                )

                if order:
                    self.position = 1
                    self.position_size = position_size
                    self.entry_price = current_price
                    self.highest_price = current_price
                    self.entry_time = time.time()

                    # Calculate SL/TP
                    sl_pct, tp_pct = self.risk_manager.get_asset_specific_params(self.symbol_ccxt)
                    self.sl_price = current_price * (1 - sl_pct)
                    self.tp_price = current_price * (1 + tp_pct)

                    logger.info(f"🟢 LONG opened @ ${current_price:.2f} | SL: ${self.sl_price:.2f} | TP: ${self.tp_price:.2f}")
                    return order

            elif action == 2:  # SELL/SHORT
                if self.position == -1:
                    logger.info("Already in SHORT position - skipping")
                    return None

                # Close LONG if exists
                if self.position == 1:
                    logger.info("Closing LONG position before opening SHORT")
                    self.exchange.place_market_order(
                        symbol=self.symbol,
                        side='sell',
                        amount=self.position_size,
                    )
                    self.position = 0

                # For SHORT in spot: Need to have base currency
                balance = self.get_account_balance()
                base_balance = float(balance[base_currency])

                if base_balance < position_size:
                    logger.warning(f"Insufficient {base_currency} for SHORT: {base_balance}")
                    return None

                # Place SELL order
                order = self.exchange.place_market_order(
                    symbol=self.symbol,
                    side='sell',
                    amount=base_balance,
                )

                if order:
                    self.position = -1
                    self.position_size = base_balance
                    self.entry_price = current_price
                    self.lowest_price = current_price
                    self.entry_time = time.time()

                    # Calculate SL/TP
                    sl_pct, tp_pct = self.risk_manager.get_asset_specific_params(self.symbol_ccxt)
                    self.sl_price = current_price * (1 + sl_pct)
                    self.tp_price = current_price * (1 - tp_pct)

                    logger.info(f"🔴 SHORT opened @ ${current_price:.2f} | SL: ${self.sl_price:.2f} | TP: ${self.tp_price:.2f}")
                    return order

            else:  # HOLD
                return None

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None

    def check_sl_tp(self, current_price: float) -> Optional[str]:
        """Check if SL or TP hit."""
        if self.position == 0:
            return None

        # Update highest/lowest
        if self.position == 1:
            self.highest_price = max(self.highest_price, current_price)

            # Trailing SL
            if self.highest_price > self.entry_price:
                gain = self.highest_price - self.entry_price
                gain_pct = gain / self.entry_price

                if gain_pct > 0.015:  # 1.5% gain
                    new_sl = self.entry_price + gain * 0.6
                    if new_sl > self.sl_price:
                        logger.info(f"📈 Trailing SL: ${self.sl_price:.2f} → ${new_sl:.2f}")
                        self.sl_price = new_sl

            # Check SL/TP
            if current_price <= self.sl_price:
                return "STOP_LOSS"
            elif current_price >= self.tp_price:
                return "TAKE_PROFIT"

        elif self.position == -1:
            self.lowest_price = min(self.lowest_price, current_price) if self.lowest_price > 0 else current_price

            # Trailing SL
            if self.lowest_price < self.entry_price:
                gain = self.entry_price - self.lowest_price
                gain_pct = gain / self.entry_price

                if gain_pct > 0.015:
                    new_sl = self.entry_price - gain * 0.6
                    if new_sl < self.sl_price:
                        logger.info(f"📉 Trailing SL: ${self.sl_price:.2f} → ${new_sl:.2f}")
                        self.sl_price = new_sl

            # Check SL/TP
            if current_price >= self.sl_price:
                return "STOP_LOSS"
            elif current_price <= self.tp_price:
                return "TAKE_PROFIT"

        return None

    def close_position(self, reason: str, current_price: float):
        """Close current position."""
        if self.position == 0:
            return

        try:
            side = 'sell' if self.position == 1 else 'buy'

            order = self.exchange.place_market_order(
                symbol=self.symbol,
                side=side,
                amount=self.position_size,
            )

            if order:
                # Calculate P&L
                if self.position == 1:
                    pnl = (current_price - self.entry_price) * self.position_size
                else:
                    pnl = (self.entry_price - current_price) * self.position_size

                pnl_pct = (pnl / (self.entry_price * self.position_size)) * 100

                logger.info(
                    f"🛑 {reason}: Closed {['SHORT', 'FLAT', 'LONG'][self.position+1]} "
                    f"@ ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"
                )

                # Reset position
                self.position = 0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.sl_price = 0.0
                self.tp_price = 0.0

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def run_single_iteration(self) -> Dict:
        """Run one trading iteration."""
        try:
            # Get current price
            current_price = self.get_current_price()

            # Check SL/TP first
            if self.position != 0:
                exit_reason = self.check_sl_tp(current_price)
                if exit_reason:
                    self.close_position(exit_reason, current_price)

            # Get market data
            df = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe='1h',
                limit=100,
            )

            # Generate features
            features = self.feature_engine.generate_features(df, self.symbol_ccxt)

            # Get observation (last row)
            obs = features.iloc[-1].values.astype(np.float32)

            # Normalize if available
            if self.vec_normalize:
                obs = self.vec_normalize.normalize_obs(obs)

            # Get PPO action
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(action)

            # Map action: 0=HOLD, 1=BUY, 2=SELL
            action_name = ['HOLD', 'BUY', 'SELL'][action]

            # Execute trade if needed
            if action != 0:
                self.execute_trade(action, current_price)

            # Get account status
            balance = self.get_account_balance()

            return {
                'timestamp': datetime.now().isoformat(),
                'price': current_price,
                'action': action_name,
                'position': self.position,
                'entry_price': self.entry_price,
                'sl': self.sl_price,
                'tp': self.tp_price,
                'balance_usdt': balance['USDT'],
                'balance_base': balance[self.symbol.split('/')[0]],
            }

        except Exception as e:
            logger.error(f"Error in trading iteration: {e}")
            return {}

    def run(self, interval_seconds: int = 300):
        """Run trading bot continuously."""
        logger.info(f"🚀 Starting testnet trading bot for {self.symbol}")
        logger.info(f"📊 Trading interval: {interval_seconds}s")

        while True:
            try:
                result = self.run_single_iteration()
                if result:
                    logger.info(
                        f"💹 {result['action']} | Price: ${result['price']:.2f} | "
                        f"Position: {['SHORT', 'FLAT', 'LONG'][result['position']+1]} | "
                        f"Balance: ${result['balance_usdt']:.2f}"
                    )

                # Wait for next iteration
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("🛑 Stopping trading bot...")
                if self.position != 0:
                    current_price = self.get_current_price()
                    self.close_position("MANUAL_STOP", current_price)
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Binance Testnet Trading Bot')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--interval', type=int, default=300, help='Trading interval in seconds')
    args = parser.parse_args()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Initialize and run bot
    bot = TestnetTradingBot(symbol=args.symbol)
    bot.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
