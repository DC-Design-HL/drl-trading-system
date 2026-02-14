"""
DRL Trading System - Main Entry Point
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = './config/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    if config.get('exchange', {}).get('api_key', '').startswith('${'):
        config['exchange']['api_key'] = os.environ.get('BINANCE_TESTNET_API_KEY', '')
    if config.get('exchange', {}).get('api_secret', '').startswith('${'):
        config['exchange']['api_secret'] = os.environ.get('BINANCE_TESTNET_API_SECRET', '')
        
    return config


def run_training(config: dict, timesteps: int = 100000):
    """Train the DRL agent."""
    from src.env import CryptoTradingEnv
    from src.brain import TradingAgent
    from src.backtest import DataLoader
    
    logger.info("Starting training...")
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load(
        symbol=config['trading']['symbol'],
        timeframe=config['trading']['timeframe'],
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
    )
    
    # Create environment
    env = CryptoTradingEnv(
        df=df,
        initial_balance=config['trading']['initial_balance'],
        lookback_window=config['environment']['lookback_window'],
    )
    
    # Create agent
    agent = TradingAgent(
        env=env,
        config=config.get('model', {}),
    )
    
    # Train
    agent.train(
        total_timesteps=timesteps,
        save_path='./data/models/',
    )
    
    # Save final model
    agent.save('./data/models/trained_agent.zip')
    logger.info("Training complete!")
    
    return agent


def run_backtest(config: dict, model_path: str = None):
    """Run backtest on historical data."""
    from src.env import CryptoTradingEnv
    from src.brain import TradingAgent
    from src.backtest import BacktestEngine, DataLoader
    
    logger.info("Starting backtest...")
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load(
        symbol=config['trading']['symbol'],
        timeframe=config['trading']['timeframe'],
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
    )
    
    # Create environment
    env = CryptoTradingEnv(
        df=df,
        initial_balance=config['trading']['initial_balance'],
        lookback_window=config['environment']['lookback_window'],
    )
    
    # Create agent
    agent = TradingAgent(
        env=env,
        config=config.get('model', {}),
        model_path=model_path,
    )
    
    # Train if no model provided
    if model_path is None:
        logger.info("No model provided, training first...")
        agent.train(total_timesteps=50000)
    
    # Run backtest
    engine = BacktestEngine(config=config, data_loader=data_loader)
    results = engine.run(
        agent=agent,
        start_date=config['backtest']['start_date'],
        end_date=config['backtest']['end_date'],
    )
    
    # Generate report
    report = engine.generate_report(results, save_path='./data/backtest_report.txt')
    print(report)
    
    return results


def run_live_trading(config: dict, model_path: str, dry_run: bool = True):
    """Run live trading on Binance Testnet."""
    from src.env import CryptoTradingEnv
    from src.brain import TradingAgent, HighRewardBuffer, SelfImprovementTrainer
    from src.api import BinanceConnector, RiskManager, OrderExecutor
    from src.backtest import DataLoader
    import time
    
    logger.info("Starting live trading...")
    
    # Initialize Binance connector
    connector = BinanceConnector(
        api_key=config['exchange']['api_key'],
        api_secret=config['exchange']['api_secret'],
        testnet=config['exchange']['testnet'],
    )
    
    # Test connectivity
    if not connector.test_connectivity():
        logger.error("Failed to connect to Binance!")
        return
        
    logger.info("Connected to Binance Testnet")
    
    # Initialize risk manager
    risk_manager = RiskManager(
        initial_balance=config['trading']['initial_balance'],
        max_position_size=config['trading']['position_size'],
        stop_loss_pct=config['risk']['stop_loss_pct'],
        take_profit_pct=config['risk']['take_profit_pct'],
        max_daily_loss_pct=config['risk']['max_daily_loss_pct'],
        max_drawdown_pct=config['risk']['max_drawdown_pct'],
    )
    
    # Initialize order executor
    executor = OrderExecutor(
        connector=connector,
        risk_manager=risk_manager,
        symbol=config['trading']['symbol'],
        dry_run=dry_run,
    )
    
    # Load data for environment
    data_loader = DataLoader(connector=connector)
    df = connector.fetch_ohlcv(
        symbol=config['trading']['symbol'],
        timeframe=config['trading']['timeframe'],
        limit=500,
    )
    
    # Create environment
    env = CryptoTradingEnv(
        df=df,
        initial_balance=config['trading']['initial_balance'],
        lookback_window=config['environment']['lookback_window'],
    )
    
    # Load trained agent
    agent = TradingAgent(
        env=env,
        config=config.get('model', {}),
        model_path=model_path,
    )
    
    # Initialize replay buffer and trainer
    replay_buffer = HighRewardBuffer(
        save_path='./data/replay_buffer.pkl',
        reward_threshold=config['self_improvement']['reward_threshold'],
    )
    
    trainer = SelfImprovementTrainer(
        agent=agent,
        replay_buffer=replay_buffer,
        finetune_interval_hours=config['self_improvement']['finetune_interval_hours'],
        min_samples_for_finetune=config['self_improvement']['min_high_reward_samples'],
    )
    
    # Start self-improvement loop
    trainer.start()
    
    # Trading loop
    try:
        while True:
            # Get latest data
            df = connector.fetch_ohlcv(
                symbol=config['trading']['symbol'],
                timeframe=config['trading']['timeframe'],
                limit=100,
            )
            
            # Get current state
            current_price = float(df.iloc[-1]['close'])
            balance = connector.get_balance('USDT')
            
            # Check risk
            if not risk_manager.can_trade(balance):
                logger.warning("Trading paused by risk manager")
                time.sleep(60)
                continue
                
            # Get action from agent
            env.df = df
            obs, _ = env.reset()
            action, _, confidence = agent.predict(obs)
            
            logger.info(f"Action: {action}, Confidence: {confidence:.2%}, Price: ${current_price:,.2f}")
            
            # Execute
            order = executor.execute_signal(
                action=action,
                current_price=current_price,
                current_balance=balance,
            )
            
            if order:
                logger.info(f"Order executed: {order.to_dict()}")
                
            # Sleep until next candle
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Stopping trading...")
        trainer.stop()
        executor.close_all_positions(current_price)
        

def run_ui():
    """Launch the Streamlit dashboard."""
    import subprocess
    
    logger.info("Launching Streamlit dashboard...")
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        'src/ui/app.py',
        '--server.port=8501',
        '--server.headless=true',
    ])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='DRL Trading System')
    parser.add_argument('command', choices=['train', 'backtest', 'trade', 'ui'],
                       help='Command to run')
    parser.add_argument('--config', default='./config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps')
    parser.add_argument('--dry-run', action='store_true', default=True,
                       help='Dry run mode (no real orders)')
    
    args = parser.parse_args()
    
    # Create directories
    Path('./data/models').mkdir(parents=True, exist_ok=True)
    Path('./data/historical').mkdir(parents=True, exist_ok=True)
    Path('./logs').mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(args.config)
    
    # Run command
    if args.command == 'train':
        run_training(config, args.timesteps)
    elif args.command == 'backtest':
        run_backtest(config, args.model)
    elif args.command == 'trade':
        if not args.model:
            logger.error("Model path required for live trading!")
            sys.exit(1)
        run_live_trading(config, args.model, args.dry_run)
    elif args.command == 'ui':
        run_ui()


if __name__ == '__main__':
    main()
