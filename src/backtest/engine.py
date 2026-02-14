"""
Backtest Engine
Runs backtests on historical data and generates performance reports.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging

from src.env import CryptoTradingEnv, TechnicalIndicators
from src.brain import TradingAgent
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    
    Runs the DRL agent on historical data and calculates
    performance metrics including Sharpe ratio.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        data_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize the backtest engine.
        
        Args:
            config: Configuration dictionary
            data_loader: Data loader instance
        """
        self.config = config or {}
        self.data_loader = data_loader or DataLoader()
        
        # Default backtest parameters
        self.symbol = self.config.get('symbol', 'BTC/USDT')
        self.timeframe = self.config.get('timeframe', '1h')
        self.initial_balance = self.config.get('initial_balance', 10000.0)
        
        # Results storage
        self.results: List[Dict] = []
        
    def run(
        self,
        agent: TradingAgent,
        start_date: str = '2024-01-01',
        end_date: str = '2025-01-01',
        episodes: int = 1,
    ) -> Dict[str, Any]:
        """
        Run a backtest.
        
        Args:
            agent: Trained trading agent
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            episodes: Number of episodes to run
            
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        # Load data
        df = self.data_loader.load(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date,
        )
        
        if len(df) < 100:
            raise ValueError(f"Insufficient data: {len(df)} candles")
            
        logger.info(f"Loaded {len(df)} candles for backtesting")
        
        # Create environment
        env = CryptoTradingEnv(
            df=df,
            initial_balance=self.initial_balance,
            lookback_window=self.config.get('lookback_window', 30),
        )
        
        # Run episodes
        episode_results = []
        
        for episode in range(episodes):
            logger.info(f"Running episode {episode + 1}/{episodes}")
            result = self._run_episode(agent, env)
            episode_results.append(result)
            
        # Aggregate results
        aggregated = self._aggregate_results(episode_results)
        
        # Store results
        self.results.append({
            'timestamp': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'episodes': episodes,
            **aggregated,
        })
        
        return aggregated
        
    def _run_episode(
        self,
        agent: TradingAgent,
        env: CryptoTradingEnv,
    ) -> Dict[str, Any]:
        """Run a single backtest episode."""
        obs, info = env.reset()
        
        total_reward = 0
        step = 0
        done = False
        
        # State for LSTM
        lstm_state = None
        
        # Tracking
        actions_taken = []
        rewards = []
        portfolio_values = []
        
        while not done:
            # Get action from agent
            action, lstm_state, confidence = agent.predict(
                obs,
                state=lstm_state,
                deterministic=True,
            )
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track
            total_reward += reward
            actions_taken.append(action)
            rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
            
            step += 1
            
        # Get episode metrics
        episode_metrics = env.get_episode_metrics()
        
        # Calculate additional metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else [0]
        
        return {
            'total_reward': total_reward,
            'steps': step,
            'final_balance': info['balance'],
            'final_portfolio_value': info['portfolio_value'],
            'total_return': (info['portfolio_value'] - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': episode_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': episode_metrics.get('sortino_ratio', 0),
            'max_drawdown': episode_metrics.get('max_drawdown', 0),
            'trade_count': info['trade_count'],
            'actions': actions_taken,
            'rewards': rewards,
            'portfolio_values': portfolio_values,
            'returns': returns.tolist() if hasattr(returns, 'tolist') else list(returns),
        }
        
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple episodes."""
        if not results:
            return {}
            
        # Calculate means
        metrics = [
            'total_reward', 'total_return', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'trade_count'
        ]
        
        aggregated = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in results]
            aggregated[f'mean_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
            
        # Best and worst
        returns = [r.get('total_return', 0) for r in results]
        aggregated['best_return'] = max(returns)
        aggregated['worst_return'] = min(returns)
        
        # Pass/fail based on Sharpe threshold
        mean_sharpe = aggregated['mean_sharpe_ratio']
        min_sharpe = self.config.get('min_sharpe_ratio', 0.5)
        aggregated['passed'] = mean_sharpe >= min_sharpe
        aggregated['min_sharpe_threshold'] = min_sharpe
        
        logger.info(f"Backtest complete: Sharpe={mean_sharpe:.3f}, Passed={aggregated['passed']}")
        
        return aggregated
        
    def validate_for_live_trading(
        self,
        agent: TradingAgent,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.20,
    ) -> bool:
        """
        Validate that agent meets criteria for live trading.
        
        Args:
            agent: Agent to validate
            min_sharpe: Minimum required Sharpe ratio
            max_drawdown: Maximum allowed drawdown
            
        Returns:
            True if agent passes validation
        """
        logger.info("Running validation backtest...")
        
        results = self.run(
            agent=agent,
            start_date=self.config.get('backtest_start', '2024-01-01'),
            end_date=self.config.get('backtest_end', '2025-01-01'),
            episodes=3,
        )
        
        sharpe = results.get('mean_sharpe_ratio', 0)
        drawdown = results.get('mean_max_drawdown', 1)
        
        passed = sharpe >= min_sharpe and drawdown <= max_drawdown
        
        if passed:
            logger.info(f"✅ Validation PASSED: Sharpe={sharpe:.3f}, MaxDD={drawdown:.2%}")
        else:
            logger.warning(f"❌ Validation FAILED: Sharpe={sharpe:.3f}, MaxDD={drawdown:.2%}")
            
        return passed
        
    def generate_report(
        self,
        result: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate a text report from backtest results.
        
        Args:
            result: Backtest result dictionary
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST REPORT                            ║
╠══════════════════════════════════════════════════════════════╣
║ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
║ Symbol: {self.symbol}
║ Timeframe: {self.timeframe}
║ Initial Balance: ${self.initial_balance:,.2f}
╠══════════════════════════════════════════════════════════════╣
║                    PERFORMANCE METRICS                        ║
╠══════════════════════════════════════════════════════════════╣
║ Total Return:     {result.get('mean_total_return', 0)*100:>8.2f}% (±{result.get('std_total_return', 0)*100:.2f}%)
║ Sharpe Ratio:     {result.get('mean_sharpe_ratio', 0):>8.3f}  (±{result.get('std_sharpe_ratio', 0):.3f})
║ Sortino Ratio:    {result.get('mean_sortino_ratio', 0):>8.3f}  (±{result.get('std_sortino_ratio', 0):.3f})
║ Max Drawdown:     {result.get('mean_max_drawdown', 0)*100:>8.2f}% (±{result.get('std_max_drawdown', 0)*100:.2f}%)
║ Trade Count:      {result.get('mean_trade_count', 0):>8.0f}  (±{result.get('std_trade_count', 0):.0f})
╠══════════════════════════════════════════════════════════════╣
║                    VALIDATION STATUS                          ║
╠══════════════════════════════════════════════════════════════╣
║ Min Sharpe Required: {result.get('min_sharpe_threshold', 0.5):.2f}
║ Status: {'✅ PASSED' if result.get('passed', False) else '❌ FAILED'}
╚══════════════════════════════════════════════════════════════╝
"""
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {save_path}")
            
        return report
        
    def save_results(self, path: str = "./data/backtest_results.json"):
        """Save all results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to: {path}")


def main():
    """CLI entry point for backtesting."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2025-01-01', help='End date')
    parser.add_argument('--model', help='Path to trained model')
    parser.add_argument('--config', default='./config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
    )
    
    # Create environment
    env = CryptoTradingEnv(
        df=df,
        initial_balance=config.get('trading', {}).get('initial_balance', 10000.0),
    )
    
    # Create or load agent
    from src.brain import TradingAgent
    agent = TradingAgent(
        env=env,
        config=config.get('model', {}),
        model_path=args.model,
    )
    
    # If no model provided, train first
    if not args.model:
        logger.info("No model provided, training new agent...")
        agent.train(total_timesteps=100000)
        
    # Run backtest
    engine = BacktestEngine(config=config)
    results = engine.run(
        agent=agent,
        start_date=args.start,
        end_date=args.end,
    )
    
    # Generate report
    report = engine.generate_report(results)
    print(report)


if __name__ == '__main__':
    main()
