import sys
import logging
from src.models.ensemble_orchestrator import EnsembleOrchestrator
from src.models.regime_classifier import RegimeClassifier

logging.basicConfig(level=logging.INFO)

symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
for symbol in symbols:
    print(f"\n--- Testing Orchestrator for {symbol} ---")
    orchestrator = EnsembleOrchestrator(symbol=symbol)
    success = orchestrator.load()
    print(f"Orchestrator Loaded: {success}")
