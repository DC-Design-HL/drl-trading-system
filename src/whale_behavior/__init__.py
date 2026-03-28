"""
Whale Behavior Prediction Module

Per-wallet behavioral pattern learning for ETH whale wallets.
Learns wallet-specific action sequences that precede major price moves.

This is a NEW module — does NOT replace the existing whale tracking system
in src/features/whale_tracker.py. Instead, it produces an additional signal
that can be integrated alongside existing signals.
"""
