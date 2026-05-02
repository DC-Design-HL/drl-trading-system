"""Simulation primitives — broker, position, portfolio.

These mirror the live bot's _open_position / _close_position math so
the framework's PnL is bit-identical to live (within tolerance) on the
same trade sequence.
"""
