#!/usr/bin/env python3
"""
Fetch BTC whale transactions for known exchange wallets.

Sources (all free, public):
  * mempool.space — preferred, well-maintained, no auth needed
  * blockchain.info — fallback for some endpoints

Approach:
  1. Hardcoded list of known BTC exchange wallets (cold + hot for major
     exchanges). Sources: walletexplorer.com, bitinfocharts.com,
     public on-chain analysis.
  2. For each wallet, fetch confirmed transactions via mempool.space.
  3. Filter for ≥ MIN_BTC value (default 50 BTC ≈ $3.5M).
  4. Tag each tx with direction (in/out from the wallet's perspective).
  5. Save to data/whale_behavior/btc/<wallet>.jsonl in the SAME format
     as data/whale_behavior/eth/ so downstream code is uniform.

Output format (matches eth/):
  {"timestamp": <unix>, "value_btc": <float>, "tx_hash": <str>,
   "direction": "in"|"out", "block": <int>, "wallet": <str>}

Run:
    python3 scripts/fetch_btc_whales.py
        [--min_btc 50]            # filter dust
        [--limit_per_wallet 1000] # cap per wallet (newest-first)
        [--start 2026-01-01]      # filter timestamp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "whale_behavior" / "btc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEMPOOL_BASE = "https://mempool.space/api"

# Known BTC exchange wallets (cold + hot). Sourced from public on-chain
# analysis. Some of these are clusters; addresses below are well-known
# representatives that handle a meaningful fraction of each exchange's flow.
KNOWN_WALLETS = {
    # Binance — well-documented multiple cold wallets
    "binance_cold_1":   "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo",   # Binance #1 (the iconic 200K BTC cold)
    "binance_cold_14":  "1NDyJtNTjmwk5xPNhjgAMu4HDHigtobu1s",   # Binance #14
    "binance_hot_1":    "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",  # Binance hot 1
    # Bitfinex
    "bitfinex_cold":    "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
    # Coinbase
    "coinbase_1":       "1FxkfJQLJTXpW6QmxGT6oF43ZH959ns8Cq",   # Coinbase #1
    "coinbase_2":       "385cR5DM96n1HvBDMzLHPYcw89fZAXULJP",   # Coinbase #2
    # Kraken
    "kraken_1":         "3FupZp77ySr7jwoLYEJ9mwzJpvoNBXsBnE",
    # Bitstamp
    "bitstamp_1":       "385cR5DM96n1HvBDMzLHPYcw89fZAXULJP",   # Note: dup with coinbase, will dedupe
    # OKX
    "okx_1":            "bc1qjasf9z3h7w3jspkhtgatgpyvvzgpa2wwd2lr0eh5tx44reyn2k7sfc27a4",
    # Bittrex
    "bittrex_1":        "385cR5DM96n1HvBDMzLHPYcw89fZAXULJP",   # placeholder, will dedupe
    # MicroStrategy treasury (institutional, not exchange but whale)
    "microstrategy_1":  "bc1qazcm763858nkj2dj986etajv6wquslv8uxwczt",
    # Mt. Gox cold (legacy, still active for distributions)
    "mtgox_trustee":    "1Jzo8MV77zCFMcc4Q22e6PQzyuHN1Pu5JT",
}


def fetch_address_txs(address: str, limit: int = 1000) -> list[dict]:
    """Fetch most-recent confirmed transactions for an address.
    mempool.space returns up to 50 per page via 'before' pagination
    by tx hash. We'll page through up to `limit` total.
    """
    out = []
    last_seen_txid = None
    while len(out) < limit:
        url = f"{MEMPOOL_BASE}/address/{address}/txs"
        if last_seen_txid:
            url = f"{url}/chain/{last_seen_txid}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 429:
                print(f"  rate-limited; sleeping 30s")
                time.sleep(30); continue
            if resp.status_code == 404:
                print(f"  address not found / no txs: {address}")
                break
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}: {resp.text[:120]}")
                break
            data = resp.json()
        except Exception as exc:
            print(f"  fetch error: {exc}; sleeping 5s")
            time.sleep(5); continue
        if not data: break
        out.extend(data)
        last_seen_txid = data[-1]["txid"]
        if len(data) < 25:  # last page
            break
        time.sleep(0.4)  # gentle on the public API
    return out[:limit]


def classify_tx(tx: dict, address: str, min_btc: float) -> dict | None:
    """Determine direction & value from the address's POV.
    A tx may have many inputs and many outputs; we sum values per side
    where the address is the sender (input) or receiver (output).
    """
    sent_sat = 0
    recv_sat = 0
    for vin in tx.get("vin", []):
        prev = vin.get("prevout") or {}
        if (prev.get("scriptpubkey_address") or "") == address:
            sent_sat += int(prev.get("value") or 0)
    for vout in tx.get("vout", []):
        if (vout.get("scriptpubkey_address") or "") == address:
            recv_sat += int(vout.get("value") or 0)
    net_recv_btc = (recv_sat - sent_sat) / 1e8
    abs_btc = abs(net_recv_btc)
    if abs_btc < min_btc:
        return None
    direction = "in" if net_recv_btc > 0 else "out"
    status = tx.get("status") or {}
    block_time = status.get("block_time")
    if block_time is None:
        return None  # unconfirmed
    return {
        "timestamp": int(block_time),
        "value_btc": abs_btc,
        "tx_hash": tx.get("txid"),
        "direction": direction,
        "block": status.get("block_height"),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--min_btc", type=float, default=50.0)
    p.add_argument("--limit_per_wallet", type=int, default=1000)
    p.add_argument("--start", default="2026-01-01")
    args = p.parse_args(argv)

    start_dt = datetime.fromisoformat(args.start + "T00:00:00").replace(tzinfo=timezone.utc)
    start_ts = int(start_dt.timestamp())

    print(f"Filter: ≥{args.min_btc} BTC, since {start_dt.isoformat()}, "
          f"limit {args.limit_per_wallet}/wallet")

    seen_addresses: set[str] = set()
    summary = {}
    for wallet_name, address in KNOWN_WALLETS.items():
        if address in seen_addresses:
            print(f"\n{wallet_name}: duplicate address {address}, skipping")
            continue
        seen_addresses.add(address)
        out_path = OUT_DIR / f"{wallet_name}.jsonl"
        if out_path.exists():
            print(f"\n{wallet_name}: already fetched ({out_path.stat().st_size} bytes), skipping")
            summary[wallet_name] = "cached"
            continue
        print(f"\n{wallet_name}: fetching txs for {address}")
        txs = fetch_address_txs(address, limit=args.limit_per_wallet)
        print(f"  retrieved {len(txs)} raw txs")
        kept = []
        with open(out_path, "w") as fp:
            for tx in txs:
                row = classify_tx(tx, address, args.min_btc)
                if row is None: continue
                if row["timestamp"] < start_ts: continue
                row["wallet"] = wallet_name
                fp.write(json.dumps(row) + "\n")
                kept.append(row)
        print(f"  wrote {len(kept)} ≥{args.min_btc} BTC txs since {start_dt.date()} "
              f"to {out_path.name}")
        summary[wallet_name] = len(kept)

    print(f"\nSummary:")
    for w, n in summary.items():
        print(f"  {w}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
