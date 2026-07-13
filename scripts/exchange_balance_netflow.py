#!/usr/bin/env python3
"""
Exchange-netflow v2 — balance-delta method (2026-07-13, foundation rebuild).

The transfer-scrape netflow (netflow_signal_validation.py) was blocked by two
data problems: only ~25 usable days and 99.7% unlabeled counterparties (can't
separate trader deposits from Binance internal hot<->cold shuffling).

This harness sidesteps both: fetch each exchange wallet's native-ETH balance
at daily UTC-midnight block boundaries via archive eth_getBalance, and sum
across the exchange's whole wallet CLUSTER. Internal shuffles between cluster
wallets cancel out in the sum by construction, and archive state gives years
of history. Balance deltas also capture beacon withdrawals that transfer
scraping misses. Gas spend is negligible at cluster scale (~10 ETH/day vs
thousands netflow).

Daily netflow(d) = cluster_balance(end of d) - cluster_balance(end of d-1).
Positive = net deposits to exchange = expected BEARISH for forward returns.

Data + stats only (no training) — safe on server. Resumable: balances cached
to data/alternative_cache/exchange_balance_daily.json; rerun to extend.

Usage:
  python3 scripts/exchange_balance_netflow.py fetch [START_DAY]   # default 2024-01-01
  python3 scripts/exchange_balance_netflow.py analyze
"""
import json
import sys
import time
import datetime
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "alternative_cache" / "exchange_balance_daily.json"
DEFAULT_START = "2024-01-01"
AVG_BLOCK_S = 12.06  # post-merge average incl. missed slots

# Wallet clusters. binance_core = the 4 registry wallets the transfer scraper
# already tracked. binance_ext = additional long-standing Etherscan-labeled
# Binance wallets (incl. 0x56Eddb... which the registry mislabels as
# "Smart Money Whale 1" — it is Etherscan-labeled Binance 17). The analysis
# reports core-only vs full-cluster so a mislabeled ext address can't silently
# corrupt conclusions.
CLUSTERS = {
    "binance_core": {
        "0x28C6c06298d514Db089934071355E5743bf21d60": "Binance 14 (hot)",
        "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549": "Binance 15 (cold)",
        "0xF977814e90dA44bFA03b6295A0616a897441aceC": "Binance 8 (reserve)",
        "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8": "Binance 7 (cold)",
    },
    "binance_ext": {
        "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE": "Binance 1 (legacy main)",
        "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF": "Binance 2",
        "0x564286362092D8e7936f0549571a803B203aAceD": "Binance 3",
        "0x0681d8Db095565FE8A346fA0277bFfdE9C0eDBBF": "Binance 4",
        "0xfE9e8709d3215310075d67E3ed32A380CCf451C8": "Binance 5",
        "0xDFd5293D8e347dFe59E90eFd55b2956a1343963d": "Binance 16",
        "0x9696f59E4d72E237BE84fFD425DCaD154Bf96976": "Binance 18",
        "0x5a52E96BAcdaBb82fd05763E25335261B270Efcb": "Binance 28",
        "0x4976A4A02f38326660D17bf34b431dC6e2eb2327": "Binance 20",
        "0x56Eddb7aa87536c09CCc2793473599fD21A8b17F": "Binance 17 (registry mislabel: smart_money_whale_1)",
    },
    "coinbase": {
        "0xA090e606E30bD747d4E6245a1517EbE430F0057e": "Coinbase Institutional",
    },
    "kraken": {
        "0x2910543af39abA0Cd09dBb2D50200b3E800A63D2": "Kraken Deposit",
    },
    "robinhood": {
        "0x40B38765696e3d5d8d9d834D8AaD4bB6e418E489": "Robinhood",
    },
}
ALL_ADDRS = {a: lbl for c in CLUSTERS.values() for a, lbl in c.items()}

# ERC-20 stablecoins for the reserve signal (rising exchange stablecoin
# reserve = dry powder = classically BULLISH). balanceOf via eth_call.
TOKENS = {
    "USDT": ("0xdAC17F958D2ee523a2206206994597C13D831ec7", 6),
    "USDC": ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6),
}


def _alchemy_url():
    for line in open(REPO / ".env"):
        line = line.strip()
        if line.startswith("ALCHEMY_API_KEY="):
            return "https://eth-mainnet.g.alchemy.com/v2/" + line.split("=", 1)[1].strip()
    raise SystemExit("ALCHEMY_API_KEY not found in .env")


def rpc(url, payload, tries=8):
    body = json.dumps(payload).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            if i == tries - 1:
                raise
            # 429: the live whale collector shares this key — back off hard
            time.sleep(min(60, 5 * (i + 1)) if e.code == 429 else 2 ** i)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def get_block_ts(url, block_num):
    for attempt in range(6):
        r = rpc(url, {"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber",
                      "params": [hex(block_num), False]})
        if "error" in r:  # rate limit etc. — NOT the same as beyond-head
            time.sleep(1.5 * (attempt + 1))
            continue
        b = r.get("result")
        if b is None:
            return None  # block genuinely doesn't exist (beyond head)
        return int(b["timestamp"], 16)
    raise RuntimeError(f"getBlockByNumber rate-limited after retries: {block_num}")


def find_boundary_block(url, target_ts, seed_block, seed_ts):
    """Last block with timestamp < target_ts, via estimate-and-correct."""
    block, ts = seed_block, seed_ts
    lo, hi = None, None  # lo: ts < target, hi: ts >= target
    for _ in range(40):
        if ts < target_ts:
            lo = (block, ts)
        else:
            hi = (block, ts)
        if lo and hi and hi[0] - lo[0] == 1:
            return lo[0], lo[1]
        step = int((target_ts - ts) / AVG_BLOCK_S)
        if step == 0:
            step = 1 if ts < target_ts else -1
        nxt = block + step
        if lo and hi:  # keep inside bracket
            nxt = max(lo[0] + 1, min(hi[0] - 1, nxt))
        elif lo:
            nxt = max(nxt, lo[0] + 1)
        elif hi:
            nxt = min(nxt, hi[0] - 1)
        block = nxt
        t = get_block_ts(url, block)
        if t is None:  # beyond head
            hi = (block, target_ts)
            block -= max(1, step // 2)
            t = get_block_ts(url, block)
            if t is None:
                raise RuntimeError("cannot find head")
        ts = t
    # fall back to best lower bound
    if lo:
        return lo[0], lo[1]
    raise RuntimeError(f"bracket failed for ts={target_ts}")


def fetch_balances_at(url, block_num, addrs):
    out = {}
    pending = list(addrs)
    for attempt in range(6):
        batch = [{"jsonrpc": "2.0", "id": i,
                  "method": "eth_getBalance", "params": [a, hex(block_num)]}
                 for i, a in enumerate(pending)]
        res = rpc(url, batch)
        failed = []
        for item in res:
            if "result" in item:
                out[pending[item["id"]]] = int(item["result"], 16) / 1e18
            elif item.get("error", {}).get("code") == 429:
                failed.append(pending[item["id"]])
            else:
                raise RuntimeError(f"getBalance error at block {block_num}: {item}")
        if not failed:
            return out
        pending = failed
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"getBalance still rate-limited after retries at block {block_num}")


def fetch_token_balances_at(url, block_num, addrs, token_addr, decimals):
    out = {}
    pending = list(addrs)
    for attempt in range(6):
        batch = []
        for i, a in enumerate(pending):
            data = "0x70a08231" + a[2:].lower().rjust(64, "0")  # balanceOf(addr)
            batch.append({"jsonrpc": "2.0", "id": i, "method": "eth_call",
                          "params": [{"to": token_addr, "data": data}, hex(block_num)]})
        res = rpc(url, batch)
        failed = []
        for item in res:
            if "result" in item:
                raw = item["result"]
                out[pending[item["id"]]] = (int(raw, 16) if raw and raw != "0x" else 0) / 10 ** decimals
            elif item.get("error", {}).get("code") == 429:
                failed.append(pending[item["id"]])
            else:
                raise RuntimeError(f"eth_call error at block {block_num}: {item}")
        if not failed:
            return out
        pending = failed
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"eth_call still rate-limited after retries at block {block_num}")


def cmd_fetch_tokens():
    """Fetch daily stablecoin balances for the binance cluster at the already-
    cached day-boundary blocks (run `fetch` first)."""
    url = _alchemy_url()
    cache = load_cache()
    if not cache["blocks"]:
        raise SystemExit("no cached day boundaries — run fetch first")
    cache.setdefault("token_balances", {})
    addrs = list(CLUSTERS["binance_core"]) + list(CLUSTERS["binance_ext"])
    days = sorted(cache["blocks"])
    t0 = time.time()
    for tok, (tok_addr, dec) in TOKENS.items():
        store = cache["token_balances"].setdefault(tok, {})
        done = 0
        for dstr in days:
            missing = [a for a in addrs if dstr not in store.get(a, {})]
            if missing:
                bal = fetch_token_balances_at(url, cache["blocks"][dstr], missing,
                                              tok_addr, dec)
                for a, v in bal.items():
                    store.setdefault(a, {})[dstr] = v
                time.sleep(0.6)
            done += 1
            if done % 100 == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"  {tok} {dstr} ({done}/{len(days)}, "
                      f"{done/(time.time()-t0):.1f} days/s)", flush=True)
        json.dump(cache, open(CACHE, "w"))
        print(f"{tok} fetch complete")


def cmd_analyze_tokens():
    import statistics
    cache = load_cache()
    days = sorted(cache["blocks"])
    closes = eth_daily_closes(days[0], days[-1])
    addrs = list(CLUSTERS["binance_core"]) + list(CLUSTERS["binance_ext"])
    # reserve = USDT+USDC summed over binance cluster; delta = stablecoin inflow
    combined = {}
    for d in days:
        tot, ok = 0.0, True
        for tok in TOKENS:
            store = cache.get("token_balances", {}).get(tok, {})
            vals = [store.get(a, {}).get(d) for a in addrs]
            if any(v is None for v in vals):
                ok = False
                break
            tot += sum(vals)
        if ok:
            combined[d] = tot
    print(f"stablecoin reserve days: {len(combined)}")
    sd = sorted(combined)
    sample = [f"{d}: ${combined[d]/1e9:.2f}B" for d in sd[::180]]
    print("  reserve level over time:", "  ".join(sample))
    # reuse analyze via a synthetic single-cluster cache; signal here is
    # POSITIVE delta = bullish, so print with that framing
    rows = [(sd[i], combined[sd[i]] - combined[sd[i - 1]]) for i in range(1, len(sd))]
    idx = {d: i for i, d in enumerate(sorted(closes))}
    sdays = sorted(closes)
    for h in (1, 3, 7):
        pairs = []
        for d, nf in rows:
            i = idx.get(d)
            if i is None or i + h >= len(sdays):
                continue
            pairs.append((d, nf, closes[sdays[i + h]] / closes[sdays[i]] - 1))
        nfs = [p[1] for p in pairs]
        fwds = [p[2] for p in pairs]
        labels = [1 if f > 0 else 0 for f in fwds]
        print(f"  h={h}d n={len(pairs)}  Pearson={pearson(nfs, fwds):+.3f}  "
              f"Spearman={spearman(nfs, fwds):+.3f}  "
              f"AUC(+inflow->UP)={auc(nfs, labels):.3f}  "
              f"(expect POS corr / AUC>0.5 if dry-powder thesis real)")
        order = sorted(pairs, key=lambda p: p[1])
        q = len(order) // 4 or 1
        line = "    "
        for nm, ch in [("Q1 outflow", order[:q]), ("Q2", order[q:2 * q]),
                       ("Q3", order[2 * q:3 * q]), ("Q4 inflow", order[3 * q:])]:
            fr = [c[2] for c in ch]
            wr = sum(1 for x in fr if x > 0) / len(fr) * 100
            line += f"{nm}: {statistics.mean(fr)*100:+.2f}%/{wr:.0f}%UP  "
        print(line)
        t = max(len(order) // 20, 5)
        print(f"    extremes: bot5% n={t} avg={statistics.mean([c[2] for c in order[:t]])*100:+.2f}%"
              f"  |  top5% n={t} avg={statistics.mean([c[2] for c in order[-t:]])*100:+.2f}%")


def load_cache():
    if CACHE.exists():
        return json.load(open(CACHE))
    return {"blocks": {}, "balances": {}}


def cmd_fetch(start_day):
    url = _alchemy_url()
    cache = load_cache()
    # only complete days: end-of-day boundary for today doesn't exist yet
    yday = (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    day = datetime.datetime.strptime(start_day, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
    end = datetime.datetime.strptime(yday, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)

    # seed anchor: latest block
    head = rpc(url, {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []})
    seed_block = int(head["result"], 16) - 10
    seed_ts = get_block_ts(url, seed_block)
    addrs = list(ALL_ADDRS)
    n_days = int((end - day).days) + 1
    done = 0
    t0 = time.time()
    while day <= end:
        dstr = day.strftime("%Y-%m-%d")
        # boundary = last block before the NEXT midnight => end-of-day dstr
        next_mid = int((day + datetime.timedelta(days=1)).timestamp())
        if dstr not in cache["blocks"]:
            blk, blk_ts = find_boundary_block(url, next_mid, seed_block, seed_ts)
            cache["blocks"][dstr] = blk
            seed_block, seed_ts = blk, blk_ts
        blk = cache["blocks"][dstr]
        missing = [a for a in addrs if dstr not in cache["balances"].get(a, {})]
        if missing:
            bal = fetch_balances_at(url, blk, missing)
            for a, v in bal.items():
                cache["balances"].setdefault(a, {})[dstr] = v
            time.sleep(0.25)  # stay well under CU/s cap
        done += 1
        if done % 50 == 0:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            json.dump(cache, open(CACHE, "w"))
            rate = done / (time.time() - t0)
            print(f"  {dstr} done ({done}/{n_days}, {rate:.1f} days/s)", flush=True)
        day += datetime.timedelta(days=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(CACHE, "w"))
    print(f"fetch complete: {len(cache['blocks'])} day-boundaries cached -> {CACHE}")


# ---------------- analysis ----------------

def eth_daily_closes(start_day, end_day):
    start_ms = int(datetime.datetime.strptime(start_day, "%Y-%m-%d")
                   .replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.datetime.strptime(end_day, "%Y-%m-%d")
                 .replace(tzinfo=datetime.timezone.utc).timestamp() * 1000) + 86400_000
    closes = {}
    cur = start_ms
    while cur < end_ms:
        url = (f"https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1d"
               f"&startTime={cur}&limit=1000")
        data = json.load(urllib.request.urlopen(url, timeout=20))
        if not data:
            break
        for k in data:
            d = datetime.datetime.fromtimestamp(k[0] / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
            closes[d] = float(k[4])
        cur = data[-1][0] + 86400_000
        if len(data) < 1000:
            break
    return closes


def pearson(xs, ys):
    import statistics
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    return pearson(ranks(xs), ranks(ys))


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
        1 for p in pos for n in neg if p == n)
    return w / (len(pos) * len(neg))


def analyze_cluster(name, addrs, cache, closes, horizons=(1, 3, 7)):
    import statistics
    days = sorted(cache["blocks"])
    bal = []
    for d in days:
        vals = [cache["balances"].get(a, {}).get(d) for a in addrs]
        if any(v is None for v in vals):
            bal.append(None)
            continue
        bal.append(sum(vals))
    # netflow(d) = bal(d) - bal(d-1)
    rows = []  # (day, netflow)
    for i in range(1, len(days)):
        if bal[i] is None or bal[i - 1] is None:
            continue
        rows.append((days[i], bal[i] - bal[i - 1]))
    sdays = sorted(closes)
    idx = {d: i for i, d in enumerate(sdays)}
    print(f"\n=== cluster: {name} ({len(addrs)} wallets, {len(rows)} netflow days) ===")
    if len(rows) < 30:
        print("  insufficient data")
        return
    nf_abs = [abs(r[1]) for r in rows]
    print(f"  |netflow| median={statistics.median(nf_abs):,.0f} ETH  "
          f"p90={sorted(nf_abs)[int(len(nf_abs)*0.9)]:,.0f} ETH")
    for h in horizons:
        pairs = []
        for d, nf in rows:
            i = idx.get(d)
            if i is None or i + h >= len(sdays):
                continue
            fwd = closes[sdays[i + h]] / closes[sdays[i]] - 1
            pairs.append((d, nf, fwd))
        if len(pairs) < 30:
            continue
        nfs = [p[1] for p in pairs]
        fwds = [p[2] for p in pairs]
        labels = [1 if f > 0 else 0 for f in fwds]
        print(f"  h={h}d n={len(pairs)}  Pearson={pearson(nfs, fwds):+.3f}  "
              f"Spearman={spearman(nfs, fwds):+.3f}  "
              f"AUC(-nf->UP)={auc([-x for x in nfs], labels):.3f}  "
              f"(expect neg corr / AUC>0.5 if inflow=bearish)")
        order = sorted(pairs, key=lambda p: p[1])
        q = len(order) // 4 or 1
        chunks = [("Q1 outflow", order[:q]), ("Q2", order[q:2 * q]),
                  ("Q3", order[2 * q:3 * q]), ("Q4 inflow", order[3 * q:])]
        line = "    "
        for nm, ch in chunks:
            fr = [c[2] for c in ch]
            wr = sum(1 for x in fr if x > 0) / len(fr) * 100
            line += f"{nm}: {statistics.mean(fr)*100:+.2f}%/{wr:.0f}%UP  "
        print(line)
        # extreme tails — maybe signal lives only in p95 events
        t = max(len(order) // 20, 5)
        lo_t, hi_t = order[:t], order[-t:]
        print(f"    extremes: bot5% (outflow) n={t} avg={statistics.mean([c[2] for c in lo_t])*100:+.2f}%"
              f"  |  top5% (inflow) n={t} avg={statistics.mean([c[2] for c in hi_t])*100:+.2f}%")


def cmd_analyze():
    cache = load_cache()
    if not cache["blocks"]:
        raise SystemExit("no cached balances — run fetch first")
    days = sorted(cache["blocks"])
    closes = eth_daily_closes(days[0], days[-1])
    print(f"balance days: {len(days)} ({days[0]} -> {days[-1]}), "
          f"price days: {len(closes)}")
    core = list(CLUSTERS["binance_core"])
    full = core + list(CLUSTERS["binance_ext"])
    analyze_cluster("binance_core (4 registry wallets)", core, cache, closes)
    analyze_cluster("binance_full (core + 10 labeled)", full, cache, closes)
    analyze_cluster("coinbase_inst", list(CLUSTERS["coinbase"]), cache, closes)
    analyze_cluster("kraken", list(CLUSTERS["kraken"]), cache, closes)
    all_ex = full + list(CLUSTERS["coinbase"]) + list(CLUSTERS["kraken"]) + \
        list(CLUSTERS["robinhood"])
    analyze_cluster("all_exchanges", all_ex, cache, closes)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if mode == "fetch":
        start = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START
        cmd_fetch(start)
        cmd_analyze()
    elif mode == "analyze":
        cmd_analyze()
    elif mode == "fetch-tokens":
        cmd_fetch_tokens()
        cmd_analyze_tokens()
    elif mode == "analyze-tokens":
        cmd_analyze_tokens()
    else:
        raise SystemExit("usage: exchange_balance_netflow.py "
                         "[fetch [START]|analyze|fetch-tokens|analyze-tokens]")
