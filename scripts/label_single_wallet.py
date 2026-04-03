#!/usr/bin/env python3
"""Label a single wallet. Run as subprocess to ensure clean memory."""
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

wallet = sys.argv[1]

from src.whale_behavior.data.price_labeler import PriceLabeler
labeler = PriceLabeler()
labeled = labeler.label_wallet(wallet)
print(f"DONE:{wallet}:{len(labeled)}")
del labeled
del labeler
gc.collect()
