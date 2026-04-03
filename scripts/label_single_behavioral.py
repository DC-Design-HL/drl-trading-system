#!/usr/bin/env python3
"""Label a single wallet with behavioral labels. Run as subprocess."""
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

from src.whale_behavior.data.behavioral_labeler import label_and_save_wallet
count = label_and_save_wallet(wallet)
print(f"DONE:{wallet}:{count}")
gc.collect()
