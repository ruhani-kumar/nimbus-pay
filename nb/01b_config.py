import csv
import json
import random
import time
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np

NOTEBOOK_START = time.time()

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/nimbus_triage")
else:
    ROOT = Path.cwd()

PACK = ROOT / "candidate_pack"
DATA = PACK / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
for d in (RUNS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SEED = 42
MAX_NEW_TOKENS = 200

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
CATEGORIES = ["payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"]
PRIORITIES = ["P1", "P2", "P3"]
CHANNELS = ["upi", "card", "netbanking", "wallet"]
LANGUAGES = ["en", "hi-en"]
SOURCES = ["app-chat", "email", "twitter", "ivr-transcript"]
TXN_FIELDS = ["amount_paise", "txn_date", "txn_id", "channel"]
NON_TXN_CATEGORIES = ["account_access", "kyc", "other"]

for p in [DATA / "train.jsonl", DATA / "dev.jsonl", DATA / "test_inputs.jsonl", PACK / "score.py", PACK / "SCHEMA.md"]:
    assert p.exists(), f"missing {p}"

print("root:", ROOT)
print("runs (adapters, large files):", RUNS)
print("results (small files committed to the repo):", RESULTS)
