"""Test defaults: never touch the network or a cloud quantum provider."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["QUANTUMAGENT_OFFLINE"] = "1"
os.environ.pop("QUANTUMAGENT_ENABLE_CLOUD", None)
