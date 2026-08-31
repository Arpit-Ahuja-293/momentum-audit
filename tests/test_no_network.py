"""The audit is only reproducible if analysis code cannot phone home.

Only scripts/download.py may import yfinance or touch the network.
"""

import pathlib

MODULE_DIR = pathlib.Path(__file__).resolve().parents[1] / "momaudit"
FORBIDDEN = ["yfinance", "requests", "urllib.request", "urllib3", "httpx"]


def test_momaudit_package_has_no_network_imports():
    offenders = []
    for path in MODULE_DIR.rglob("*.py"):
        text = path.read_text()
        for token in FORBIDDEN:
            if f"import {token}" in text:
                offenders.append(f"{path.name}: {token}")
    assert offenders == [], f"network imports found in momaudit/: {offenders}"
