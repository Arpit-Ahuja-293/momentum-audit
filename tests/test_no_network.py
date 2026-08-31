"""The audit is only reproducible if analysis code cannot phone home.

Only scripts/download.py may import yfinance or touch the network.
"""

import pathlib
import re

MODULE_DIR = pathlib.Path(__file__).resolve().parents[1] / "momaudit"
FORBIDDEN = ["yfinance", "requests", "urllib.request", "urllib3", "httpx"]


def find_network_imports(directory: pathlib.Path, forbidden_tokens: list[str]) -> list[str]:
    """Scan a directory for forbidden network imports (both `import` and `from` forms).

    Returns a list of offending files with the format "filename: module".
    """
    offenders = []
    for path in directory.rglob("*.py"):
        text = path.read_text()
        for token in forbidden_tokens:
            # Match both "import token" and "from ... import token" forms
            pattern = rf"^\s*(import|from)\s+{re.escape(token)}\b"
            if re.search(pattern, text, re.MULTILINE):
                offenders.append(f"{path.name}: {token}")
    return offenders


def test_momaudit_package_has_no_network_imports():
    offenders = find_network_imports(MODULE_DIR, FORBIDDEN)
    assert offenders == [], f"network imports found in momaudit/: {offenders}"


def test_network_import_guard_catches_from_import_form(tmp_path):
    # Verify the guard actually catches "from X import Y" form
    test_file = tmp_path / "test_module.py"
    test_file.write_text("from yfinance import Ticker\n")
    offenders = find_network_imports(tmp_path, ["yfinance"])
    assert len(offenders) == 1
    assert "test_module.py" in offenders[0]
    assert "yfinance" in offenders[0]
