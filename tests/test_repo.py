import re
from pathlib import Path

import neshter

ROOT = Path(__file__).resolve().parent.parent


def test_docs_have_no_private_links_or_local_paths():
    files = [
        *ROOT.glob("README*.md"),
        ROOT / "SECURITY.md",
    ]
    assert ROOT / "SECURITY.md" in files and all(f.exists() for f in files)
    for f in files:
        t = f.read_text(encoding="utf-8")
        assert "claude.ai/artifact" not in t and not re.search(r"\b[A-Za-z]:[\\/]", t), f


def test_version_is_0_2_0():
    assert neshter.__version__ == "0.2.0"


def test_license_is_apache_2():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
