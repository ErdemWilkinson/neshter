"""E2E sunucusu: minik rastgele modelle gerçek Neşter sunucusu. `W` D:/neshter/e2e-work altında olmalı."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

E2E_ROOT = Path("D:/neshter/e2e-work").resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--token", default="e2e-token")
    args = parser.parse_args()

    work = Path(args.workdir).resolve()
    if E2E_ROOT not in work.parents and work != E2E_ROOT:
        raise SystemExit(f"workdir {E2E_ROOT} altında olmalı: {work}")
    os.environ["NESHTER_HOME"] = str(work / "home")
    for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        os.environ.setdefault(name, str(work / "cache"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from tiny_models import build_tiny_qwen2  # noqa: E402

    hub = work / "hub"
    snapshot = hub / "models--neshter--tiny-e2e" / "snapshots" / "e2e"
    if not (snapshot / "config.json").exists():  # bir kez kurulur; yeniden başlatmalarda SHA sabit kalır
        snapshot.mkdir(parents=True, exist_ok=True)
        build_tiny_qwen2(snapshot)

    from neshter import cli  # noqa: E402
    from neshter.server.app import create_app  # noqa: E402

    app = create_app(args.token, args.port, roots={"huggingface": [hub]})
    cli._run_server(app, "127.0.0.1", args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
