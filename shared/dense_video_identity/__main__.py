from __future__ import annotations
import argparse
from pathlib import Path
from .common import read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["smoke", "features", "features-demo", "audit", "train-identity", "train-utility", "evaluate", "serve"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test", "demo"], default="dev")
    p.add_argument("--port", type=int, default=8871)
    a = p.parse_args()
    config = read_json(a.root / "config.json")
    if a.command in ("smoke", "features", "features-demo"):
        from .infer import extract
        extract(a.root, config["detector"], smoke=a.command == "smoke", demo_only=a.command == "features-demo")
    elif a.command == "train-identity":
        from .learn import train_identity
        train_identity(a.root, config)
    elif a.command == "train-utility":
        from .learn import train_utility
        train_utility(a.root, config)
    elif a.command == "evaluate":
        from .evaluate import evaluate
        evaluate(a.root, config, a.split)
    elif a.command == "audit":
        from .audit import audit
        audit(a.root, config)
    elif a.command == "serve":
        from .server import serve
        serve(a.root, a.port)


if __name__ == "__main__":
    main()
