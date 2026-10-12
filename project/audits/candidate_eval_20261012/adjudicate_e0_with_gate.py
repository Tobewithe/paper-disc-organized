#!/usr/bin/env python3
"""Adjudicate E0 only after mandatory official-native baseline parity passes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validate_e0_receipt import main as validate_receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--official-per-candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    eval_dir = args.eval_dir.resolve()
    gate_file = eval_dir / "E0_GATE.json"
    checked = validate_receipt(["--eval-dir", str(eval_dir),
                                "--official-per-candidate", str(args.official_per_candidate),
                                "--out", str(gate_file)])
    if checked:
        print("E0 scientific decision BLOCKED: see E0_GATE.json", file=sys.stderr)
        return 2
    # Keep existing pre-specified dev selection and val-qualification rule intact.
    root = Path(__file__).resolve().parents[3]
    scripts = root / "experiments" / "coefficient_shared_affine_alignment_audit_20261011" / "scripts"
    sys.path.insert(0, str(scripts))
    import decide_cross
    decision = decide_cross.decide(eval_dir, out=(args.out or eval_dir))
    decision["input_gate"] = {"status": "valid_for_E0_adjudication",
                              "receipt": str(gate_file),
                              "rule": "candidate identity + native A parity + converged maps"}
    target = (args.out or eval_dir) / "DECISION.json"
    target.write_text(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gate": "pass",
                      "decision": str(target)}, ensure_ascii=False))
    return 0 if decision["status"] != "insufficient_inputs" else 2


if __name__ == "__main__":
    sys.exit(main())
