"""Finite CPU-only WMI launch survival contract, independent of GPU work."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    value = wintypes.BOOL()
    if not k.IsProcessInJob(k.GetCurrentProcess(), None, ctypes.byref(value)):
        raise OSError(ctypes.get_last_error())
    initial = dict(pid=os.getpid(), in_job=bool(value.value), cpu_only=True, source=str(Path(__file__).resolve()))
    (out / "STARTED.json").write_text(json.dumps(initial), encoding="utf-8")
    time.sleep(5)
    summary = dict(passed=not bool(value.value), **initial, delayed_output_written=True,
                   gt_parsed=False, torch_imported=False, models_loaded=False,
                   limit="proves this WMI-created fresh chain survives launcher return; does not detach an existing job")
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError("Fresh WMI child unexpectedly inherited a Job")


if __name__ == "__main__":
    main()
