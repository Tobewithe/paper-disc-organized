"""Read-only Windows job-membership probe for explicitly owned processes."""
import argparse
import ctypes
import json
from ctypes import wintypes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, action="append", required=True)
    a = p.parse_args()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    rows = []
    for pid in a.pid:
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            rows.append(dict(pid=pid, in_job=None, error=ctypes.get_last_error()))
            continue
        value = wintypes.BOOL()
        passed = kernel.IsProcessInJob(handle, None, ctypes.byref(value))
        rows.append(dict(pid=pid, in_job=bool(value.value) if passed else None,
                         error=None if passed else ctypes.get_last_error()))
        kernel.CloseHandle(handle)
    print(json.dumps(dict(read_only=True, processes=rows,
                         scope="job membership only; neither job limits nor disconnect survival proven")))


if __name__ == "__main__":
    main()
