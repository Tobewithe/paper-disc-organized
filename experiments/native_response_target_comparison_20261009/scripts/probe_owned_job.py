"""Read-only query of Job handles held by one explicitly owned SSH shell PID.

DuplicateHandle is used only for query; no limits are changed, no process is
assigned/terminated, and all duplicate handles are closed before exit.
"""
import argparse
import ctypes
import json
from ctypes import wintypes


class BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--owner-pid", type=int, required=True)
    p.add_argument("--target-pid", type=int, required=True, action="append")
    a = p.parse_args()
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.DuplicateHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    k.QueryInformationJobObject.restype = wintypes.BOOL
    owner = k.OpenProcess(0x0040, False, a.owner_pid)
    if not owner:
        print(json.dumps(dict(read_only=True, owner_pid=a.owner_pid, error=ctypes.get_last_error())))
        return
    found = []
    try:
        for value in range(4, 65536, 4):
            handle = wintypes.HANDLE()
            if not k.DuplicateHandle(owner, wintypes.HANDLE(value), k.GetCurrentProcess(), ctypes.byref(handle), 0, False, 2):
                continue
            try:
                buffer = ctypes.create_string_buffer(8 + 8 * 4096)
                returned = wintypes.DWORD()
                if not k.QueryInformationJobObject(handle, 3, buffer, len(buffer), ctypes.byref(returned)):
                    continue
                count = int.from_bytes(buffer.raw[4:8], "little")
                ids = [int.from_bytes(buffer.raw[8+8*j:16+8*j], "little") for j in range(count)]
                if not set(a.target_pid).issubset(ids):
                    continue
                limits = BasicLimits()
                ok = k.QueryInformationJobObject(handle, 2, ctypes.byref(limits), ctypes.sizeof(limits), ctypes.byref(returned))
                found.append(dict(owner_handle_value=value, members=ids, limit_flags=limits.LimitFlags if ok else None,
                                  kill_on_job_close=bool(limits.LimitFlags & 0x2000) if ok else None,
                                  breakaway_ok=bool(limits.LimitFlags & 0x0800) if ok else None,
                                  silent_breakaway_ok=bool(limits.LimitFlags & 0x1000) if ok else None))
                break
            finally:
                k.CloseHandle(handle)
    finally:
        k.CloseHandle(owner)
    print(json.dumps(dict(read_only=True, owner_pid=a.owner_pid, target_pids=a.target_pid, matched_jobs=found,
                         mutation=False, processes_stopped=False)))


if __name__ == "__main__":
    main()
