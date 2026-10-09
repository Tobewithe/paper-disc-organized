"""Finite owned-process WDDM sampling, never a scheduler or GPU workload."""
import json
import os
import subprocess
import threading
import time


class WDDMMemoryProbe:
    def __init__(self, interval_seconds=1.0):
        self.pid = os.getpid()
        self.interval = interval_seconds
        self.samples = []
        self.stop_event = threading.Event()
        self.started = time.time()
        self.thread = threading.Thread(target=self._observe, daemon=True)
    def start(self):
        self.thread.start()
    def _observe(self):
        command = ("$ErrorActionPreference='Stop'; "
                   "Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | "
                   "Where-Object { $_.Name -match 'pid_" + str(self.pid) + "_' } | "
                   "Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted | ConvertTo-Json -Compress")
        while not self.stop_event.is_set():
            began = time.time()
            process = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                     capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            rows = None
            error = process.stderr.strip() or None
            if process.returncode == 0 and process.stdout.strip():
                parsed = json.loads(process.stdout)
                rows = parsed if isinstance(parsed, list) else [parsed]
            self.samples.append({"query_started_at_unix": began, "query_finished_at_unix": time.time(),
                                 "return_code": process.returncode, "rows": rows, "error": error})
            if self.stop_event.wait(self.interval):
                break
    def stop(self):
        self.endpoint_stopped = time.time()
        self.stop_event.set()
    def finish(self):
        began = time.perf_counter()
        self.thread.join()
        available = [sample for sample in self.samples if sample["rows"]]
        def peak(field):
            values = [sum(int(row[field]) for row in sample["rows"]) for sample in available]
            return max(values) if values else None
        return {"pid": self.pid, "sampling_interval_seconds": self.interval,
                "observer_started_at_unix": self.started, "endpoint_stopped_at_unix": self.endpoint_stopped,
                "samples": self.samples, "dedicated_sampled_peak_bytes": peak("DedicatedUsage"),
                "shared_sampled_peak_bytes": peak("SharedUsage"), "committed_sampled_peak_bytes": peak("TotalCommitted"),
                "all_samples_present": bool(available), "observer_join_seconds_outside_endpoint": time.perf_counter() - began,
                "scope": "owned-process WDDM rows summed across returned adapter instances; sampled peak, not exact instantaneous peak",
                "observer_overhead_exactly_measured": False,
                "last_inflight_query_can_finish_after_endpoint": True}
