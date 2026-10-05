"""One two-arm, at-most-32MiB send-buffer probe; never edits active relay code."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import traceback

HOST = "connect.bjb2.seetacloud.com"
PORT = 33953
ROOT = "/root/autodl-tmp/prototype_guided_evidence_selection_20261003"


def dump(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def worker():
    import paramiko
    req = json.load(sys.stdin)
    result = dict(arm=req["arm"], requested_send_buffer=req["send_buffer"], completed=False,
                  payload_bytes=16*1024**2, started_at=datetime.now(timezone.utc).isoformat())
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client = paramiko.SSHClient()
    started = time.perf_counter()
    try:
        result["default_send_buffer_before_connect"] = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        if req["send_buffer"] is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, req["send_buffer"])
        sock.setsockopt(socket.IPPROTO_IP, 14, 0)  # Windows IP_DONTFRAGMENT; same active-relay route workaround.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(20)
        sock.connect((req["ipv4"], PORT))
        result.update(send_buffer_after_connect=sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
                      receive_buffer_after_connect=sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
                      dont_fragment=sock.getsockopt(socket.IPPROTO_IP, 14),
                      local_address=list(sock.getsockname()), remote_address=list(sock.getpeername()))
        dump(req["result_path"], result)
        client.load_host_keys(str(Path.home()/".ssh/known_hosts"))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(HOST, port=PORT, username="root", key_filename=req["key"], sock=sock,
                       look_for_keys=False, allow_agent=False, timeout=20, auth_timeout=20, banner_timeout=20)
        transport = client.get_transport()
        transport.set_keepalive(5)
        result.update(send_buffer_after_auth=sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
                      cipher=transport.local_cipher, ssh_compression=transport.local_compression,
                      connect_auth_seconds=time.perf_counter()-started)
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(60)
        parent = ROOT+"/transport_probe"
        try:
            sftp.mkdir(parent)
        except OSError:
            if not sftp.stat(parent):
                raise
        directory = parent+"/"+req["run_id"]
        try:
            sftp.mkdir(directory)
        except OSError:
            if not sftp.stat(directory):
                raise
        destination = directory+"/"+req["arm"]+".bin"
        try:
            sftp.stat(destination)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("Probe destination already exists; no retry or overwrite")
        source = Path(req["source"])
        if source.stat().st_size != result["payload_bytes"]:
            raise ValueError("Payload must be exactly the frozen 16MiB prefix")
        if hashlib.sha256(source.read_bytes()).hexdigest() != req["sha256"]:
            raise ValueError("Frozen existing-bundle prefix changed")
        dump(req["result_path"], result)
        t0 = time.perf_counter()
        attributes = sftp.put(str(source), destination, confirm=True)
        result.update(upload_seconds=time.perf_counter()-t0, remote_bytes=attributes.st_size,
                      send_buffer_after_upload=sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF))
        sftp.close()
        stdin, stdout, stderr = client.exec_command("sha256sum -- "+shlex.quote(destination), timeout=20)
        stdin.close()
        checksum = stdout.read().decode("utf-8").split()[0]
        error = stderr.read().decode("utf-8")
        if stdout.channel.recv_exit_status() or error or checksum != req["sha256"]:
            raise AssertionError("Received SHA256 verification failed")
        if attributes.st_size != result["payload_bytes"]:
            raise AssertionError("Received byte count differs")
        result.update(completed=True, received_sha256=checksum, exact_received_bytes=True,
                      mib_per_second=16/result["upload_seconds"], remote_path=destination)
    except BaseException as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        result["elapsed_seconds"] = time.perf_counter()-started
        dump(req["result_path"], result)
        client.close()
        sock.close()
    print(json.dumps(result), flush=True)
    return 0 if result["completed"] else 1


def relay_snapshot(root, pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, 0, pid)
    alive = False
    if handle:
        code = ctypes.c_ulong()
        if kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            alive = code.value == 259
        kernel.CloseHandle(handle)
    progress = root/"runs/RUN_973a07a938934b81bdd46796fcd203da/PROGRESS.json"
    return dict(at=datetime.now(timezone.utc).isoformat(), pid=pid, alive=alive,
                progress=json.loads(progress.read_text()) if progress.exists() else None)


def main():
    if os.name != "nt":
        raise RuntimeError("This probe must originate from the current Windows relay host")
    out = Path(__file__).resolve().parent
    root = out.parents[1]
    record = json.loads((out/"run.json").read_text())
    if record["status"] != "prepared":
        raise RuntimeError("One attempt per arm only; probe already used")
    key = Path.home()/".ssh/paper_disc_bjb2_33953"
    if not key.exists():
        key = Path.home()/".ssh/id_ed25519"
    # Resolve once, so the two sockets use the same destination IP and host key.
    ipv4 = socket.gethostbyname(HOST)
    record.update(status="running", command=[sys.executable, str(Path(__file__).resolve())],
                  ipv4=ipv4, key_authentication=True, strict_known_hosts=True,
                  relay_before=relay_snapshot(root, record["main_relay_pid"]))
    dump(out/"run.json", record)
    start = subprocess.STARTUPINFO()
    start.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    start.wShowWindow = subprocess.SW_HIDE
    rows = []
    for arm, buffer in (("baseline", None), ("send4mib", 4*1024**2)):
        result_path = out/(arm+".json")
        req = dict(arm=arm, send_buffer=buffer, ipv4=ipv4, key=str(key),
                   run_id=record["run_id"], source=str(out/"prefix_16MiB.bin"),
                   sha256=record["prefix_sha256"], result_path=str(result_path))
        try:
            process = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(req), capture_output=True, text=True, encoding="utf-8", timeout=90,
                creationflags=subprocess.CREATE_NO_WINDOW, startupinfo=start)
            (out/(arm+".stderr.log")).write_text(process.stderr, encoding="utf-8")
            result = json.loads(result_path.read_text())
            result["process_returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            result = json.loads(result_path.read_text()) if result_path.exists() else dict(arm=arm)
            result.update(completed=False, timeout_seconds=90, no_retry=True)
            dump(result_path, result)
        rows.append(result)
        print(json.dumps(result), flush=True)
    complete = all(row.get("completed") for row in rows)
    report = dict(completed=complete, arms=rows, fixed_source=record["source_bundle"],
                  source_prefix_sha256=record["prefix_sha256"], extra_payload_limit_bytes=32*1024**2,
                  repeats_per_arm=1, automatic_retry=False, order="baseline then send4mib",
                  concurrent_relay=True, relay_before=record["relay_before"],
                  relay_after=relay_snapshot(root, record["main_relay_pid"]),
                  active_code_or_configuration_changed=False, model_execution=False,
                  limitations=["Main relay remains active and contends for the same route.",
                               "Single sequential observation per arm; time-varying network not controlled.",
                               "32MiB is application payload; SSH/SFTP/TCP headers add protocol overhead."])
    if complete:
        report["speedup"] = rows[0]["upload_seconds"] / rows[1]["upload_seconds"]
        report["greater_than_twofold"] = report["speedup"] > 2
    dump(out/"REPORT.json", report)
    record.update(status="completed" if complete else "incomplete",
                  finished_at=datetime.now(timezone.utc).isoformat(), report="REPORT.json")
    dump(out/"run.json", record)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(worker())
    main()
