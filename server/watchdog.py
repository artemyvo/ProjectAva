#!/usr/bin/env python3
"""Ava watchdog — a thin, never-upgraded process supervisor.

The watchdog is the root of the process tree on the GPU box: it launches and
kills ``inference/server.py``, and NOTHING on the box can restart the watchdog
itself without SSH. So it must contain only generic, stable logic — process
supervision plus the git-pull bootstrap that keeps everything ELSE upgradeable.
Anything project-specific (data layout, config schema, training params, tar
bundle shapes) lives in the inference server (which git-pulls freely) or in
git-tracked repo scripts the watchdog invokes generically.

Two responsibilities remain here, both irreducibly watchdog-side:

  1. Process supervision + code update — launch/kill/status/restart, and
     ``git pull --ff-only`` + relaunch (the ONLY code-update path; the watchdog
     never accepts code over the wire, it only fast-forwards the checkout it has).

  2. A GENERIC offline-job runner. Some jobs need the GPU to themselves (LoRA
     training) or need the model unloaded (state wipe), so they
     can only run while inference is DOWN — which only the watchdog can arrange.
     Rather than hardcode each job, the watchdog reads a git-tracked manifest
     (``watchdog_jobs.json``) at request time and runs the named command. Adding
     or changing a job is a repo edit + ``git pull`` — the watchdog is untouched.
     Jobs are mutually exclusive (they all need the GPU or a stopped server), so
     there is one job slot; the caller forms the job's CLI flags (POST body
     ``args``) and the watchdog stays semantics-free.

HTTP endpoints (default port 8766):
  GET  /status              → {running, pid, uptime_s, role, hostname, data_dir, job}
  POST /restart             → kill + relaunch
  POST /pull                → stop inference, `git pull --ff-only`, relaunch
  POST /job/<name>          → run manifest job <name>; body {args?: [str, ...]}
                              appended verbatim to the job's cmd. Async jobs return
                              immediately (poll /job/status + /job/progress); a
                              `sync` job blocks and returns the job's JSON result.
  GET  /job/status          → {name, running, started_at, finished_at, returncode,
                              stage?, last_message?, result?}
  GET  /job/progress?after_seq=N → structured per-step progress events (for jobs
                              that write a progress journal)
  POST /job/stop            → best-effort; most jobs have no mid-cycle halt
  GET  /logs?lines=N        → last N lines of server.log (default 50). Stays here
                              because it must work while inference is DOWN.

The read-mostly, project-specific endpoints (artifacts / export / chats sync /
precision) moved to the inference HTTP sidecar (``inference/core/mgmt_http.py``,
default port 8767): they don't need inference down, so they upgrade freely.

Usage:
    python watchdog.py [--host 0.0.0.0] [--port 8765] [--mgmt-port 8766] [--http-port 8767]
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Structured progress-journal reader (stdlib-only; a generic seq'd-JSONL reader,
# not training-specific). Serves GET /job/progress for jobs that write a journal.
try:
    from training.train_progress import (
        last_event as _last_progress_event,
        read_events as _read_progress_events,
    )
except Exception:  # pragma: no cover - keeps the watchdog importable in odd setups
    def _read_progress_events(path, after_seq: int = 0) -> list:
        return []

    def _last_progress_event(path):
        return None

_WATCHDOG_DIR = Path(__file__).resolve().parent
_REPO_DIR = _WATCHDOG_DIR.parent  # git repository root (server/ lives one level down)
_INFERENCE_DIR = _WATCHDOG_DIR / "inference"
_DATA_DIR = _INFERENCE_DIR / "data"
_JOBS_FILE = _WATCHDOG_DIR / "watchdog_jobs.json"  # git-tracked offline-job manifest
_LOG_FILE = _WATCHDOG_DIR / "server.log"
_PYTHON = sys.executable

_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()
_start_time: float | None = None
_server_host: str = "0.0.0.0"
_server_port: int = 8765
_http_port: int = 8767  # inference HTTP sidecar port (passed through on launch)

# One job slot: offline jobs are mutually exclusive (each needs the GPU or a
# stopped server), so a single lock + state covers them all.
_job_lock = threading.Lock()
_job_state: dict = {
    "name": None, "running": False, "started_at": None, "finished_at": None,
    "returncode": None, "progress_file": None, "log": None, "result": None,
}


# ──────────────────────────────────────────────────────────────────────────────
# Process management
# ──────────────────────────────────────────────────────────────────────────────

def _launch() -> None:
    """Start the inference server subprocess. Caller must hold _proc_lock."""
    global _proc, _start_time
    log_fh = open(_LOG_FILE, "a", buffering=1, encoding="utf-8", errors="replace")
    _proc = subprocess.Popen(
        [_PYTHON, "server.py", "--host", _server_host, "--port", str(_server_port),
         "--http-port", str(_http_port)],
        cwd=str(_INFERENCE_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    _start_time = time.monotonic()
    print(f"[watchdog] launched inference server pid={_proc.pid}", flush=True)


def _kill() -> None:
    """Kill the running server if any. Caller must hold _proc_lock."""
    global _proc, _start_time
    if _proc is None:
        return
    try:
        _proc.kill()
        _proc.wait(timeout=10)
    except Exception:
        pass
    _proc = None
    _start_time = None
    print("[watchdog] inference server stopped.", flush=True)


def _is_running() -> bool:
    return _proc is not None and _proc.poll() is None


def _log_server(message: str) -> None:
    """Append a timestamped watchdog line to server.log.

    The inference process is stopped during a job, so it can't log to server.log
    itself. The watchdog writes the job lifecycle here so anyone tailing server.log
    sees the stop → run → relaunch arc (and the outcome) inline. Best-effort.
    """
    try:
        with open(_LOG_FILE, "a", buffering=1, encoding="utf-8", errors="replace") as fh:
            fh.write(f"[watchdog {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


def _tail_log(n: int) -> str:
    if not _LOG_FILE.exists():
        return "(log file not found)"
    with open(_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])


def _git_pull() -> dict:
    """Stop inference, `git pull` the repository, relaunch with the new code.

    This is the only code-update path: the operator pushes from their dev machine,
    then asks the watchdog to pull. The watchdog never accepts code over the wire —
    it only fast-forwards the checkout it already has. Inference is stopped first so
    the relaunch picks up the new code; it is always relaunched, even if the pull
    fails, so the server never stays down. Returns {ok, returncode, output}.
    """
    with _proc_lock:
        _kill()
        try:
            proc = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(_REPO_DIR),
                capture_output=True, text=True, timeout=120,
            )
            result = {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "output": ((proc.stdout or "") + (proc.stderr or "")).strip(),
            }
        except Exception as e:
            result = {"ok": False, "returncode": -1, "output": str(e)}
        _launch()

    _log_server(f"git pull rc={result['returncode']} (inference server relaunched): "
                + result["output"].replace("\n", " ⏎ ")[:500])
    print(f"[watchdog] git pull rc={result['returncode']}", flush=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Generic offline-job runner
# ──────────────────────────────────────────────────────────────────────────────

def _load_jobs() -> dict:
    """Read the git-tracked job manifest fresh (so `git pull` adds jobs live)."""
    try:
        data = json.loads(_JOBS_FILE.read_text(encoding="utf-8"))
        jobs = data.get("jobs") if isinstance(data, dict) else None
        return jobs if isinstance(jobs, dict) else {}
    except Exception as e:
        print(f"[watchdog] could not read {_JOBS_FILE.name}: {e}", flush=True)
        return {}


def _progress_path(spec: dict) -> Path | None:
    """Resolve a job's progress-journal path (relative to server/), if any."""
    p = spec.get("progress")
    return (_WATCHDOG_DIR / p) if p else None


def _log_path(name: str, spec: dict) -> Path:
    """Resolve a job's stdout/stderr log path (relative to server/)."""
    return _WATCHDOG_DIR / (spec.get("log") or f"{name}.log")


def _reset_progress(path: Path | None) -> None:
    """Truncate a job's progress journal before it runs (best-effort).

    A job may truncate its own journal only lazily (after heavy ML imports), so
    until then GET /job/progress would serve the *previous* run's events. Clearing
    it here closes that window.
    """
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    except Exception:
        pass


def _build_argv(spec: dict, args: list) -> list[str]:
    """The full subprocess argv: the python executable + the manifest cmd + args.

    `cmd` is the argv after the executable (a module via ["-m", "pkg.mod"] or a
    script filename). `args` is the caller-formed flag list, appended verbatim —
    the caller owns the job's semantics, the watchdog stays dumb.
    """
    cmd = spec.get("cmd") or []
    return [_PYTHON, *[str(c) for c in cmd], *[str(a) for a in args]]


def _run_async_job(name: str, spec: dict, args: list) -> None:
    """Stop inference (free the GPU), run a long job, relaunch. Runs in a thread.

    The inference server holds the model in VRAM, so it must come down before a
    job that needs the GPU (train) or the model unloaded can run. The job may
    repoint server_config.json (adapter_id / model_id); relaunching picks it up.
    """
    progress = _progress_path(spec)
    logf = _log_path(name, spec)
    argv = _build_argv(spec, args)
    rc = -1
    try:
        # Let the inference server flush any pending logs after the POST returns,
        # then stop it to free the GPU.
        time.sleep(2)
        with _proc_lock:
            _kill()
        if spec.get("reset_progress"):
            _reset_progress(progress)
        _log_server(f"job '{name}' started (inference server stopped). "
                    f"cmd={' '.join(argv[1:])} — progress in {logf.name}")
        with open(logf, "a", buffering=1, encoding="utf-8", errors="replace") as log_fh:
            log_fh.write(f"\n===== job '{name}' started {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"cmd={' '.join(argv[1:])} =====\n")
            log_fh.flush()
            proc = subprocess.run(argv, cwd=str(_WATCHDOG_DIR), stdout=log_fh,
                                  stderr=subprocess.STDOUT)
            rc = proc.returncode
            log_fh.write(f"===== job '{name}' exited rc={rc} =====\n")
        print(f"[watchdog] job '{name}' finished rc={rc}", flush=True)
        outcome = None
        if progress is not None:
            last = _last_progress_event(progress)
            outcome = last.get("message") if isinstance(last, dict) else None
        _log_server(f"job '{name}' finished rc={rc}"
                    + (f" — {outcome}" if outcome else "")
                    + "; relaunching inference server.")
    except Exception as e:
        print(f"[watchdog] job '{name}' error: {e}", flush=True)
        _log_server(f"job '{name}' errored: {e}; relaunching inference server.")
    finally:
        with _proc_lock:
            _launch()
        with _job_lock:
            _job_state.update(running=False, finished_at=time.time(), returncode=rc)


def _run_sync_job(name: str, spec: dict, args: list) -> dict:
    """Stop inference, run a fast job to completion, relaunch, return its result.

    For disaster-recovery-class jobs (wipe): file work is fast, and the slow part —
    loading the (possibly new) base model — happens in the relaunched subprocess,
    so the caller just reconnects once the server is back. The job prints a single
    JSON result object to stdout (its last line); we parse + return it.
    """
    argv = _build_argv(spec, args)
    logf = _log_path(name, spec)
    rc = -1
    result: dict = {}
    with _proc_lock:
        _kill()
        _log_server(f"job '{name}' (sync) started (inference server stopped). "
                    f"cmd={' '.join(argv[1:])}")
        try:
            proc = subprocess.run(argv, cwd=str(_WATCHDOG_DIR),
                                  capture_output=True, text=True)
            rc = proc.returncode
            # Persist full output for post-mortem, then parse the JSON result line.
            try:
                with open(logf, "a", buffering=1, encoding="utf-8", errors="replace") as fh:
                    fh.write(f"\n===== job '{name}' (sync) {time.strftime('%Y-%m-%d %H:%M:%S')} "
                             f"rc={rc} =====\n{proc.stderr or ''}\n{proc.stdout or ''}\n")
            except Exception:
                pass
            result = _parse_result(proc.stdout, rc)
        except Exception as e:
            result = {"ok": False, "error": str(e), "returncode": rc}
            print(f"[watchdog] job '{name}' (sync) error: {e}", flush=True)
        _launch()

    _log_server(f"job '{name}' (sync) finished rc={rc} ok={result.get('ok')}; "
                f"inference server relaunched.")
    print(f"[watchdog] job '{name}' (sync) finished rc={rc}", flush=True)
    with _job_lock:
        _job_state.update(running=False, finished_at=time.time(),
                          returncode=rc, result=result)
    return result


def _parse_result(stdout: str, rc: int) -> dict:
    """Parse a sync job's JSON result from the last JSON line of its stdout."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    data.setdefault("returncode", rc)
                    data.setdefault("ok", rc == 0)
                    return data
            except Exception:
                continue
    return {"ok": rc == 0, "returncode": rc,
            "output": (stdout or "")[-500:]}


def _job_status() -> dict:
    """Current job state + the stage/last message from its progress journal."""
    with _job_lock:
        state = dict(_job_state)
    pf = state.get("progress_file")
    if pf:
        last = _last_progress_event(Path(pf))
        if isinstance(last, dict):
            state["stage"] = last.get("stage")
            state["last_message"] = last.get("message")
            state["last_status"] = last.get("status")
    return state


# ──────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ──────────────────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress default access logging

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, status: int, text: str) -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            params = json.loads(body) if body else {}
            return params if isinstance(params, dict) else {}
        except Exception:
            return {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/status":
            with _proc_lock:
                running = _is_running()
                pid = _proc.pid if running else None
                uptime = round(time.monotonic() - _start_time, 1) if running and _start_time else None
            with _job_lock:
                job = {"name": _job_state["name"], "running": _job_state["running"],
                       "returncode": _job_state["returncode"]}
            self._json(200, {
                "running": running, "pid": pid, "uptime_s": uptime, "role": "inference",
                "hostname": socket.gethostname(),
                "data_dir": str(_DATA_DIR.resolve()),
                "job": job,
            })

        elif parsed.path == "/job/status":
            self._json(200, _job_status())

        elif parsed.path == "/job/progress":
            qs = parse_qs(parsed.query)
            try:
                after_seq = max(0, int(qs.get("after_seq", ["0"])[0]))
            except ValueError:
                after_seq = 0
            with _job_lock:
                running = bool(_job_state.get("running"))
                returncode = _job_state.get("returncode")
                pf = _job_state.get("progress_file")
            events = _read_progress_events(Path(pf), after_seq=after_seq) if pf else []
            self._json(200, {"running": running, "returncode": returncode, "events": events})

        elif parsed.path == "/logs":
            qs = parse_qs(parsed.query)
            try:
                n = max(1, int(qs.get("lines", ["50"])[0]))
            except ValueError:
                n = 50
            self._text(200, _tail_log(n))

        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/restart":
            with _proc_lock:
                _kill()
                _launch()
            self._json(200, {"ok": True})

        elif parsed.path == "/pull":
            result = _git_pull()
            self._json(200 if result.get("ok") else 500, result)

        elif parsed.path == "/job/stop":
            # Most offline jobs have no mid-cycle halt (killing a train mid-write
            # could corrupt state), so this is an honest no-op ack: the client
            # detaches its poll, the job runs to completion.
            with _job_lock:
                running = bool(_job_state.get("running"))
                name = _job_state.get("name")
            self._json(200, {"ok": True, "stopped": False, "job": name,
                             "note": "no mid-cycle halt; the job will finish. "
                                     "The inference server relaunches when it ends."})

        elif parsed.path.startswith("/job/"):
            name = parsed.path[len("/job/"):]
            self._start_job(name)

        else:
            self._json(404, {"error": "not found"})

    def _start_job(self, name: str) -> None:
        spec = _load_jobs().get(name)
        if not spec:
            self._json(404, {"error": f"unknown job '{name}'"})
            return
        params = self._read_json_body()
        args = params.get("args", [])
        if not isinstance(args, list) or not all(isinstance(a, (str, int, float)) for a in args):
            self._json(400, {"error": "`args` must be a list of scalars"})
            return

        with _job_lock:
            if _job_state["running"]:
                self._json(409, {"error": f"job '{_job_state['name']}' is already in "
                                          f"progress; cannot start '{name}'"})
                return
            _job_state.update(
                name=name, running=True, started_at=time.time(), finished_at=None,
                returncode=None, result=None,
                progress_file=str(_progress_path(spec)) if _progress_path(spec) else None,
                log=str(_log_path(name, spec)),
            )

        # Truncate the progress journal NOW, before the poller can attach: the job
        # only truncates it lazily (after heavy ML imports), so until then
        # /job/progress would serve the *previous* run's events — the poller would
        # render the stale log, then suppress the real new events whose seq hasn't
        # yet overtaken the stale max. (The async thread resets again in-flight; a
        # second truncate of an already-empty file is a no-op.)
        if spec.get("reset_progress"):
            _reset_progress(_progress_path(spec))

        if spec.get("sync"):
            result = _run_sync_job(name, spec, args)
            self._json(200 if result.get("ok") else 500, result)
        else:
            threading.Thread(target=_run_async_job, args=(name, spec, args),
                             daemon=True).start()
            self._json(200, {"ok": True, "job": name, "started": True,
                             "log": str(_log_path(name, spec))})


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _server_host, _server_port, _http_port

    parser = argparse.ArgumentParser(description="Ava watchdog")
    parser.add_argument("--host",      default="0.0.0.0", help="Bind address for both servers")
    parser.add_argument("--port",      type=int, default=8765, help="Inference server WebSocket port")
    parser.add_argument("--mgmt-port", type=int, default=8766, dest="mgmt_port",
                        help="Watchdog HTTP management port")
    parser.add_argument("--http-port", type=int, default=8767, dest="http_port",
                        help="Inference HTTP sidecar port (passed through on launch)")
    args = parser.parse_args()

    _server_host = args.host
    _server_port = args.port
    _http_port = args.http_port

    with _proc_lock:
        _launch()

    # Threaded so a long job-status poll never serializes behind another request,
    # and a sync job (which blocks its handler) doesn't freeze the UI's /status
    # polls. Handlers guard all shared state with _proc_lock / _job_lock.
    mgmt = ThreadingHTTPServer((args.host, args.mgmt_port), _Handler)
    mgmt.daemon_threads = True
    print(f"[watchdog] management API on http://{args.host}:{args.mgmt_port}", flush=True)
    try:
        mgmt.serve_forever()
    except KeyboardInterrupt:
        with _proc_lock:
            _kill()


if __name__ == "__main__":
    main()
