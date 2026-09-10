#!/usr/bin/env python3
"""
Ava deployment CLI — communicates with the running watchdog over HTTP.

The watchdog must already be running on the remote host. Start it once manually:
    cd ~/ava/server && python watchdog.py [--port 8765] [--mgmt-port 8766]

Usage:
    python deploy.py setup --host HOST [--port 8765] [--mgmt-port 8766]
    python deploy.py pull       # tell watchdog to `git pull` + restart  (default)
    python deploy.py restart    # tell watchdog to restart the inference server
    python deploy.py status     # query watchdog and inference server state
    python deploy.py logs       # fetch recent server.log from watchdog

The watchdog never accepts code over the wire — push your changes to the repo
(`git push`) and run `pull` to fast-forward the server's checkout and restart.

Config is saved to ~/.avadeploy.json.
Per-run overrides: any --flag takes precedence over saved config.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_CONFIG_PATH = Path.home() / ".avadeploy.json"


# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        return json.loads(_CONFIG_PATH.read_text())
    return {}


def _save_config(cfg: dict) -> None:
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"Config saved to {_CONFIG_PATH}")


def _merge(saved: dict, args) -> dict:
    cfg = dict(saved)
    for key in ("host", "port", "mgmt_port"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    cfg.setdefault("port", 8765)
    cfg.setdefault("mgmt_port", 8766)
    return cfg


def _require_host(cfg: dict) -> None:
    if not cfg.get("host"):
        sys.exit("Missing 'host'. Run: python deploy.py setup --host HOST")


def _mgmt_url(cfg: dict, path: str) -> str:
    return f"http://{cfg['host']}:{cfg['mgmt_port']}{path}"


# ------------------------------------------------------------------ #
# HTTP helpers                                                         #
# ------------------------------------------------------------------ #

def _get(url: str, timeout: int = 15) -> dict | str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if "json" in resp.headers.get("Content-Type", "") else body.decode()
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach watchdog at {url}: {e}")


def _post(url: str, data: bytes = b"", content_type: str = "application/json", timeout: int = 120) -> dict:
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(data)))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach watchdog at {url}: {e}")


# ------------------------------------------------------------------ #
# Commands                                                             #
# ------------------------------------------------------------------ #

def cmd_setup(args) -> None:
    cfg = _merge(_load_config(), args)
    _save_config(cfg)
    print("Settings:")
    for k in ("host", "port", "mgmt_port"):
        print(f"  {k}: {cfg.get(k)}")


def cmd_pull(args) -> None:
    cfg = _merge(_load_config(), args)
    _require_host(cfg)
    url = _mgmt_url(cfg, "/pull")
    print(f"Requesting git pull + restart at {url} …")
    result = _post(url, timeout=180)
    if result.get("output"):
        print(result["output"])
    if result.get("ok"):
        print("Server pulled the latest code — inference server is restarting.")
    else:
        sys.exit(f"Watchdog returned error: {result}")


def cmd_restart(args) -> None:
    cfg = _merge(_load_config(), args)
    _require_host(cfg)
    url = _mgmt_url(cfg, "/restart")
    print(f"Sending restart to {url} …")
    result = _post(url)
    if result.get("ok"):
        print("Inference server restarting.")
    else:
        sys.exit(f"Watchdog returned error: {result}")


def cmd_status(args) -> None:
    cfg = _merge(_load_config(), args)
    _require_host(cfg)
    data = _get(_mgmt_url(cfg, "/status"))
    if isinstance(data, dict):
        running = data.get("running", False)
        pid = data.get("pid")
        uptime = data.get("uptime_s")
        role = data.get("role", "?")
        print(f"Watchdog:  UP  (mgmt port {cfg['mgmt_port']})")
        status_line = f"Inference: {'UP' if running else 'DOWN'}  (port {cfg['port']}, role={role}"
        if running:
            status_line += f", pid={pid}, uptime={uptime}s"
        print(status_line + ")")
    else:
        print(data)


def cmd_logs(args) -> None:
    cfg = _merge(_load_config(), args)
    _require_host(cfg)
    print(f"--- last {args.lines} lines ---")
    print(_get(_mgmt_url(cfg, f"/logs?lines={args.lines}"), timeout=15))


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deploy.py",
        description="Ava deployment CLI — communicates with the watchdog over HTTP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "commands:\n"
            "  setup    save host/port settings to ~/.avadeploy.json\n"
            "  pull     tell watchdog to `git pull` + restart  (default)\n"
            "  restart  tell watchdog to restart the inference server\n"
            "  status   query watchdog and inference server state\n"
            "  logs     fetch recent server.log from watchdog\n"
        ),
    )
    parser.add_argument(
        "command", nargs="?", default="pull",
        choices=["setup", "pull", "restart", "status", "logs"],
    )
    parser.add_argument("--host",      help="Remote hostname or IP")
    parser.add_argument("--port",      type=int, help="Inference server WebSocket port (default: 8765)")
    parser.add_argument("--mgmt-port", type=int, dest="mgmt_port",
                        help="Watchdog management HTTP port (default: 8766)")
    parser.add_argument("--lines",     type=int, default=50,
                        help="Lines to show with 'logs' (default: 50)")

    args = parser.parse_args()

    dispatch = {
        "setup":   cmd_setup,
        "pull":    cmd_pull,
        "restart": cmd_restart,
        "status":  cmd_status,
        "logs":    cmd_logs,
    }

    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
