# Deployment Guide

## Server — first-time setup (done once on the GPU machine)

> **DGX Spark / unified-memory box:** read `SPARK_LOADING.md` first. Model loads that take
> minutes are a known trap with a shipped fix; `cd server/inference && ../.venv/bin/python -m core.fast_load --bench`
> tells you in 30 s whether the fix matters on the box in front of you.

**1. Clone the repo and run the install script:**

```bash
git clone <repo-url> ~/ava
cd ~/ava/server
bash install.sh
```

The script creates `server/.venv` with `--system-site-packages` (so a
system-level PyTorch / unsloth installation is visible without reinstalling
it) and installs `inference/requirements.txt` into the venv.

**2. Start the watchdog:**

```bash
cd ~/ava/server
.venv/bin/python watchdog.py [--port 8765] [--mgmt-port 8766]
```

The watchdog immediately launches the inference server as a subprocess and
exposes an HTTP management API on port 8766. Run it inside `tmux` or `screen`
so it survives your SSH session.

---

## Day-to-day workflow (from your dev machine)

**1. Save the host once:**

```bash
cd tools
python deploy.py setup --host 192.168.1.x
```

Settings are saved to `~/.avadeploy.json`.

**2. Ship a code update:**

```bash
git push                  # publish your changes to the repo
python deploy.py pull     # tell the watchdog to fast-forward + restart
```

The watchdog stops the running inference server, runs `git pull --ff-only` on
its own checkout, and relaunches it with the new code. The watchdog never
accepts code over the wire — it only pulls what you have already pushed. No SSH
required. (The same action is available in the client UI's Chat tab as the
**Update and restart server** button.)

**3. Other commands:**

```bash
python deploy.py status    # check watchdog and inference server state
python deploy.py restart   # tell watchdog to restart (no git pull)
python deploy.py logs      # tail server.log (inference output)
python deploy.py logs --lines 100
```

---

## Deploy tool reference

```
setup    save host/port settings to ~/.avadeploy.json
pull     tell watchdog to `git pull` + restart  (default)
restart  tell watchdog to restart the inference server
status   query watchdog and inference server state
logs     fetch recent server.log from watchdog
```

Any flag overrides the saved config for that run:

```bash
python deploy.py status --host other-machine
python deploy.py pull --mgmt-port 9000
```

---

## Watchdog HTTP API

The watchdog runs on port 8766 (configurable with `--mgmt-port`).

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | `{"running": bool, "pid": int\|null, "uptime_s": float\|null, "role": "inference"}` |
| `POST` | `/pull` | Stop inference, `git pull --ff-only` the repo, relaunch — returns the git output |
| `POST` | `/restart` | Kill and relaunch the inference server |
| `GET` | `/logs?lines=N` | Last N lines of `server.log` (default 50) |

---

## Client setup

```bash
cd client
pip install -r requirements.txt
python main.py --server ws://HOST:8765
```

The `--server` flag is optional — without it the client reads `server_url` from
`config.json` at the repo root, or defaults to `ws://localhost:8765`.
