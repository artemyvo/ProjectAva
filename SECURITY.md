# Security

Ava is research software meant to run on a machine (or a few) inside a **private,
trusted network that is not reachable from the internet**. It is not hardened for
anything else. If any of its ports can be reached from the internet, or from a
network you do not control, everything below is open to whoever connects.

## What listens, by default

Every server binds `0.0.0.0` (all interfaces) unless you pass `--host`, and none of
them authenticates callers, except the public API when `api.api_key` is set.

| Port | Process | What an unauthenticated caller can do |
| --- | --- | --- |
| 8765 | Inference WebSocket server | Chat as any user name; list, read, load and delete every stored conversation; rewrite chat history; start reflection runs; load and unload models. |
| 8766 | Watchdog management API | `git pull` and restart the server (running whatever the configured git remote serves); run the offline jobs, including `wipe`, which deletes all state; read the server log. |
| 8767 | Inference HTTP sidecar | Download the whole chat corpus, the reflection archive and the trained adapter weights (`/export`, `/snapshot/export`, `/adapter/export`, `/chats/export`); import chats; change the model load precision; talk to Ava through the peer-gossip chat endpoint (on by default). |
| 8000 | Public OpenAI-compatible API | Spend GPU time talking to Ava. She answers from her memory of your conversations, so private facts can come out in her replies. On by default, and open unless `api.api_key` is set. |

## Data at rest

Conversations, distilled memory, per-person portraits and reflection archives are
stored as plain text under `server/data/`, `server/inference/data/` and
`server/reflections/`. The LoRA adapters are trained on those conversations and can
reproduce parts of them. Treat adapter weights, runnable snapshots
(`server/snapshot_state.py`) and Migrate-tab bundles as being as sensitive as the chats
themselves.

## Recommended setup

- Keep the host unreachable from the internet. For remote access, use a VPN
  (WireGuard, Tailscale, …) or an SSH tunnel; never port-forward these ports.
- Firewall ports 8765–8767 and 8000 so that only the machines that need them can
  connect.
- When the client runs on the same machine as the server, start the server with
  `--host 127.0.0.1`.
- In `server/server_config.json`, set `api.enabled: false` unless you need the public
  API. If you do need it, set `api.api_key` and bind it with `api.host`.
- Set `gossip.enabled: false` unless you are connecting two Ava instances.
- Outbound access is still needed to download models and for the wander/TIL jobs,
  which fetch pages from the wikis listed in `server/til/wiki_sources.json`.

## Reporting a vulnerability

Please report security problems privately by email to artemyvo@gmail.com, not in a
public issue. This is a one-person research project: fixes are best-effort and there
is no bug bounty.
