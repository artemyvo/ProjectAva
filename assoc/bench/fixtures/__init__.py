"""Shared fixtures for the benches (ASSOCIATIVE_MEMORY.md §9): a synthetic product's
documentation tree in two versions and two platforms, chat transcripts (EN + RU), a small
code project (Python + C) and a news article. Everything expected is known by construction.
"""

from __future__ import annotations

import json

PRODUCT = "nimbus"

# ----- documentation --------------------------------------------------------------------------

def error_code_table(n: int = 200, *, version: str) -> str:
    rows = ["| Code | Meaning | Action |", "|------|---------|--------|"]
    for i in range(n):
        code = f"E{100 + i}"
        meaning = ["Timeout", "Auth failed", "Quota exceeded", "Bad config", "Peer unreachable"][i % 5]
        action = ["Increase connect_timeout", "Rotate the token", "Raise the plan limit", "Check nimbus.yaml", "Check the peer's firewall"][i % 5]
        if version == "2.0" and code == "E142":
            action = "Set retry_backoff (new in 2.0)"
        rows.append(f"| {code} | {meaning} {i} | {action} |")
    return "\n".join(rows)


def docs_tree(version: str, platform: str) -> list[dict]:
    """Pages of the Nimbus Gateway docs for one version + platform: (meta, text)."""
    v = version
    timeout_default = "30 seconds" if v == "1.0" else "60 seconds"
    svc = "systemd" if platform == "linux" else "the Windows Service Manager"
    logs = "/var/log/nimbus" if platform == "linux" else r"C:\ProgramData\Nimbus\logs"
    insecure = ("\n\n### The --insecure flag\n\nThe `--insecure` flag disables certificate checks. Use it only in a lab.\n"
                if v == "1.0" else "")
    pages = []
    pages.append({"key": f"docs/{platform}/install.md", "title": "Install", "text": f"""# Install

## Requirements

Nimbus Gateway {v} requires 2 GB of RAM and a supported {platform} release. The gateway
listens on port 8443 by default.

## Installing on {platform}

Install the package, then register the service with {svc}:

```bash
nimbus-ctl install --platform {platform}
```

After installation the service starts automatically. Logs are written to `{logs}`.

## Upgrading

Stop the service, install the new package, and run `nimbus-ctl migrate`. Configuration in
`nimbus.yaml` is preserved across upgrades.
"""})
    pages.append({"key": f"docs/{platform}/config.md", "title": "Configuration", "text": f"""# Configuration

## nimbus.yaml

All settings live in `nimbus.yaml`. The file is read once at startup; changes require a
restart of the service.

## Timeouts

### connect_timeout

`connect_timeout` sets how long the gateway waits for an upstream connection. It defaults to
{timeout_default}. Values are given in seconds.

### Logging of timeouts

Timeouts are logged at WARN level in the gateway log. Set `log_level: debug` to also log the
retry attempts.
{"" if v == "1.0" else '''
### retry_backoff

`retry_backoff` (new in 2.0) sets the initial backoff between retries. It defaults to 2
seconds and doubles on every attempt up to `retry_backoff_max`.
'''}
## Authentication

### Rotating the token

To rotate the API token:

1. Open the admin console.
2. Choose *Security* → *Tokens*.
3. Click *Rotate* and copy the new token.
4. Restart the service.
{insecure}"""})
    pages.append({"key": f"docs/{platform}/errors.md", "title": "Error codes", "text": f"""# Error codes

The gateway reports errors with a code of the form `E<number>`. Look the code up below.

## Table of error codes

{error_code_table(200, version=v)}
"""})
    pages.append({"key": f"docs/{platform}/faq.md", "title": "FAQ", "text": f"""# FAQ

## How do I rotate the token?

1. Open the admin console.
2. Choose Security → Tokens.
3. Click Rotate.

## Does the gateway support HTTP/3?

No. Nimbus Gateway {v} speaks HTTP/1.1 and HTTP/2 only.

## Where are the logs?

Logs are written to `{logs}`.
"""})
    pages.append({"key": f"docs/{platform}/release-notes.md", "title": "Release notes", "text": f"""# Release notes

## v{v}

- Added `connect_timeout` with a default of {timeout_default}.
{"- Added `retry_backoff` with exponential backoff." if v == "2.0" else "- Added the `--insecure` flag for lab use."}
{"- Removed the `--insecure` flag; use a lab certificate instead." if v == "2.0" else ""}
- Improved the {platform} installer.
"""})
    for p in pages:
        p["meta"] = {"key": p["key"], "title": p["title"], "version": v, "platform": platform, "product": PRODUCT,
                     "date": "2026-03-01" if v == "1.0" else "2026-08-01"}
    return pages


def all_docs() -> list[dict]:
    out = []
    for v in ("1.0", "2.0"):
        for plat in ("linux", "windows"):
            out.extend(docs_tree(v, plat))
    return out


# ----- chats ---------------------------------------------------------------------------------

def chat_kestrel() -> str:
    return json.dumps({"user": "Pavel", "exchanges": [
        {"user_prompt": "Are you guys hiring at Kestrel? I've been looking since June.", "assistant_response": "Not right now, unfortunately. There are no open positions at Kestrel this quarter.", "ts": "2026-08-14T10:00"},
        {"user_prompt": "Pity. I interviewed at Brightmem back in 2019 but went to Linqua instead.", "assistant_response": "I remember you mentioning Brightmem. Noam Keller founded it.", "ts": "2026-08-14T10:05"},
    ]}, ensure_ascii=False)


def chat_noam() -> str:
    return json.dumps({"user": "Artemy", "exchanges": [
        {"user_prompt": "Noam Keller offered me a CTO position at his new startup, Starling. I declined — the timing is wrong.", "assistant_response": "That is a big offer to turn down. Is the position still open?", "ts": "2026-07-02T19:00"},
        {"user_prompt": "As far as I know, yes. They are still looking.", "assistant_response": "Noted.", "ts": "2026-07-02T19:04"},
    ]}, ensure_ascii=False)


def chat_ru() -> str:
    return json.dumps({"user": "Артемий", "exchanges": [
        {"user_prompt": "Мы потеряли ключ от квартиры, пришлось вызывать слесаря.", "assistant_response": "Слесарь сделал новый замок?", "ts": "2026-06-20T20:00"},
        {"user_prompt": "Да. А ещё на даче нашли родник — ключ бьёт прямо из-под камня.", "assistant_response": "Хорошая вода?", "ts": "2026-06-20T20:10"},
        {"user_prompt": "Отличная. Сталь на старом ноже, кстати, стала ржаветь.", "assistant_response": "Нож стоит смазать.", "ts": "2026-06-20T20:15"},
    ]}, ensure_ascii=False)


def chat_ru_protocol_lines() -> str:
    """A hand-written witness output for chat_ru (for parser/anchor tests)."""
    return "\n".join([
        "[fact] (about: Артемий) (class: event) (chunk: 1) (entities: квартира, слесарь) (when: 2026-06-20) Потерял ключ от квартиры и вызвал слесаря.",
        "[fact] (about: Артемий) (class: standing) (chunk: 2) (entities: дача, родник) На даче есть родник — ключ бьёт из-под камня.",
        "[fact] (about: Артемий) (class: event) (chunk: 3) (entities: нож) Сталь на старом ноже стала ржаветь.",
        "[fact] (about: self) (class: stated) (chunk: 3) Нож стоит смазать.",
    ])


# ----- code ----------------------------------------------------------------------------------

PY_MODULE = '''"""Connection helpers for the demo client."""
import os
import socket
from demo.errors import ConnectError


def resolve(host):
    """Resolve a host name to an address. TODO: cache results"""
    return socket.gethostbyname(host)


class Client:
    """A client that connects to the gateway."""

    def __init__(self, host, timeout=30):
        self.host = host
        self.timeout = timeout
        self.api_key = "sk-THISISASECRETKEY0123456789"

    def connect(self):
        """Open the connection, resolving the host first."""
        addr = resolve(self.host)
        sock = socket.create_connection((addr, 8443), timeout=self.timeout)
        if sock is None:
            raise ConnectError(self.host)
        return sock

    def close(self):
        return None
'''

C_MODULE = '''#include <stdio.h>
#include "gateway.h"

struct conn { int fd; int timeout; };

/* Resolve and open a connection. */
static int open_conn(struct conn *c, const char *host) {
    c->fd = gw_resolve(host);
    if (c->fd < 0) {
        return -1;
    }
    return gw_connect(c->fd, c->timeout);
}

int main(void) {
    struct conn c = { -1, 30 };
    /* TODO: read the timeout from the config */
    printf("%d\\n", open_conn(&c, "gateway"));
    return 0;
}
'''

# ----- news ----------------------------------------------------------------------------------

NEWS_ARTICLE = """# Gateway vendor Nimbus acquires Starling

Nimbus Networks announced on 2026-08-20 that it has acquired Starling, the startup founded by
Noam Keller in 2025. The deal was valued at 40 million dollars, the company said.

Starling builds a memory-pooling layer for GPU clusters. Keller previously founded Brightmem,
which was acquired by Halden in 2021.

"We are thrilled," said Nimbus CEO Dana Levi. Analysts called the price high.
"""

NEWS_PROTOCOL_LINES = "\n".join([
    "[fact] (about: Nimbus Networks) (class: event) (chunk: 1) (entities: Starling, Noam Keller) (when: 2026-08-20) Nimbus Networks announced it has acquired Starling.",
    "[fact] (about: Starling) (class: standing) (chunk: 1) (entities: Noam Keller) (when: 2025) Starling is a startup founded by Noam Keller in 2025.",
    "[fact] (about: Starling) (class: stated) (chunk: 1) The deal was valued at 40 million dollars.",
    "[fact] (about: Starling) (class: standing) (chunk: 2) Starling builds a memory-pooling layer for GPU clusters.",
    "[fact] (about: Brightmem) (class: event) (chunk: 2) (entities: Noam Keller, Halden) (when: 2021) Brightmem was acquired by Halden in 2021.",
    "[fact] (about: Dana Levi) (class: stated) (chunk: 3) (entities: Nimbus) Dana Levi said 'We are thrilled'.",
    "[fact] (about: Mars) (class: standing) (chunk: 3) Mars has two moons.",   # planted: unsupported
])


# ----- milestone 2: cross-lingual, sources, contests ------------------------------------------

def chat_haifa_en() -> str:
    return json.dumps({"user": "Artemy", "exchanges": [
        {"user_prompt": "By the way, my sister lives in Haifa now, near the Bahai gardens.", "assistant_response": "Haifa is a good city for that. Does she like it?", "ts": "2026-05-10T18:00"},
        {"user_prompt": "She does. She works at the Technion as a lab manager.", "assistant_response": "Noted.", "ts": "2026-05-10T18:03"},
    ]}, ensure_ascii=False)


def chat_haifa_ru() -> str:
    return json.dumps({"user": "Артемий", "exchanges": [
        {"user_prompt": "Сестра переехала в Хайфу, живёт рядом с Бахайскими садами.", "assistant_response": "Хайфа — хороший город. Ей нравится?", "ts": "2026-06-01T19:00"},
        {"user_prompt": "Да. Она работает в Технионе, заведует лабораторией.", "assistant_response": "Понял.", "ts": "2026-06-01T19:02"},
    ]}, ensure_ascii=False)


CHAT_HAIFA_EN_LINES = "\n".join([
    "[fact] (about: Artemy) (class: standing) (chunk: 1) (entities: Haifa, Bahai gardens) Artemy's sister lives in Haifa, near the Bahai gardens.",
    "[fact] (about: Artemy) (class: standing) (chunk: 2) (entities: Technion) Artemy's sister works at the Technion as a lab manager.",
])
CHAT_HAIFA_RU_LINES = "\n".join([
    "[fact] (about: Артемий) (class: standing) (chunk: 1) (entities: Хайфа, Бахайские сады) Сестра Артемия живёт в Хайфе, рядом с Бахайскими садами.",
    "[fact] (about: Артемий) (class: standing) (chunk: 2) (entities: Технион) Сестра Артемия работает в Технионе и заведует лабораторией.",
])

ALIASES = {"артемий": "artemy", "хайфа": "haifa", "технион": "technion"}

# A syndicated article and its mirror (near-copies), plus a contradicting fifth source.
ARTICLE_SYNDICATED = """# Starling deal valued at 40 million

Nimbus Networks has acquired Starling, the memory-pooling startup founded by Noam Keller.
The deal was valued at 40 million dollars, the company said on 2026-08-20.

Starling builds a memory-pooling layer for GPU clusters. Keller previously founded Brightmem.
"""
ARTICLE_MIRROR = ARTICLE_SYNDICATED.replace("# Starling deal valued at 40 million", "# Starling deal valued at 40 million (syndicated)")
ARTICLE_CONTRA = """# Starling price disputed

Sources close to the deal say Nimbus Networks did not pay 40 million dollars for Starling;
the price was closer to 25 million dollars.
"""
SYNDICATED_LINES = "\n".join([
    "[fact] (about: Nimbus Networks) (class: event) (chunk: 1) (entities: Starling, Noam Keller) (when: 2026-08-20) Nimbus Networks acquired Starling.",
    "[fact] (about: Starling) (class: stated) (chunk: 1) (entities: Nimbus Networks) The deal was valued at 40 million dollars.",
    "[fact] (about: Starling) (class: standing) (chunk: 2) Starling builds a memory-pooling layer for GPU clusters.",
])
CONTRA_LINES = "\n".join([
    "[fact] (about: Starling) (class: stated) (chunk: 1) (entities: Nimbus Networks) The deal was not valued at 40 million dollars.",
    "[fact] (about: Starling) (class: stated) (chunk: 1) (entities: Nimbus Networks) The price was closer to 25 million dollars.",
])

# Two libraries defining `connect`; an index page that lists everything; order + negation pairs.
LIB_A_DOC = """# Alpha client

## connect

`connect(host, port)` opens a connection to the Alpha broker. It must be called before `publish`.

## publish

`publish(topic, payload)` sends a message. `publish` must not be called before `connect`.
"""
LIB_B_DOC = """# Beta client

## connect

`connect(url)` opens a websocket to the Beta gateway and returns a session token.
"""
INDEX_PAGE = """# Reference index

- connect — opens a connection
- publish — sends a message
- subscribe — receives messages
- disconnect — closes the connection
- reconnect — reopens a connection
- ping — checks the connection
- status — reports the connection state
"""
LIB_A_LINES = "\n".join([
    "[fact] (about: connect) (class: signature) (chunk: 1) (entities: Alpha broker) `connect(host, port)` opens a connection to the Alpha broker.",
    "[fact] (about: connect) (class: spec) (chunk: 1) (entities: publish) connect must be called before publish.",
    "[fact] (about: publish) (class: signature) (chunk: 2) `publish(topic, payload)` sends a message.",
    "[fact] (about: publish) (class: spec) (chunk: 2) (entities: connect) publish must not be called before connect.",
])
LIB_B_LINES = "\n".join([
    "[fact] (about: connect) (class: signature) (chunk: 1) (entities: Beta gateway) `connect(url)` opens a websocket to the Beta gateway and returns a session token.",
])
INDEX_LINES = "\n".join([
    "[fact] (about: connect) (class: standing) (chunk: 1) connect opens a connection.",
    "[fact] (about: publish) (class: standing) (chunk: 1) publish sends a message.",
    "[fact] (about: subscribe) (class: standing) (chunk: 1) subscribe receives messages.",
    "[fact] (about: disconnect) (class: standing) (chunk: 1) disconnect closes the connection.",
    "[fact] (about: reconnect) (class: standing) (chunk: 1) reconnect reopens a connection.",
    "[fact] (about: ping) (class: standing) (chunk: 1) ping checks the connection.",
    "[fact] (about: status) (class: standing) (chunk: 1) status reports the connection state.",
])
# Restatements of one fact (merge), an order pair (must stay two), a negated restatement (contests).
RESTATE_DOC_A = "# Notes A\n\nThe gateway listens on port 8443 by default. The service must start before the agent.\n"
RESTATE_DOC_B = "# Notes B\n\nBy default the gateway listens on port 8443. The agent must start before the service.\n"
RESTATE_DOC_C = "# Notes C\n\nThe gateway does not listen on port 8443 by default.\n"
RESTATE_A_LINES = "\n".join([
    "[fact] (about: gateway) (class: standing) (chunk: 1) (entities: port 8443) The gateway listens on port 8443 by default.",
    "[fact] (about: service) (class: spec) (chunk: 1) (entities: service, agent) The service must start before the agent.",
])
RESTATE_B_LINES = "\n".join([
    "[fact] (about: gateway) (class: standing) (chunk: 1) (entities: port 8443) By default the gateway listens on port 8443.",
    "[fact] (about: service) (class: spec) (chunk: 1) (entities: agent, service) The agent must start before the service.",
])
RESTATE_C_LINES = "\n".join([
    "[fact] (about: gateway) (class: standing) (chunk: 1) (entities: port 8443) The gateway does not listen on port 8443 by default.",
])


# ----- milestone 3: the aha story with typed relations (§4, §9 bench/aha) --------------------

CHAT_KESTREL_LINES = "\n".join([
    "[fact] (about: Pavel) (class: event) (chunk: 1) (entities: Kestrel) (when: 2026-08-14) (rel: asked_about(Pavel, Kestrel)) Pavel asked whether Kestrel has open positions.",
    "[fact] (about: Pavel) (class: event) (chunk: 1) (entities: job) (when: 2026-06) (rel: looking_for(Pavel, job)) Pavel has been looking for a job since June.",
    "[fact] (about: self) (class: stated) (chunk: 1) (entities: Kestrel) There are no open positions at Kestrel this quarter.",
    "[fact] (about: Pavel) (class: event) (chunk: 2) (entities: Brightmem, Linqua) (when: 2019) (rel: interviewed_at(Pavel, Brightmem)) Pavel interviewed at Brightmem in 2019 but went to Linqua instead.",
    "[fact] (about: self) (class: stated) (chunk: 2) (entities: Noam Keller, Brightmem) Noam Keller founded Brightmem.",
])
CHAT_NOAM_LINES = "\n".join([
    "[fact] (about: Artemy) (class: event) (chunk: 1) (entities: Noam Keller, Starling, CTO position) (when: 2026-07-02) (rel: offered(Noam Keller, CTO position)) Noam Keller offered Artemy a CTO position at his startup Starling.",
    "[fact] (about: Artemy) (class: event) (chunk: 1) (entities: CTO position) (when: 2026-07-02) (rel: declined(Artemy, CTO position)) Artemy declined the CTO position because the timing is wrong.",
    "[fact] (about: Artemy) (class: stated) (chunk: 2) (entities: Starling) Starling is still looking for a CTO.",
])
NEWS_PROTOCOL_LINES_REL = NEWS_PROTOCOL_LINES.replace(
    "[fact] (about: Brightmem) (class: event) (chunk: 2) (entities: Noam Keller, Halden) (when: 2021) Brightmem was acquired by Halden in 2021.",
    "[fact] (about: Brightmem) (class: event) (chunk: 2) (entities: Noam Keller, Halden) (when: 2021) (rel: acquired(Halden, Brightmem)) Brightmem was acquired by Halden in 2021.\n"
    "[fact] (about: Noam Keller) (class: standing) (chunk: 2) (entities: Brightmem) (rel: founded(Noam Keller, Brightmem)) Noam Keller founded Brightmem.")


# ----- milestone 4: pivots (§6) — two senses of ключ / стали, a root family, a sound pair ----

def pivot_corpus() -> list[dict]:
    """Articles (paragraph-primary) planting each sense with enough contexts to cluster."""
    key_door = [
        "Мы потеряли ключ от квартиры, и пришлось вызывать слесаря, чтобы открыть дверь.",
        "Запасной ключ от входной двери лежит у соседей, замок старый и капризный.",
        "Слесарь сделал новый ключ и поменял замок на двери квартиры.",
        "Ключ от подъезда сломался в замке, дверь пришлось выламывать.",
        "Он оставил ключ в замке двери и ушёл, квартира осталась открытой.",
        "Дубликат ключа от квартиры сделали за десять минут в мастерской у метро.",
        "Ключ от дома нашли в кармане старой куртки, замок так и не поменяли.",
    ]
    key_spring = [
        "На даче в лесу нашли родник: ключ бьёт прямо из-под камня, вода ледяная.",
        "Ключ у ручья пересох к августу, вода в лесу ушла глубже под камни.",
        "Из-под скалы бьёт ключ, вода чистая и холодная, родник известен всей деревне.",
        "Лесной ключ питает ручей, вода из родника идёт к озеру мимо камней.",
        "Рядом с дачей бьёт ключ, и вода в роднике не замерзает даже зимой.",
        "Мы набрали воды из ключа в лесу; родник холодный, камни вокруг мокрые.",
        "Ключ под горой дал начало ручью, вода в нём холодная круглый год.",
    ]
    steel = [
        "Сталь на старом ноже стала ржаветь, лезвие покрылось рыжими пятнами.",
        "Нож из углеродистой стали ржавеет быстрее, чем клинок из нержавеющей стали.",
        "Клинок выкован из дамасской стали, лезвие держит заточку годами.",
        "Сталь для ножа закалили дважды, лезвие стало твёрже и не тупится.",
        "Кухонный нож из мягкой стали быстро тупится, лезвие приходится править.",
        "Ржавчина съела сталь на клинке старого ножа, лезвие пришлось выбросить.",
    ]
    became = [
        "Они стали друзьями после долгой поездки и с тех пор не расставались.",
        "Дни стали короче, вечера стали холоднее, осень пришла в город.",
        "Мы стали видеться реже, когда он переехал в другой город.",
        "После переезда соседи стали ближе, чем старые друзья.",
        "Цены стали выше, а зарплаты остались прежними, люди стали экономить.",
        "Дети стали взрослыми и разъехались по разным городам.",
    ]
    bells = [
        "Колокол на старой колокольне звонил к вечерне каждый день.",
        "Колокольчик на двери лавки звенел, когда входили покупатели.",
        "Колокольня собора видна с любой улицы города.",
        "Время колокольчиков — так называлась песня, которую он пел под гитару.",
        "Большой колокол отлили в прошлом веке, колокольчики звенят тоньше.",
    ]
    shops = [
        "Магазин на углу закрылся, продукты теперь покупают в супермаркете.",
        "The magazine published a long interview with the founder in its spring issue.",
        "В магазине у дома продавали свежий хлеб и молоко по утрам.",
        "She reads the magazine on the train; the spring issue covered startups.",
    ]

    def doc(key, title, paras):
        return {"key": key, "title": title, "kind": "article", "date": "2026-05-01",
                "text": f"# {title}\n\n" + "\n\n".join(paras) + "\n"}
    return [doc("pivot/keys", "Ключи", key_door + key_spring), doc("pivot/steel", "Сталь и стали", steel + became),
            doc("pivot/bells", "Колокола", bells), doc("pivot/shops", "Магазины", shops)]
