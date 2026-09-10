"""Lexical layer: tokens, scripts, lemmas, identifiers, stoplists (ASSOCIATIVE_MEMORY.md §2.1).

Per-language, per-token by script: `pymorphy3` for Cyrillic (every candidate lemma under
homonymy — the Segalovich rule, recall-biased on purpose), `simplemma` for Latin-script
text. Identifiers (`snake_case`, `camelCase`, dotted, digit-bearing) are indexed verbatim
and never lemmatized. Both lemmatizers are loaded lazily so a process that never indexes
Russian never pays for the dictionary.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, Iterator

LEX_VERSION = "lex-1"

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_]|[^\W\d_]+(?:-[^\W\d_]+)*|\d+(?:[.,]\d+)*", re.UNICODE)
_IDENT_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z]+_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z_]+\d[A-Za-z0-9_]*|[A-Z]{2,}[a-z0-9]*[A-Z][A-Za-z0-9]*)$")

# Small built-in stoplists. The IDF-based auto stop (glossary) catches the corpus-specific
# filler; these catch the words that are filler in every corpus.
STOP_EN = frozenset("""
a an the and or but if then than so of to in on at by for with from as into onto about over
under between through during before after above below up down out off again further once
here there when where why how all any both each few more most other some such no nor not
only own same too very can will just should now is are was were be been being have has had
having do does did doing i me my we our you your he him his she her it its they them their
what which who whom this that these those am s t don ll re ve d m isn aren wasn weren hasn
haven hadn doesn didn won wouldn couldn shouldn also because while would could may might
must shall get got let like one two also yes
""".split())
STOP_RU = frozenset("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее
мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был
него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо
ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот
того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда
зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть
том нельзя такой им более всегда конечно всю между это который которые которая слушай ага
угу ок окей типа вообще просто очень
""".split())


def script_of(token: str) -> str:
    """'cyr' | 'lat' | 'num' | 'other' — by the first letter that decides it."""
    for ch in token:
        if ch.isdigit():
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("CYRILLIC"):
            return "cyr"
        if name.startswith("LATIN"):
            return "lat"
        if ch.isalpha():
            return "other"
    return "num" if token and token[0].isdigit() else "other"


def is_identifier(token: str) -> bool:
    return bool(_IDENT_RE.match(token))


def tokenize(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield (surface, start, end) over *text*. Surfaces keep their case."""
    for m in _TOKEN_RE.finditer(text or ""):
        yield m.group(0), m.start(), m.end()


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=200_000)
def lemmas(surface: str) -> tuple[str, ...]:
    """Every candidate lemma of *surface* (lowercased), the surface form included.

    Identifiers: the verbatim token only. Cyrillic: all pymorphy3 normal forms. Latin:
    simplemma's English lemma. Numbers: as written.
    """
    if is_identifier(surface):
        return (surface,)
    low = surface.lower()
    sc = script_of(surface)
    out: list[str] = [low]
    try:
        if sc == "cyr":
            for p in _morph().parse(low):
                nf = (p.normal_form or "").lower()
                if nf and nf not in out:
                    out.append(nf)
        elif sc == "lat":
            import simplemma
            # simplemma returns proper nouns capitalized ("haifa" → "Haifa"); a lemma is a key.
            nf = (simplemma.lemmatize(low, lang="en") or "").lower()
            if nf and nf != low:
                out.append(nf)
    except Exception:
        pass
    return tuple(out)


def is_stop(term: str) -> bool:
    return term in STOP_EN or term in STOP_RU or len(term) < 2


def terms_of(text: str, *, keep_stop: bool = False) -> list[str]:
    """Flat list of index terms for *text*: every candidate lemma of every token."""
    out: list[str] = []
    for surface, _s, _e in tokenize(text):
        for t in lemmas(surface):
            if keep_stop or not is_stop(t):
                out.append(t)
    return out


def content_terms(text: str) -> set[str]:
    return set(terms_of(text))


def canon_terms(text: str) -> set[str]:
    """One canonical lemma per token (the dictionary form when there is one, else the
    surface) — for overlap measures, where counting every variant of one word inflates
    the score (`acquired` + `acquire` from one token)."""
    out: set[str] = set()
    for surface, _s, _e in tokenize(text):
        ls = lemmas(surface)
        t = ls[-1] if len(ls) > 1 else ls[0]
        if not is_stop(t):
            out.add(t)
    return out


def script_mix(text: str) -> dict:
    """Character counts per script — the basis of the per-script token estimate (§1.5)."""
    cyr = lat = other = 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("CYRILLIC"):
            cyr += 1
        elif name.startswith("LATIN"):
            lat += 1
        else:
            other += 1
    return {"cyr": cyr, "lat": lat, "other": other}


def dominant_script(text: str) -> str:
    m = script_mix(text)
    if not any(m.values()):
        return "lat"
    return max(m, key=m.get)


def overlap(a_terms: Iterable[str], b_terms: Iterable[str]) -> float:
    """Fraction of *a*'s terms present in *b* (containment)."""
    a = set(a_terms)
    if not a:
        return 0.0
    b = set(b_terms)
    return len(a & b) / len(a)
