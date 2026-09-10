"""bench/split — the splitter on its own, and the two-budget fit rule (§1.5)."""

from assoc.budget import Budget, stamp_fits
from assoc.kinds import get
from assoc.kinds.markdown import ROWGROUP_ROWS
from assoc.bench import fixtures as fx

PAGE = """# Guide

Intro sentence that is long enough to be a paragraph of its own.

## Install

### Linux

Run the installer, then enable the service with the following command:

```bash
sudo systemctl enable nimbus
```

Short.

#### Logs

Logs live in /var/log/nimbus and rotate daily.

### Windows

Run the MSI.

## Codes

| Code | Meaning |
|------|---------|
""" + "\n".join(f"| E{i} | meaning {i} |" for i in range(20)) + """

## Steps

1. Open the console.
2. Click rotate.
3. Restart.
"""


def _units():
    text, units = get("tech_doc").split(PAGE, {"key": "k/guide.md"})
    return text, units


def test_sections_are_primaries_with_heading_paths():
    _, units = _units()
    prim = {" > ".join(u.path): u for u in units if u.role == "primary"}
    assert "Guide > Install > Linux" in prim
    assert "Guide > Install > Linux > Logs" in prim
    assert "Guide > Install > Windows" in prim
    assert prim["Guide > Install > Linux > Logs"].text.startswith("#### Logs")
    # A heading with no own content is an ancestor, never a primary.
    anc = [u for u in units if u.role == "ancestor"]
    assert any(u.path == ["Guide", "Install"] for u in anc)


def test_never_cut_inside_a_code_block_and_lead_in_kept():
    _, units = _units()
    linux = next(u for u in units if u.role == "primary" and u.path[-1] == "Linux")
    pieces = [u for u in units if u.parent_id == linux.chunk_id]
    code = next(u for u in pieces if u.unit_type == "code")
    assert "enable the service with the following command" in code.text and "```bash" in code.text and "systemctl" in code.text
    # The one-word fragment "Short." is merged upward (no piece of its own).
    assert not any(u.text.strip() == "Short." for u in pieces)


def test_table_row_groups_repeat_the_header():
    _, units = _units()
    groups = [u for u in units if u.unit_type == "rowgroup"]
    assert len(groups) == -(-20 // ROWGROUP_ROWS)
    assert all(u.keys["header"] == ["Code", "Meaning"] for u in groups)
    assert all(u.keys["header_line"].startswith("| Code") for u in groups)
    assert groups[0].text.startswith("| E0 |") and groups[-1].text.endswith("| E19 | meaning 19 |")


def test_lists_stay_whole():
    _, units = _units()
    steps = next(u for u in units if u.role == "primary" and u.path[-1] == "Steps")
    assert "1. Open the console." in steps.text and "3. Restart." in steps.text


def test_chunk_ids_are_structural_not_positional():
    _, units_a = _units()
    _, units_b = get("tech_doc").split("Preamble line inserted above.\n\n" + PAGE, {"key": "k/guide.md"})
    ids_a = {" > ".join(u.path): u.chunk_id for u in units_a if u.role == "primary"}
    ids_b = {" > ".join(u.path): u.chunk_id for u in units_b if u.role == "primary"}
    for p, cid in ids_a.items():
        assert ids_b.get(p) == cid, p          # every section keeps its id though every span moved


def test_two_budgets_same_chunks_only_fits_change():
    _, units = _units()
    big = stamp_fits(units, Budget(total=8000))
    small = stamp_fits(units, Budget(total=120))
    rb, rs = big.pop("__report__"), small.pop("__report__")
    assert set(big) == set(small)
    assert rb["oversize"] == 0 and rs["oversize"] >= 1 and rs["split"] >= 1
    # Under the small budget the table section is oversize and its row groups carry the injection.
    codes = next(u for u in units if u.role == "primary" and u.path[-1] == "Codes")
    assert big[codes.chunk_id]["injectable"] and not small[codes.chunk_id]["injectable"]
    pieces = [u for u in units if u.parent_id == codes.chunk_id]
    assert any(small[p.chunk_id]["injectable"] for p in pieces)


def test_chat_exchange_never_split():
    text, units = get("chat").split(fx.chat_ru(), {"key": "c"})
    f = stamp_fits(units, Budget(total=20), split_oversize=False)
    rep = f.pop("__report__")
    assert rep["oversize"] == len(units) and rep["split"] == 0 and rep["still_oversize"] == len(units)


def test_news_chunks_are_paragraphs():
    text, units = get("news").split(fx.NEWS_ARTICLE, {"key": "n"})
    prim = [u for u in units if u.role == "primary"]
    assert len(prim) == 3
    assert all(u.path[0].startswith("Gateway vendor") for u in prim)
    for u in prim:
        assert text[u.span[0]:u.span[1]] == u.text
