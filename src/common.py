"""Shared helpers: colour palette, text wrapping and dataset lookup.

Used by the visualizer now and by the query engine later.  Fuzzy matching uses
``rapidfuzz`` when installed and falls back to the standard library otherwise.
"""
from __future__ import annotations

import difflib
import math
import re
import textwrap

import networkx as nx

from src.graph_builder import normalize_text, out_by_rel, dataset_nodes  # noqa: F401

try:  # optional speed/quality boost
    from rapidfuzz import fuzz, process
    from rapidfuzz import utils as rf_utils
    HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_RAPIDFUZZ = False

PALETTE = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948",
           "#B07AA1", "#FF9DA7", "#9C755F", "#86BCB6", "#D37295", "#A0CBE8",
           "#8CD17D", "#B6992D", "#FABFD2", "#499894"]
GREY = "#BAB0AC"

# colour, vis.js shape, (min size, max size) per non-dataset node type
TYPE_STYLE: dict[str, tuple[str, str, tuple[int, int]]] = {
    "Dataset": ("#4E79A7", "dot", (10, 42)),
    "Paper": ("#C8C8C8", "dot", (5, 5)),
    "Field": ("#F28E2B", "diamond", (16, 30)),
    "Person": ("#59A14F", "triangle", (9, 20)),
    "Organization": ("#B07AA1", "square", (10, 22)),
    "Keyword": ("#76B7B2", "triangleDown", (7, 18)),
}

# colour, dashed? per relation
RELATION_STYLE: dict[str, tuple[str, bool]] = {
    "REFERENCED_BY": ("#BDBDBD", False),
    "HAS_THEME": ("#F28E2B", False),
    "CREATED_BY": ("#59A14F", False),
    "PUBLISHED_BY": ("#B07AA1", False),
    "CONTRIBUTED_BY": ("#B07AA1", True),
    "HAS_KEYWORD": ("#76B7B2", False),
    "CO_CITED": ("#7F7F7F", False),
}


def community_color(cid: int) -> str:
    return GREY if cid < 0 else PALETTE[cid % len(PALETTE)]


def wrap_text(text: str, width: int = 70, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return textwrap.fill(text, width=width)


def log_scale(value: float, vmax: float, lo: float, hi: float) -> float:
    """Map ``value`` in [0, vmax] to [lo, hi] on a log scale."""
    if vmax <= 0:
        return lo
    return lo + (hi - lo) * math.log1p(max(value, 0.0)) / math.log1p(vmax)


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def _fuzzy(query: str, labels: dict[str, str], limit: int) -> list[tuple[str, float]]:
    if HAVE_RAPIDFUZZ:
        res = process.extract(query, labels, scorer=fuzz.WRatio,
                              processor=rf_utils.default_process,
                              limit=limit, score_cutoff=55)
        return [(key, float(score)) for _label, score, key in res]
    qn = normalize_text(query)
    scored: list[tuple[str, float]] = []
    for key, label in labels.items():
        ln = normalize_text(label)
        if not ln or not qn:
            continue
        s = difflib.SequenceMatcher(None, qn, ln).ratio() * 100
        if qn in ln:
            s = max(s, 90.0)
        if s >= 55:
            scored.append((key, s))
    scored.sort(key=lambda t: -t[1])
    return scored[:limit]


def find_nodes(G: nx.MultiDiGraph, query: str, node_type: str = "Dataset",
               limit: int = 5) -> list[tuple[str, float]]:
    """Best-matching nodes of ``node_type`` for a free-text name.

    Exact (normalized) matches come first, most-connected first, so duplicate
    titles such as several 'Pinakes' records resolve to the biggest one.
    A bare number or ``dataset:123`` selects a dataset directly.
    """
    q = query.strip()
    if not q:
        return []
    if node_type == "Dataset":
        direct = q if q.startswith("dataset:") else (f"dataset:{q}" if q.isdigit() else None)
        if direct and direct in G and G.nodes[direct]["type"] == "Dataset":
            return [(direct, 100.0)]
    labels = {n: d["label"] for n, d in G.nodes(data=True)
              if d["type"] == node_type and d.get("label")}
    qn = normalize_text(q)
    exact = sorted((n for n, l in labels.items() if normalize_text(l) == qn),
                   key=lambda n: -G.degree(n))
    out: list[tuple[str, float]] = [(n, 100.0) for n in exact]
    seen = set(exact)
    fuzzy = [(k, s) for k, s in _fuzzy(q, labels, limit * 3) if k not in seen]
    fuzzy.sort(key=lambda t: (-t[1], -G.degree(t[0])))
    out.extend(fuzzy)
    return out[:limit]


_STOP = {"the", "a", "an", "of", "and", "or", "for", "in", "on", "to", "with", "data", "dataset"}


def search_datasets(G: nx.MultiDiGraph, text: str, limit: int = 25
                    ) -> list[tuple[str, float]]:
    """Keyword search over dataset titles, keywords, fields and descriptions."""
    tokens = [t for t in normalize_text(text).split() if t not in _STOP] \
        or normalize_text(text).split()
    if not tokens:
        return []
    results: list[tuple[str, float]] = []
    for d in dataset_nodes(G):
        a = G.nodes[d]
        title = normalize_text(a["label"])
        title_tokens = set(title.split())
        kw = " ".join(G.nodes[k]["label"] for k in out_by_rel(G, d, "HAS_KEYWORD"))
        fld = " ".join(normalize_text(G.nodes[f]["label"]) for f in out_by_rel(G, d, "HAS_THEME"))
        desc = set(normalize_text(a.get("description", "")).split())
        score, hit = 0.0, 0
        for t in tokens:
            s = 0.0
            if t in title_tokens or any(w.startswith(t) for w in title_tokens):
                s += 3
            if t in kw:
                s += 2
            if t in fld:
                s += 2
            if t in desc:
                s += 1
            if s:
                hit += 1
            score += s
        if hit * 2 >= len(tokens) and score > 0:
            results.append((d, score + 0.1 * math.log1p(a.get("n_papers", 0))))
    results.sort(key=lambda t: -t[1])
    return results[:limit]
