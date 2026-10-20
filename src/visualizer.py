"""Pillar 2 - interactive visualization (PyVis -> self-contained HTML).

Rendering the whole graph (640k nodes) would be an unreadable hairball, so
every view is a *focused subgraph*:

    overview    one node per community (+ fields), edges = co-citation / shared field
    map         ALL datasets plus the fields, organizations, people and keywords
                that connect them (papers summarised as co-citation edges)
    dataset     a dataset, its most co-cited datasets, shared papers, metadata
    community   the top datasets of one community
    field       the top datasets of one field of study
    search      datasets matching a keyword query

Colour = community (or field/origin) for datasets, fixed colours per entity
type otherwise; node size = log-scaled degree centrality; hover tooltips show
node attributes and edge relationships.  Each page has a search box and
show/hide checkboxes per node group.

The code is split into pure-NetworkX *scene builders* (``scene_*``) and a thin
:func:`render` step that hands the scene to PyVis.

Usage:
    python -m src.visualizer overview
    python -m src.visualizer map
    python -m src.visualizer dataset "ImageNet"
    python -m src.visualizer community 2
    python -m src.visualizer field "Economics"
    python -m src.visualizer search "protein structure"
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import sys
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import networkx as nx

from src.common import (GREY, PALETTE, RELATION_STYLE, TYPE_STYLE, community_color,
                        find_nodes, log_scale, search_datasets, wrap_text)
from src.graph_builder import dataset_nodes, load_graph, out_by_rel

log = logging.getLogger("visualizer")


# --------------------------------------------------------------------------- #
# Scene = plain-data description of one page
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    title: str
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    groups: dict[str, str] = field(default_factory=dict)   # legend: group -> colour
    notes: list[str] = field(default_factory=list)


class Styler:
    """Colours, sizes and tooltips derived from the graph."""

    def __init__(self, G: nx.MultiDiGraph, H: nx.Graph, color_by: str = "community"):
        self.G, self.H, self.color_by = G, H, color_by
        self.datasets = dataset_nodes(G)
        self.max_dc = max((G.nodes[n].get("degree_centrality", 0.0)
                           for n in self._non_papers()), default=0.0) * 1e5
        best: dict[int, str] = {}
        for d in self.datasets:
            cid = G.nodes[d].get("community", -1)
            if cid not in best or G.nodes[d]["n_papers"] > G.nodes[best[cid]]["n_papers"]:
                best[cid] = d
        self.comm_name = {
            cid: (f"C{cid}: {G.nodes[d]['label'][:30]}" if cid >= 0 else "Isolated datasets")
            for cid, d in best.items()}
        self._seq: dict[str, str] = {}

    def _non_papers(self) -> Iterator[str]:
        return (n for n, d in self.G.nodes(data=True) if d["type"] != "Paper")

    def _seq_color(self, key: str) -> str:
        if key not in self._seq:
            self._seq[key] = PALETTE[len(self._seq) % len(PALETTE)]
        return self._seq[key]

    def size(self, node: str) -> float:
        t = self.G.nodes[node]["type"]
        lo, hi = TYPE_STYLE[t][2]
        dc = self.G.nodes[node].get("degree_centrality", 0.0) * 1e5
        return round(log_scale(dc, self.max_dc, lo, hi), 1)

    def dataset_group(self, node: str) -> tuple[str, str]:
        a = self.G.nodes[node]
        if self.color_by == "field":
            fields = out_by_rel(self.G, node, "HAS_THEME")
            if not fields:
                return "No field", GREY
            label = self.G.nodes[max(fields, key=self.G.in_degree)]["label"]
            return label, self._seq_color(label)
        if self.color_by == "origin":
            return a["origin"], self._seq_color(a["origin"])
        cid = a.get("community", -1)
        return self.comm_name.get(cid, f"C{cid}"), community_color(cid)

    # ---- tooltips (plain text: vis-network 9 does not render HTML strings)
    def tooltip(self, node: str) -> str:
        G, a = self.G, self.G.nodes[node]
        t = a["type"]
        if t == "Dataset":
            names = lambda rel, k=4: [G.nodes[x]["label"] for x in out_by_rel(G, node, rel)][:k]
            lines = [f"{a['label']}  [{node}]",
                     f"Type: Dataset   Origin: {a['origin']}",
                     f"Papers referencing it: {a['n_papers']:,}",
                     f"Co-cited with {self.H.degree(node) if node in self.H else 0} other datasets",
                     "Community: " + self.comm_name.get(a.get("community", -1), "?"),
                     f"PageRank: {a.get('pagerank', 0):.5f}   "
                     f"Degree centrality: {a.get('degree_centrality', 0):.6f}"]
            for title, rel, k in (("Fields", "HAS_THEME", 4), ("Creators", "CREATED_BY", 4),
                                  ("Keywords", "HAS_KEYWORD", 6)):
                vals = names(rel, k)
                if vals:
                    lines.append(f"{title}: " + ", ".join(vals))
            if a.get("issued"):
                lines.append(f"Issued: {a['issued']}")
            if a.get("description"):
                lines += ["", wrap_text(a["description"], 70, 320)]
            link = (a.get("landing_pages") or [a.get("identifier")])[0]
            if link:
                lines += ["", str(link)]
            return "\n".join(lines)
        if t == "Paper":
            mag = node.split(":", 1)[1]
            return (f"Paper (Microsoft Academic id {mag})\n"
                    f"Referenced by {G.in_degree(node)} dataset(s) in DSKG\n"
                    f"http://ma-graph.org/entity/{mag}")
        n_ds = sum(1 for _ in G.in_edges(node))
        extra = f"\nOrcid: {a['orcid']}" if a.get("orcid") else ""
        return f"{a['label']}\nType: {t}\nLinked from {n_ds} dataset(s){extra}"


def add_node(scene: Scene, st: Styler, node: str, center: bool = False) -> None:
    """Add one node. NOTE: the legend category is stored as ``kg_group``, not vis.js's
    ``group``: vis.js colours nodes by ``group`` and would ignore our own colours.
    Papers get a blank label (" ") because PyVis shows the node id for empty labels."""
    if node in scene.nodes:
        return
    a = st.G.nodes[node]
    t = a["type"]
    if t == "Dataset":
        group, color = st.dataset_group(node)
        label = a["label"] if len(a["label"]) <= 30 else a["label"][:28] + ".."
        spec: dict[str, Any] = dict(label=label, color=color, shape="dot", kg_group=group,
                                    size=st.size(node))
    elif t == "Paper":
        color, shape, _ = TYPE_STYLE["Paper"]
        group = "Paper"
        spec = dict(label=" ", color=color, shape=shape, kg_group=group, size=5)
    else:
        color, shape, _ = TYPE_STYLE[t]
        group = t
        lab = a["label"] if len(a["label"]) <= 26 else a["label"][:24] + ".."
        spec = dict(label=lab, color=color, shape=shape, kg_group=group, size=st.size(node))
    spec["title"] = st.tooltip(node)
    if center:
        spec.update(size=TYPE_STYLE["Dataset"][2][1] + 10, borderWidth=4,
                    color={"background": spec["color"], "border": "#000000"},
                    font={"size": 20})
    scene.nodes[node] = spec
    scene.groups.setdefault(group, color)


def add_edge(scene: Scene, u: str, v: str, rel: str, weight: int | None = None,
             jaccard: float | None = None) -> None:
    color, dashed = RELATION_STYLE[rel]
    if rel == "CO_CITED":
        w = weight or 1
        scene.edges.append(dict(
            src=u, dst=v, color=color, dashes=False,
            width=round(1 + 1.6 * math.log10(1 + w), 2), label=f"{w:,}",
            title=f"Co-cited: {w:,} shared paper(s), Jaccard {jaccard or 0:.3f}",
            arrows={"to": {"enabled": False}}))
    else:
        text = rel.replace("_", " ").lower()
        scene.edges.append(dict(src=u, dst=v, color=color, dashes=dashed, width=1.2,
                                label=text, title=text))


def induced_edges(G: nx.MultiDiGraph, selected: set[str]) -> Iterator[tuple[str, str, str]]:
    """Edges of ``G`` with both ends inside ``selected`` (only datasets have out-edges)."""
    for u in selected:
        if G.nodes[u]["type"] != "Dataset":
            continue
        for _, v, k in G.out_edges(u, keys=True):
            if v in selected:
                yield u, v, k


def cocited_edges(H: nx.Graph, datasets: list[str], max_edges: int | None = None
                  ) -> list[tuple[str, str, dict]]:
    ds = set(datasets)
    items = [(u, v, d) for u in datasets for v, d in H[u].items() if v in ds and u < v]
    items.sort(key=lambda t: -t[2]["weight"])
    return items[:max_edges] if max_edges else items


def _finish(scene: Scene, G, H, datasets: list[str], selected: set[str],
            max_cocite: int | None = None) -> None:
    for u, v, k in induced_edges(G, selected):
        add_edge(scene, u, v, k)
    for u, v, d in cocited_edges(H, datasets, max_cocite):
        add_edge(scene, u, v, "CO_CITED", d["weight"], d.get("jaccard"))


# --------------------------------------------------------------------------- #
# Scene builders
# --------------------------------------------------------------------------- #
def scene_overview(G: nx.MultiDiGraph, H: nx.Graph, max_edges: int = 150,
                   with_fields: bool = True) -> Scene:
    """One node per community; grey edges = co-citation between communities,
    orange edges = community members tagged with a field of study (fields
    connect communities that share no papers)."""
    members: defaultdict[int, list[str]] = defaultdict(list)
    for d in dataset_nodes(G):
        cid = G.nodes[d].get("community", -1)
        if cid >= 0:
            members[cid].append(d)
    if not members:
        raise ValueError("No communities found - rebuild the graph first.")
    isolated = sum(1 for d in dataset_nodes(G) if G.nodes[d].get("community", -1) < 0)
    sc = Scene("Community overview")
    sc.notes.append(f"{isolated:,} datasets share no papers with any other dataset "
                    f"and are not shown here.")
    biggest = max(len(m) for m in members.values())
    for cid, ms in sorted(members.items()):
        top = sorted(ms, key=lambda d: -G.nodes[d]["n_papers"])
        fields = Counter(G.nodes[f]["label"] for d in ms for f in out_by_rel(G, d, "HAS_THEME"))
        kws = Counter(G.nodes[k]["label"] for d in ms for k in out_by_rel(G, d, "HAS_KEYWORD"))
        total_papers = sum(G.nodes[d]["n_papers"] for d in ms)
        name = f"C{cid}: {G.nodes[top[0]]['label'][:30]}"
        tip = [name, f"{len(ms)} datasets, {total_papers:,} paper links",
               "Top fields: " + ", ".join(f"{k} ({v})" for k, v in fields.most_common(3)),
               "Top keywords: " + ", ".join(k for k, _ in kws.most_common(5)),
               "Top datasets: " + " | ".join(G.nodes[d]["label"][:30] for d in top[:5])]
        color = community_color(cid)
        sc.nodes[f"community:{cid}"] = dict(
            label=f"C{cid} {G.nodes[top[0]]['label'][:20]}\n({len(ms)})",
            title="\n".join(tip), color=color, shape="dot", kg_group=name,
            size=round(14 + 28 * math.sqrt(len(ms) / biggest), 1))
        sc.groups[name] = color
    agg: Counter[tuple[int, int]] = Counter()
    for u, v, d in H.edges(data=True):
        cu, cv = H.nodes[u]["community"], H.nodes[v]["community"]
        if cu >= 0 and cv >= 0 and cu != cv:
            agg[(min(cu, cv), max(cu, cv))] += d["weight"]
    for (a, b), w in agg.most_common(max_edges):
        sc.edges.append(dict(
            src=f"community:{a}", dst=f"community:{b}", color="#9E9E9E", dashes=False,
            width=round(1 + 1.6 * math.log10(1 + w), 2), label="",
            title=f"C{a} <-> C{b}: {w:,} co-citation weight",
            arrows={"to": {"enabled": False}}))
    if with_fields:
        st = Styler(G, H)
        links: Counter[tuple[int, str]] = Counter()
        for cid, ms in members.items():
            for d in ms:
                for f in out_by_rel(G, d, "HAS_THEME"):
                    links[(cid, f)] += 1
        for f in sorted({f for _, f in links}):
            add_node(sc, st, f)
        for (cid, f), c in sorted(links.items()):
            sc.edges.append(dict(
                src=f"community:{cid}", dst=f, color=RELATION_STYLE["HAS_THEME"][0],
                dashes=False, width=round(0.8 + 1.2 * math.log10(1 + c), 2), label="",
                title=f"{c} dataset(s) in C{cid} tagged {G.nodes[f]['label']}",
                arrows={"to": {"enabled": False}}))
        sc.notes.append("Orange diamonds are fields of study; they link communities "
                        "that share a field even when they share no papers.")
    return sc


def scene_map(G: nx.MultiDiGraph, H: nx.Graph, min_links: int = 2, kw_min: int = 3,
              label_top: int = 60, max_cocite: int | None = 30000) -> Scene:
    """The big picture: *every* dataset plus the entities that connect them.

    Included: all datasets, all fields, and every organization / person /
    keyword linked to at least ``min_links`` (``kw_min`` for keywords)
    datasets - entities linked to a single dataset connect nothing.  The 634k
    papers are left out (they would drown the picture); they appear as the
    grey co-citation edges between datasets.
    """
    st = Styler(G, H, "community")
    datasets = dataset_nodes(G)
    sc = Scene("Knowledge graph map: all datasets and their connecting entities")
    keep: list[str] = []
    for n, d in G.nodes(data=True):
        t = d["type"]
        if t == "Field" or (t in ("Person", "Organization") and G.in_degree(n) >= min_links) \
                or (t == "Keyword" and G.in_degree(n) >= kw_min):
            keep.append(n)
    keep.sort()
    selected = set(datasets) | set(keep)
    for n in sorted(datasets) + keep:
        add_node(sc, st, n)
    _finish(sc, G, H, datasets, selected, max_cocite)   # keeps the heaviest links

    famous = set(sorted(datasets, key=lambda d: -G.nodes[d]["n_papers"])[:label_top])
    for n, spec in sc.nodes.items():
        if G.nodes[n]["type"] != "Field" and n not in famous:
            spec["label"] = " "            # names stay available in tooltips/search
    for e in sc.edges:
        if e["title"].startswith("Co-cited"):
            e["width"] = min(e["width"], 2.0)
            e["color"] = "rgba(110,110,110,0.35)"
    sc.notes += [f"{len(datasets):,} datasets, {len(keep):,} connecting entities, "
                 f"{len(sc.edges):,} edges drawn.",
                 "Only the biggest datasets are labelled; hover any node for its name, "
                 "or use the search box. This page is heavy: give it ~30 s to settle."]
    return sc


def scene_dataset(G: nx.MultiDiGraph, H: nx.Graph, node: str, neighbors: int = 12,
                  papers: int = 10, rank: str = "weight") -> Scene:
    """A dataset, its most co-cited datasets, shared papers and metadata."""
    st = Styler(G, H, "community")
    sc = Scene(f"{G.nodes[node]['label']} - co-citation neighborhood")
    nbrs = sorted(H[node].items(), key=lambda kv: -kv[1][rank])[:neighbors] if node in H else []
    nbr_nodes = [n for n, _ in nbrs]
    datasets = [node] + nbr_nodes
    if not nbr_nodes:
        sc.notes.append("This dataset shares no papers with any other dataset.")

    nbr_kw = Counter(k for n in nbr_nodes for k in out_by_rel(G, n, "HAS_KEYWORD"))
    kws = sorted(out_by_rel(G, node, "HAS_KEYWORD"),
                 key=lambda k: (-nbr_kw[k], G.degree(k)))[:8]
    meta = (out_by_rel(G, node, "HAS_THEME") + out_by_rel(G, node, "CREATED_BY")[:5]
            + out_by_rel(G, node, "PUBLISHED_BY")[:3] + kws)

    paper_nodes: list[str] = []
    if papers > 0 and nbr_nodes:
        mine = set(out_by_rel(G, node, "REFERENCED_BY"))
        cnt: Counter[str] = Counter()
        for n in nbr_nodes:
            for p in mine.intersection(out_by_rel(G, n, "REFERENCED_BY")):
                cnt[p] += 1
        paper_nodes = [p for p, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:papers]]

    selected = set(datasets) | set(meta) | set(paper_nodes)
    add_node(sc, st, node, center=True)
    for n in nbr_nodes + meta + paper_nodes:
        add_node(sc, st, n)
    _finish(sc, G, H, datasets, selected)
    return sc


def scene_from_datasets(G, H, datasets: list[str], title: str, color_by: str,
                        include_fields: bool = True, min_field_links: int = 2,
                        force_nodes: tuple[str, ...] = (), max_cocite: int | None = None
                        ) -> Scene:
    st = Styler(G, H, color_by)
    sc = Scene(title)
    selected = set(datasets)
    extra: list[str] = list(force_nodes)
    if include_fields:
        counts = Counter(f for d in datasets for f in out_by_rel(G, d, "HAS_THEME"))
        extra += [f for f, c in counts.items() if c >= min_field_links and f not in extra]
    selected |= set(extra)
    for d in datasets:
        add_node(sc, st, d)
    for n in extra:
        add_node(sc, st, n)
    _finish(sc, G, H, datasets, selected, max_cocite)
    return sc


def scene_community(G, H, cid: int, top: int = 60) -> Scene:
    members = [d for d in dataset_nodes(G) if G.nodes[d].get("community", -1) == cid]
    if not members:
        valid = sorted({G.nodes[d]["community"] for d in dataset_nodes(G)} - {-1})
        raise ValueError(f"No community {cid}. Valid ids: {valid}")
    members.sort(key=lambda d: -G.nodes[d]["pagerank"])
    chosen = members[:top]
    sc = scene_from_datasets(G, H, chosen, f"Community {cid}: top {len(chosen)} of "
                             f"{len(members)} datasets (by PageRank)", "field",
                             max_cocite=int(3.5 * top))
    if len(members) > top:
        sc.notes.append(f"Showing {len(chosen)} of {len(members)} datasets; use --top to show more.")
    return sc


def scene_field(G, H, name: str, top: int = 50) -> Scene:
    matches = find_nodes(G, name, "Field", limit=3)
    if not matches:
        labels = sorted(d["label"] for _, d in G.nodes(data=True) if d["type"] == "Field")
        raise ValueError(f"No field matches '{name}'. Available: {', '.join(labels)}")
    fid = matches[0][0]
    ds = [u for u, _, k in G.in_edges(fid, keys=True) if k == "HAS_THEME"]
    ds.sort(key=lambda d: -G.nodes[d]["n_papers"])
    chosen = ds[:top]
    sc = scene_from_datasets(G, H, chosen, f"Field: {G.nodes[fid]['label']} "
                             f"(top {len(chosen)} of {len(ds)} datasets by papers)",
                             "community", force_nodes=(fid,), max_cocite=int(3.5 * top))
    if len(ds) > top:
        sc.notes.append(f"Showing {len(chosen)} of {len(ds)} datasets; use --top to show more.")
    return sc


def scene_search(G, H, text: str, top: int = 25) -> Scene:
    hits = search_datasets(G, text, limit=top)
    if not hits:
        raise ValueError(f"No datasets match '{text}'.")
    return scene_from_datasets(G, H, [d for d, _ in hits],
                               f"Search: '{text}' ({len(hits)} datasets)", "community",
                               min_field_links=2, max_cocite=int(3.5 * top))


# --------------------------------------------------------------------------- #
# Rendering (PyVis) and in-page controls
# --------------------------------------------------------------------------- #
VIS_OPTIONS = {
    "nodes": {"font": {"size": 14}, "borderWidth": 1},
    "edges": {"smooth": {"type": "continuous"}, "font": {"size": 10, "align": "middle"}},
    "physics": {"solver": "forceAtlas2Based",
                "forceAtlas2Based": {"gravitationalConstant": -70, "springLength": 120,
                                     "avoidOverlap": 0.4},
                "stabilization": {"iterations": 250}},
    "interaction": {"hover": True, "tooltipDelay": 80, "navigationButtons": True,
                    "keyboard": True},
}

BIG_OPTIONS = {
    "nodes": {"font": {"size": 12}, "borderWidth": 1},
    "edges": {"smooth": False},
    "layout": {"improvedLayout": False},
    "physics": {"solver": "barnesHut",
                "barnesHut": {"gravitationalConstant": -4500, "springLength": 110,
                              "springConstant": 0.03, "avoidOverlap": 0.2},
                "stabilization": {"iterations": 150, "updateInterval": 25}},
    "interaction": {"hover": True, "tooltipDelay": 80, "navigationButtons": True,
                    "keyboard": True, "hideEdgesOnDrag": True, "hideEdgesOnZoom": True},
}


def vis_options(scene: Scene) -> dict:
    """Light-weight physics for pages with many nodes."""
    return BIG_OPTIONS if len(scene.nodes) > 600 else VIS_OPTIONS


_PANEL_JS = """
<script>
(function () {
  var box = document.getElementById('kg-search'), status = document.getElementById('kg-status');
  if (!box) return;
  box.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter') return;
    try {
      var q = box.value.trim().toLowerCase(); if (!q) return;
      var m = nodes.get().filter(function (n) {
        return !n.hidden && ((n.label || '').toLowerCase().indexOf(q) >= 0 ||
          String(n.id).toLowerCase() === q || (n.title || '').toLowerCase().indexOf(q) >= 0);
      });
      if (!m.length) { status.textContent = 'No match'; return; }
      var ids = m.map(function (n) { return n.id; });
      network.selectNodes(ids);
      network.focus(ids[0], {scale: 1.3, animation: {duration: 600}});
      status.textContent = m.length + ' match' + (m.length > 1 ? 'es' : '') + ' (focused on first)';
    } catch (err) { status.textContent = 'Search unavailable'; }
  });
  document.querySelectorAll('#kg-panel input[data-group]').forEach(function (cb) {
    cb.addEventListener('change', function () {
      try {
        var g = cb.getAttribute('data-group');
        nodes.update(nodes.get().filter(function (n) { return n.kg_group === g; })
          .map(function (n) { return {id: n.id, hidden: !cb.checked}; }));
      } catch (err) {}
    });
  });
})();
</script>
"""


def inject_controls(page: str, scene: Scene) -> str:
    """Add a search box and per-group show/hide checkboxes to the PyVis page."""
    rows = "".join(
        f'<label style="display:block;white-space:nowrap">'
        f'<input type="checkbox" checked data-group="{html.escape(g, quote=True)}"> '
        f'<span style="display:inline-block;width:10px;height:10px;background:{c};'
        f'border-radius:50%"></span> {html.escape(g)}</label>'
        for g, c in scene.groups.items())
    notes = "".join(f'<div style="color:#666;margin-top:4px">{html.escape(n)}</div>'
                    for n in scene.notes)
    panel = (
        '<div id="kg-panel" style="position:fixed;top:10px;right:10px;z-index:9999;'
        'background:rgba(255,255,255,.95);border:1px solid #ccc;border-radius:6px;'
        'padding:8px 10px;font:12px/1.5 Arial,sans-serif;width:250px;max-height:85vh;'
        'overflow:auto;box-shadow:0 1px 4px rgba(0,0,0,.25)">'
        f'<div style="font-weight:bold;margin-bottom:4px">{html.escape(scene.title)}</div>'
        '<input id="kg-search" placeholder="Search nodes, press Enter" '
        'style="width:100%;box-sizing:border-box;padding:3px">'
        '<div id="kg-status" style="color:#666;min-height:14px"></div>'
        '<div style="font-weight:bold;margin-top:6px">Show / hide</div>'
        f'{rows}{notes}</div>')
    block = panel + _PANEL_JS
    return page.replace("</body>", block + "</body>", 1) if "</body>" in page else page + block


def _build_net(scene: Scene, cdn: str):
    from pyvis.network import Network  # imported lazily: only needed to render
    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222222",
                  directed=True, select_menu=True, cdn_resources=cdn)
    for nid, spec in scene.nodes.items():
        net.add_node(nid, **spec)
    show_labels = len(scene.edges) <= 60
    for e in scene.edges:
        opts = {k: v for k, v in e.items() if k not in ("src", "dst")}
        if not show_labels:
            opts.pop("label", None)
        net.add_edge(e["src"], e["dst"], **opts)
    net.set_options("var options = " + json.dumps(vis_options(scene)))
    return net


def render(scene: Scene, path: Path, offline: bool = True) -> Path:
    """Write ``scene`` to a self-contained HTML file at ``path``."""
    try:
        import pyvis  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyvis is not installed. Run: pip install -r requirements.txt") from exc
    try:
        net = _build_net(scene, "in_line" if offline else "remote")
        page = net.generate_html()
    except UnicodeDecodeError:
        print("WARNING: could not inline the JavaScript libraries (Windows encoding). "
              "Falling back to online libraries - the page needs internet to open. "
              "To fix: run  $env:PYTHONUTF8=1  in PowerShell first.", file=sys.stderr)
        page = _build_net(scene, "remote").generate_html()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inject_controls(page, scene), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "view"


def build_parser() -> argparse.ArgumentParser:
    """CLI parser. The global options work before *or* after the subcommand."""
    def add_globals(parser: argparse.ArgumentParser, defaults: bool) -> None:
        # In sub-parsers the defaults are suppressed so they cannot overwrite
        # values given before the subcommand.
        kw = (lambda d: {"default": d}) if defaults else (lambda d: {})
        parser.add_argument("--graph", type=Path, **kw(Path("data/processed/kg.pkl")))
        parser.add_argument("--out", type=Path, **kw(Path("output")))
        parser.add_argument("--open", action="store_true",
                            **({} if defaults else {"default": argparse.SUPPRESS}),
                            help="open the page in the browser")
        parser.add_argument("--cdn", action="store_true",
                            **({} if defaults else {"default": argparse.SUPPRESS}),
                            help="load JS libraries online instead of embedding them")

    ap = argparse.ArgumentParser(prog="python -m src.visualizer",
                                 description="Generate interactive HTML views of the KG.")
    add_globals(ap, defaults=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def new(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, argument_default=argparse.SUPPRESS)
        add_globals(sp, defaults=False)
        return sp

    p = new("overview", "community-level map of the whole graph")
    p.add_argument("--no-fields", action="store_true", default=False,
                   help="hide the field-of-study nodes")
    new("map", "every dataset plus the entities connecting them (large page)")
    p = new("dataset", "neighborhood of one dataset")
    p.add_argument("query", help="title (fuzzy), number, or dataset:ID")
    p.add_argument("--neighbors", type=int, default=12)
    p.add_argument("--papers", type=int, default=10, help="shared papers to draw")
    p.add_argument("--rank", choices=["weight", "jaccard"], default="weight")
    p = new("community", "top datasets of one community")
    p.add_argument("id", type=int)
    p.add_argument("--top", type=int, default=60)
    p = new("field", "top datasets of one field of study")
    p.add_argument("name")
    p.add_argument("--top", type=int, default=50)
    p = new("search", "datasets matching a keyword query")
    p.add_argument("text")
    p.add_argument("--top", type=int, default=25)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    try:
        G, H = load_graph(args.graph)
        if args.cmd == "overview":
            scene = scene_overview(G, H, with_fields=not args.no_fields)
            name = "overview"
        elif args.cmd == "map":
            scene, name = scene_map(G, H), "map"
        elif args.cmd == "dataset":
            hits = find_nodes(G, args.query, "Dataset", limit=5)
            if not hits:
                raise ValueError(f"No dataset matches '{args.query}'.")
            if len(hits) > 1:
                print("Closest matches (using the first; pass dataset:ID to pick another):")
                for n, s in hits:
                    print(f"  {n:<16} {G.nodes[n]['label'][:50]:<50} "
                          f"{G.nodes[n]['n_papers']:>8,} papers  (score {s:.0f})")
            node = hits[0][0]
            scene = scene_dataset(G, H, node, args.neighbors, args.papers, args.rank)
            name = f"dataset-{_slug(G.nodes[node]['label'])}-{node.split(':')[1]}"
        elif args.cmd == "community":
            scene, name = scene_community(G, H, args.id, args.top), f"community-{args.id}"
        elif args.cmd == "field":
            scene, name = scene_field(G, H, args.name, args.top), f"field-{_slug(args.name)}"
        else:
            scene, name = scene_search(G, H, args.text, args.top), f"search-{_slug(args.text)}"
        out = render(scene, args.out / f"{name}.html", offline=not args.cdn)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for note in scene.notes:
        print("Note:", note)
    print(f"Wrote {out}  ({len(scene.nodes)} nodes, {len(scene.edges)} edges)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
