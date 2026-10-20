"""Pillar 3 - Functional downstream query, recommendation, pathfinding & QA engine.

Operates directly on ``data/processed/kg.pkl`` (built by ``src.graph_builder``)
and reuses lookup/rendering helpers from ``src.common`` and ``src.visualizer``.

Subcommands:
    profile     Full semantic profile + removal impact analysis for a dataset
    recommend   Hybrid dataset recommender (co-citation + Adamic-Adar + metadata)
    path        Shortest co-citation & semantic paths between two datasets (+ --html)
    bridges     Top interdisciplinary bridge datasets (betweenness & cross-community)
    compare     Side-by-side structural & semantic comparison of two datasets
    ask         Natural-language Graph-RAG question answering (100% local)
    repl        Interactive live shell (loads kg.pkl once for instant demo queries)

Usage (from the project root):
    python -m src.graph_engine profile "ImageNet"
    python -m src.graph_engine recommend "ImageNet" --top 10
    python -m src.graph_engine path "ImageNet" "GenBank" --html --open
    python -m src.graph_engine bridges --field "Computer science"
    python -m src.graph_engine compare "ImageNet" "CIFAR-10"
    python -m src.graph_engine ask "How are ImageNet and Gene Ontology connected?"
    python -m src.graph_engine repl
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import shlex
import sys
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from src.common import find_nodes, search_datasets, wrap_text
from src.graph_builder import dataset_nodes, load_graph, normalize_text, out_by_rel
from src.visualizer import Scene, Styler, _slug, add_edge, add_node, render

log = logging.getLogger("graph_engine")

METADATA_RELS = ("HAS_THEME", "HAS_KEYWORD", "CREATED_BY", "PUBLISHED_BY", "CONTRIBUTED_BY")


# --------------------------------------------------------------------------- #
# Data containers for structured engine outputs
# --------------------------------------------------------------------------- #
@dataclass
class Recommendation:
    node: str
    title: str
    total_score: float
    cocite_weight: int
    jaccard: float
    adamic_adar: float
    shared_fields: list[str] = field(default_factory=list)
    shared_keywords: list[str] = field(default_factory=list)
    shared_agents: list[str] = field(default_factory=list)
    n_papers: int = 0
    community: int = -1


@dataclass
class PathResult:
    source: str
    target: str
    cocite_path: list[str] = field(default_factory=list)
    cocite_hops: list[dict[str, Any]] = field(default_factory=list)
    semantic_path: list[str] = field(default_factory=list)
    semantic_edges: list[tuple[str, str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def resolve_dataset(G: nx.MultiDiGraph, query: str, verbose: bool = True) -> str:
    """Resolve a free-text query, integer ID, or ``dataset:ID`` to a Dataset node."""
    hits = find_nodes(G, query, "Dataset", limit=5)
    if not hits:
        kw_hits = search_datasets(G, query, limit=5)
        if not kw_hits:
            raise ValueError(f"No dataset matches '{query}'.")
        hits = kw_hits
    if verbose and len(hits) > 1 and hits[0][1] < 100.0:
        print(f"Resolved '{query}' -> {G.nodes[hits[0][0]]['label']} [{hits[0][0]}] "
              f"(other matches: {', '.join(f'{G.nodes[n]['label'][:30]} [{n}]' for n, _ in hits[1:4])})")
    return hits[0][0]


def _meta_sets(G: nx.MultiDiGraph, ds: str) -> tuple[set[str], set[str], set[str]]:
    """Return ``(fields, keywords, agents)`` neighbor node IDs for dataset ``ds``."""
    fields = set(out_by_rel(G, ds, "HAS_THEME"))
    kws = set(out_by_rel(G, ds, "HAS_KEYWORD"))
    agents = (set(out_by_rel(G, ds, "CREATED_BY"))
              | set(out_by_rel(G, ds, "PUBLISHED_BY"))
              | set(out_by_rel(G, ds, "CONTRIBUTED_BY")))
    return fields, kws, agents


def build_semantic_skeleton(G: nx.MultiDiGraph) -> nx.Graph:
    """Build an undirected bipartite/multipartite graph of all non-Paper nodes.

    Excluding the 634k Paper nodes allows semantic pathfinding across Datasets,
    Fields, Keywords, Persons and Organizations in a few milliseconds.
    """
    S = nx.Graph()
    for n, d in G.nodes(data=True):
        if d["type"] != "Paper":
            S.add_node(n, **d)
    for u, v, k in G.edges(keys=True):
        if k == "REFERENCED_BY":
            continue
        # Weight metadata hubs by log(degree) so specific links (creators/keywords)
        # are preferred over broad Field hubs when finding semantic paths.
        hub_deg = max(G.in_degree(v), 1)
        type_penalty = 2.5 if G.nodes[v]["type"] == "Field" else 1.0
        w = type_penalty * (1.0 + 0.35 * math.log1p(hub_deg))
        if not S.has_edge(u, v) or w < S[u][v]["weight"]:
            S.add_edge(u, v, relation=k, weight=w)
    return S


# --------------------------------------------------------------------------- #
# 1. Dataset Profile & Removal Impact Analysis
# --------------------------------------------------------------------------- #
def analyze_dataset_profile(G: nx.MultiDiGraph, H: nx.Graph, node: str) -> dict[str, Any]:
    """Gather comprehensive metadata, centrality, and removal impact metrics."""
    a = G.nodes[node]
    fields = [G.nodes[f]["label"] for f in out_by_rel(G, node, "HAS_THEME")]
    creators = [G.nodes[p]["label"] for p in out_by_rel(G, node, "CREATED_BY")]
    publishers = [G.nodes[o]["label"] for o in out_by_rel(G, node, "PUBLISHED_BY")]
    contributors = [G.nodes[c]["label"] for c in out_by_rel(G, node, "CONTRIBUTED_BY")]
    keywords = [G.nodes[k]["label"] for k in out_by_rel(G, node, "HAS_KEYWORD")]

    nbrs = sorted(H[node].items(), key=lambda kv: -kv[1]["weight"]) if node in H else []
    top_cocited = [
        {"node": v, "label": G.nodes[v]["label"], "weight": d["weight"],
         "jaccard": d.get("jaccard", 0.0), "community": G.nodes[v].get("community", -1)}
        for v, d in nbrs[:10]
    ]

    # Removal Impact Analysis (Tutorial 354: downstream impact of removing a node)
    # Check how many co-citation neighbors become completely isolated in H if `node` is removed,
    # and how many distinct communities `node` directly bridges.
    isolated_if_removed = [v for v, _ in nbrs if H.degree(v) == 1]
    bridged_communities = sorted({G.nodes[v].get("community", -1) for v, _ in nbrs} - {-1})
    Exclusive_papers = sum(1 for p in out_by_rel(G, node, "REFERENCED_BY") if G.in_degree(p) == 1)

    return {
        "node": node,
        "label": a["label"],
        "origin": a.get("origin", "Unknown"),
        "identifier": a.get("identifier"),
        "issued": a.get("issued"),
        "language": a.get("language"),
        "access_rights": a.get("access_rights"),
        "formats": a.get("formats", []),
        "licences": a.get("licences", []),
        "landing_pages": a.get("landing_pages", []),
        "description": a.get("description", ""),
        "n_papers": a.get("n_papers", 0),
        "exclusive_papers": Exclusive_papers,
        "shared_papers": a.get("n_papers", 0) - Exclusive_papers,
        "community": a.get("community", -1),
        "pagerank": a.get("pagerank", 0.0),
        "degree_centrality": a.get("degree_centrality", 0.0),
        "weighted_degree": a.get("weighted_degree", 0),
        "co_cited_count": len(nbrs),
        "fields": fields,
        "creators": creators,
        "publishers": publishers,
        "contributors": contributors,
        "keywords": keywords,
        "top_cocited": top_cocited,
        "impact_isolated_datasets": [f"{G.nodes[v]['label']} [{v}]" for v in isolated_if_removed],
        "impact_bridged_communities": bridged_communities,
    }


def format_profile(prof: dict[str, Any]) -> str:
    lines = [
        "=" * 74,
        f"DATASET PROFILE: {prof['label']}  [{prof['node']}]",
        "=" * 74,
        f"Origin        : {prof['origin']}    Issued: {prof['issued'] or 'N/A'}    "
        f"Access: {prof['access_rights'] or 'N/A'}",
    ]
    if prof["identifier"]:
        lines.append(f"Identifier    : {prof['identifier']}")
    if prof["landing_pages"]:
        lines.append(f"Landing Page  : {prof['landing_pages'][0]}")
    if prof["fields"]:
        lines.append(f"Fields        : {', '.join(prof['fields'])}")
    if prof["creators"]:
        lines.append(f"Creators      : {', '.join(prof['creators'][:6])}"
                     + (f" (+{len(prof['creators']) - 6} more)" if len(prof["creators"]) > 6 else ""))
    if prof["publishers"]:
        lines.append(f"Publishers    : {', '.join(prof['publishers'][:4])}")
    if prof["keywords"]:
        lines.append(f"Keywords      : {', '.join(prof['keywords'][:10])}"
                     + (f" (+{len(prof['keywords']) - 10} more)" if len(prof["keywords"]) > 10 else ""))
    if prof["formats"] or prof["licences"]:
        lines.append(f"Distribution  : Formats={prof['formats'] or ['N/A']}  "
                     f"Licences={prof['licences'] or ['N/A']}")

    lines += [
        "",
        "--- Graph Centrality & Topology Metrics ---",
        f"  Referencing Papers : {prof['n_papers']:,} "
        f"({prof['exclusive_papers']:,} exclusive, {prof['shared_papers']:,} co-citing other datasets)",
        f"  Co-cited Datasets  : {prof['co_cited_count']:,} datasets "
        f"(total co-citation weight: {prof['weighted_degree']:,})",
        f"  Louvain Community  : {prof['community']}    "
        f"PageRank: {prof['pagerank']:.6f}    Degree Centrality: {prof['degree_centrality']:.6f}",
        "",
        "--- Removal Impact Analysis (Subgraph Criticality) ---",
        f"  Co-citation edges lost if removed       : {prof['co_cited_count']:,} edges "
        f"({prof['weighted_degree']:,} paper-dataset bridge links)",
        f"  Communities bridged by this dataset     : {len(prof['impact_bridged_communities'])} "
        f"({prof['impact_bridged_communities']})",
        f"  Datasets disconnected from H if removed : {len(prof['impact_isolated_datasets'])}"
        + (f" -> {', '.join(prof['impact_isolated_datasets'][:5])}"
           if prof["impact_isolated_datasets"] else " (all neighbors have backup paths)"),
    ]
    if prof["top_cocited"]:
        lines += ["", "--- Top Co-Cited Datasets (Direct H Neighbors) ---"]
        for i, c in enumerate(prof["top_cocited"], 1):
            lines.append(f"  {i:>2}. {c['label'][:42]:<42} [{c['node']:<12}] "
                         f"shared={c['weight']:>6,}  jaccard={c['jaccard']:.4f}  (C{c['community']})")
    if prof["description"]:
        lines += ["", "--- Description ---", wrap_text(prof["description"], 74, 500)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Multi-Signal Dataset Recommendation Engine
# --------------------------------------------------------------------------- #
def recommend_datasets(G: nx.MultiDiGraph, H: nx.Graph, source: str, top: int = 10,
                       novel_only: bool = False) -> list[Recommendation]:
    """Rank candidate datasets by combining co-citation, Adamic-Adar link
    prediction, and metadata overlap (fields, keywords, creators/publishers).

    If ``novel_only=True``, only recommends datasets that have ZERO direct
    co-citations with ``source`` (pure link prediction for undiscovered pairs).
    """
    src_fields, src_kws, src_agents = _meta_sets(G, source)
    direct_nbrs = set(H[source]) if source in H else set()

    # Adamic-Adar index on H for 1-hop and 2-hop candidates
    aa_scores: Counter[str] = Counter()
    if source in H:
        for nbr in direct_nbrs:
            deg = H.degree(nbr)
            contrib = 1.0 / math.log(deg + 1.1)
            for cand in H[nbr]:
                if cand != source:
                    aa_scores[cand] += contrib

    max_cocite = max((d["weight"] for d in H[source].values()), default=1) if source in H else 1
    max_aa = max(aa_scores.values(), default=1.0)

    results: list[Recommendation] = []
    for cand in dataset_nodes(G):
        if cand == source:
            continue
        is_direct = cand in direct_nbrs
        if novel_only and is_direct:
            continue

        edge_data = H[source][cand] if is_direct else {}
        w = int(edge_data.get("weight", 0))
        jac = float(edge_data.get("jaccard", 0.0))
        aa = float(aa_scores.get(cand, 0.0))

        c_fields, c_kws, c_agents = _meta_sets(G, cand)
        sh_f = [G.nodes[x]["label"] for x in (src_fields & c_fields)]
        sh_k = [G.nodes[x]["label"] for x in (src_kws & c_kws)]
        sh_a = [G.nodes[x]["label"] for x in (src_agents & c_agents)]

        if w == 0 and aa == 0 and not sh_f and not sh_k and not sh_a:
            continue

        # Normalized multi-signal score in [0, 100]
        s_cocite = 40.0 * (math.log1p(w) / math.log1p(max_cocite)) if w > 0 else 0.0
        s_jaccard = 20.0 * min(jac * 10.0, 1.0)
        s_aa = 20.0 * (aa / max_aa) if max_aa > 0 else 0.0
        s_meta = min(20.0, 5.0 * len(sh_f) + 3.5 * len(sh_k) + 8.0 * len(sh_a))
        total = round(s_cocite + s_jaccard + s_aa + s_meta, 2)
        if total <= 0:
            continue

        results.append(Recommendation(
            node=cand,
            title=G.nodes[cand]["label"],
            total_score=total,
            cocite_weight=w,
            jaccard=jac,
            adamic_adar=round(aa, 3),
            shared_fields=sorted(sh_f),
            shared_keywords=sorted(sh_k),
            shared_agents=sorted(sh_a),
            n_papers=G.nodes[cand].get("n_papers", 0),
            community=G.nodes[cand].get("community", -1),
        ))

    results.sort(key=lambda r: (-r.total_score, -r.cocite_weight, -r.n_papers))
    return results[:top]


def format_recommendations(G: nx.MultiDiGraph, source: str, recs: list[Recommendation],
                           novel_only: bool = False) -> str:
    mode = "NOVEL LINK PREDICTION (unconnected in H)" if novel_only else "HYBRID RECOMMENDATIONS"
    lines = [
        "=" * 78,
        f"{mode} FOR: {G.nodes[source]['label']} [{source}]",
        "=" * 78,
        f"{'#':>2}  {'Dataset Title':<36} {'ID':<13} {'Score':>6} {'CoCite':>7} "
        f"{'Jaccard':>7} {'AA-Pred':>7}  Shared Metadata",
        "-" * 78,
    ]
    if not recs:
        lines.append("  No matching recommendations found.")
        return "\n".join(lines)
    for i, r in enumerate(recs, 1):
        reasons = []
        if r.shared_fields:
            reasons.append("Field:" + ",".join(r.shared_fields[:2]))
        if r.shared_keywords:
            reasons.append("KW:" + ",".join(r.shared_keywords[:2]))
        if r.shared_agents:
            reasons.append("Creator:" + ",".join(r.shared_agents[:1]))
        reason_str = " | ".join(reasons) if reasons else f"C{r.community} co-citation cluster"
        lines.append(
            f"{i:>2}. {r.title[:35]:<36} {r.node:<13} {r.total_score:>6.1f} "
            f"{r.cocite_weight:>7,} {r.jaccard:>7.4f} {r.adamic_adar:>7.2f}  {reason_str}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Cross-Domain Path-Finder & Scene Builder
# --------------------------------------------------------------------------- #
def find_cross_domain_paths(G: nx.MultiDiGraph, H: nx.Graph, source: str,
                            target: str) -> PathResult:
    """Find both the strongest co-citation path in ``H`` and the most specific
    metadata path in ``G`` between ``source`` and ``target``."""
    res = PathResult(source=source, target=target)

    # 1. Co-citation path in H (edge cost = 1 / (1 + log1p(shared_papers)))
    if source in H and target in H:
        def _cocite_cost(u: str, v: str, d: dict) -> float:
            return 1.0 / (1.0 + math.log1p(d.get("weight", 1)))

        try:
            cpath = nx.shortest_path(H, source=source, target=target, weight=_cocite_cost)
            res.cocite_path = cpath
            for a, b in zip(cpath, cpath[1:]):
                d = H[a][b]
                res.cocite_hops.append({
                    "u": a, "v": b,
                    "u_label": G.nodes[a]["label"], "v_label": G.nodes[b]["label"],
                    "weight": d["weight"], "jaccard": d.get("jaccard", 0.0),
                })
        except nx.NetworkXNoPath:
            pass

    # 2. Heterogeneous semantic path in G (via Fields, Keywords, Persons, Orgs)
    S = build_semantic_skeleton(G)
    if source in S and target in S:
        try:
            spath = nx.shortest_path(S, source=source, target=target, weight="weight")
            res.semantic_path = spath
            for a, b in zip(spath, spath[1:]):
                rel = S[a][b].get("relation", "RELATED")
                res.semantic_edges.append((a, b, rel))
        except nx.NetworkXNoPath:
            pass

    return res


def format_path_result(G: nx.MultiDiGraph, res: PathResult) -> str:
    s_lab = G.nodes[res.source]["label"]
    t_lab = G.nodes[res.target]["label"]
    lines = [
        "=" * 74,
        f"CROSS-DOMAIN PATH ANALYSIS: {s_lab} [{res.source}]  -->  {t_lab} [{res.target}]",
        "=" * 74,
        "",
        "1) Co-Citation Path in H (via shared referencing papers):",
    ]
    if res.cocite_path:
        lines.append(f"   Hops: {len(res.cocite_hops)}   "
                     f"Path: {'  ==>  '.join(G.nodes[n]['label'] for n in res.cocite_path)}")
        for i, h in enumerate(res.cocite_hops, 1):
            lines.append(f"     Hop {i}: {h['u_label']} [{h['u']}]  <-->  {h['v_label']} [{h['v']}]  "
                         f"({h['weight']:,} shared papers, Jaccard={h['jaccard']:.4f})")
    else:
        lines.append("   No co-citation path exists in H (datasets are in disconnected components).")

    lines += ["", "2) Semantic Metadata Path in G (via Fields, Keywords, Creators, Orgs):"]
    if res.semantic_path:
        steps = []
        for n in res.semantic_path:
            d = G.nodes[n]
            steps.append(f"{d['label']} ({d['type']})")
        lines.append(f"   Hops: {len(res.semantic_edges)}   Path: {'  -->  '.join(steps)}")
        for i, (u, v, rel) in enumerate(res.semantic_edges, 1):
            du, dv = G.nodes[u], G.nodes[v]
            lines.append(f"     Step {i}: {du['label']} ({du['type']}) "
                         f"--[{rel}]--> {dv['label']} ({dv['type']})")
    else:
        lines.append("   No semantic metadata path exists in G.")
    return "\n".join(lines)


def scene_path(G: nx.MultiDiGraph, H: nx.Graph, res: PathResult) -> Scene:
    """Build an interactive PyVis Scene highlighting both discovered paths."""
    st = Styler(G, H, "community")
    s_lab = G.nodes[res.source]["label"][:28]
    t_lab = G.nodes[res.target]["label"][:28]
    sc = Scene(f"Path: {s_lab} <-> {t_lab}")

    path_nodes = set(res.cocite_path) | set(res.semantic_path) | {res.source, res.target}
    for n in path_nodes:
        add_node(sc, st, n, center=(n in (res.source, res.target)))

    # Add context metadata around each dataset on the path
    for n in list(path_nodes):
        if G.nodes[n]["type"] == "Dataset":
            for f in out_by_rel(G, n, "HAS_THEME")[:2]:
                add_node(sc, st, f)
                add_edge(sc, n, f, "HAS_THEME")
            for c in out_by_rel(G, n, "CREATED_BY")[:2]:
                add_node(sc, st, c)
                add_edge(sc, n, c, "CREATED_BY")

    seen_edges: set[tuple[str, str, str]] = set()
    for h in res.cocite_hops:
        u, v = h["u"], h["v"]
        key = (min(u, v), max(u, v), "CO_CITED")
        if key not in seen_edges:
            add_edge(sc, u, v, "CO_CITED", h["weight"], h["jaccard"])
            seen_edges.add(key)

    for u, v, rel in res.semantic_edges:
        # Ensure edge direction matches Dataset -> Metadata in G
        src, dst = (u, v) if G.nodes[u]["type"] == "Dataset" else (v, u)
        key = (src, dst, rel)
        if key not in seen_edges:
            add_edge(sc, src, dst, rel)
            seen_edges.add(key)

    sc.notes.append("Highlighted endpoints have bold borders. Grey lines = shared papers; "
                    "colored lines = semantic metadata links.")
    return sc


# --------------------------------------------------------------------------- #
# 4. Interdisciplinary Bridge Detector
# --------------------------------------------------------------------------- #
def find_bridge_datasets(G: nx.MultiDiGraph, H: nx.Graph, field_query: str | None = None,
                         top: int = 15) -> list[dict[str, Any]]:
    """Identify datasets that act as bridges across communities and fields."""
    # Compute exact or sampled betweenness centrality on H if not already stored
    sample_node = next(iter(H), None)
    if sample_node and G.nodes[sample_node].get("betweenness") is not None:
        bet = {n: (G.nodes[n].get("betweenness") or 0.0) for n in H}
    else:
        k = min(350, H.number_of_nodes())
        bet = nx.betweenness_centrality(H, k=k, weight=None, seed=42)

    allowed_ds: set[str] | None = None
    if field_query:
        f_hits = find_nodes(G, field_query, "Field", limit=1)
        if not f_hits:
            raise ValueError(f"No field matches '{field_query}'.")
        fid = f_hits[0][0]
        allowed_ds = {u for u, _, k in G.in_edges(fid, keys=True) if k == "HAS_THEME"}

    rows: list[dict[str, Any]] = []
    for d in H:
        if allowed_ds is not None and d not in allowed_ds:
            continue
        deg = H.degree(d)
        if deg < 2:
            continue
        my_comm = G.nodes[d].get("community", -1)
        nbr_comms = {G.nodes[v].get("community", -1) for v in H[d]} - {-1}
        cross_edges = sum(1 for v in H[d] if G.nodes[v].get("community", -1) != my_comm)
        cross_ratio = cross_edges / deg
        fields = [G.nodes[f]["label"] for f in out_by_rel(G, d, "HAS_THEME")]

        bridge_score = 100.0 * bet.get(d, 0.0) + 0.15 * len(nbr_comms) + 0.5 * cross_ratio
        rows.append({
            "node": d,
            "label": G.nodes[d]["label"],
            "betweenness": bet.get(d, 0.0),
            "cross_ratio": cross_ratio,
            "n_communities_linked": len(nbr_comms),
            "degree_H": deg,
            "n_papers": G.nodes[d].get("n_papers", 0),
            "community": my_comm,
            "fields": "; ".join(fields) if fields else "Untagged",
            "bridge_score": bridge_score,
        })

    rows.sort(key=lambda r: (-r["betweenness"], -r["n_communities_linked"], -r["degree_H"]))
    return rows[:top]


def format_bridges(rows: list[dict[str, Any]], field_query: str | None = None) -> str:
    hdr = f" (Filtered by Field: {field_query})" if field_query else ""
    lines = [
        "=" * 86,
        f"TOP INTERDISCIPLINARY BRIDGE DATASETS IN DSKG{hdr}",
        "=" * 86,
        f"{'#':>2}  {'Dataset Title':<34} {'ID':<13} {'Betweenness':>11} "
        f"{'Comms':>5} {'Cross%':>7} {'H-Deg':>6}  Fields",
        "-" * 86,
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"{i:>2}. {r['label'][:33]:<34} {r['node']:<13} {r['betweenness']:>11.5f} "
            f"{r['n_communities_linked']:>5} {r['cross_ratio'] * 100:>6.1f}% "
            f"{r['degree_H']:>6}  {r['fields'][:24]}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 5. Side-by-Side Dataset Comparison
# --------------------------------------------------------------------------- #
def compare_datasets(G: nx.MultiDiGraph, H: nx.Graph, ds_a: str, ds_b: str) -> str:
    a, b = G.nodes[ds_a], G.nodes[ds_b]
    edge = H[ds_a][ds_b] if (ds_a in H and ds_b in H and H.has_edge(ds_a, ds_b)) else {}
    shared_papers = edge.get("weight", 0)
    jaccard = edge.get("jaccard", 0.0)

    nbrs_a = set(H[ds_a]) if ds_a in H else set()
    nbrs_b = set(H[ds_b]) if ds_b in H else set()
    common_cocited = sorted(nbrs_a & nbrs_b, key=lambda n: -(H[ds_a][n]["weight"] + H[ds_b][n]["weight"]))

    fa, ka, aa = _meta_sets(G, ds_a)
    fb, kb, ab = _meta_sets(G, ds_b)

    lines = [
        "=" * 78,
        f"DATASET COMPARISON: {a['label']} [{ds_a}]  vs.  {b['label']} [{ds_b}]",
        "=" * 78,
        f"{'Metric':<26} | {a['label'][:23]:<23} | {b['label'][:23]:<23}",
        "-" * 78,
        f"{'Node ID':<26} | {ds_a:<23} | {ds_b:<23}",
        f"{'Origin':<26} | {a.get('origin', '?'):<23} | {b.get('origin', '?'):<23}",
        f"{'Referencing Papers':<26} | {a.get('n_papers', 0):>23,} | {b.get('n_papers', 0):>23,}",
        f"{'Co-Cited Datasets (H-Deg)':<26} | {len(nbrs_a):>23,} | {len(nbrs_b):>23,}",
        f"{'Weighted Co-Citation Deg':<26} | {a.get('weighted_degree', 0):>23,} | {b.get('weighted_degree', 0):>23,}",
        f"{'PageRank':<26} | {a.get('pagerank', 0.0):>23.6f} | {b.get('pagerank', 0.0):>23.6f}",
        f"{'Louvain Community':<26} | {str(a.get('community', -1)):>23} | {str(b.get('community', -1)):>23}",
        "-" * 78,
        f"Direct Shared Papers       : {shared_papers:,}  (Co-citation Jaccard = {jaccard:.4f})",
        f"Common Co-Cited Datasets   : {len(common_cocited):,} "
        + (f"(top: {', '.join(G.nodes[n]['label'][:25] for n in common_cocited[:5])})" if common_cocited else ""),
        f"Shared Fields of Study     : {', '.join(G.nodes[x]['label'] for x in (fa & fb)) or 'None'}",
        f"Shared Keywords            : {', '.join(G.nodes[x]['label'] for x in (ka & kb)) or 'None'}",
        f"Shared Creators/Publishers : {', '.join(G.nodes[x]['label'] for x in (aa & ab)) or 'None'}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. Local Graph-RAG / Natural-Language Question Answering Engine
# --------------------------------------------------------------------------- #
def _extract_quoted_or_known_datasets(G: nx.MultiDiGraph, question: str) -> list[str]:
    """Identify Dataset nodes mentioned in a natural-language question."""
    found: list[str] = []
    # 1. Explicit dataset:ID mentions
    for m in re.finditer(r"\bdataset:\d+\b", question.lower()):
        if m.group(0) in G:
            found.append(m.group(0))
    # 2. Quoted strings ("ImageNet", 'GenBank')
    for m in re.finditer(r'["\']([^"\']{2,60})["\']', question):
        hits = find_nodes(G, m.group(1), "Dataset", limit=1)
        if hits and hits[0][0] not in found:
            found.append(hits[0][0])
    if len(found) >= 2:
        return found

    # 3. Scan capitalized / multi-word phrases or known major dataset titles
    q_norm = f" {normalize_text(question)} "
    candidates: list[tuple[int, int, str]] = []
    for d in dataset_nodes(G):
        lab_norm = normalize_text(G.nodes[d]["label"])
        if len(lab_norm) < 4 or lab_norm in {"data", "dataset", "small", "large", "scope"}:
            continue
        if f" {lab_norm} " in q_norm:
            candidates.append((len(lab_norm), G.nodes[d].get("n_papers", 0), d))
    candidates.sort(reverse=True)
    for _, _, d in candidates:
        if d not in found:
            found.append(d)
    return found[:3]


def answer_question(G: nx.MultiDiGraph, H: nx.Graph, question: str) -> str:
    """Route a natural-language question to the appropriate graph traversal
    routine and synthesize a grounded, fact-checked answer."""
    q_low = question.lower().strip()
    mentioned_ds = _extract_quoted_or_known_datasets(G, question)

    header = [
        "=" * 78,
        f"GRAPH-RAG QUERY: {question}",
        "=" * 78,
    ]

    # Intent A: Path / Connection between two datasets
    if len(mentioned_ds) >= 2 and any(w in q_low for w in ("connect", "path", "link", "between", "relate", "bridge")):
        d1, d2 = mentioned_ds[0], mentioned_ds[1]
        res = find_cross_domain_paths(G, H, d1, d2)
        return "\n".join(header + [format_path_result(G, res)])

    # Intent B: Comparison between two datasets
    if len(mentioned_ds) >= 2 and any(w in q_low for w in ("compare", "vs", "versus", "difference", "similar")):
        d1, d2 = mentioned_ds[0], mentioned_ds[1]
        return "\n".join(header + [compare_datasets(G, H, d1, d2)])

    # Intent C: Recommendations / similar datasets for a mentioned dataset
    if mentioned_ds and any(w in q_low for w in ("recommend", "similar", "like", "co-cit", "cocit", "related", "with")):
        target = mentioned_ds[0]
        novel = "novel" in q_low or "predict" in q_low or "unconnected" in q_low
        recs = recommend_datasets(G, H, target, top=8, novel_only=novel)
        return "\n".join(header + [format_recommendations(G, target, recs, novel_only=novel)])

    # Intent D: Bridge datasets (overall or within a field)
    if any(w in q_low for w in ("bridge", "interdisciplinary", "betweenness", "cross-domain")):
        field_match = None
        for n, d in G.nodes(data=True):
            if d["type"] == "Field" and d["label"].lower() in q_low:
                field_match = d["label"]
                break
        rows = find_bridge_datasets(G, H, field_query=field_match, top=10)
        return "\n".join(header + [format_bridges(rows, field_query=field_match)])

    # Intent E: Single dataset profile / impact question
    if len(mentioned_ds) == 1:
        prof = analyze_dataset_profile(G, H, mentioned_ds[0])
        return "\n".join(header + [format_profile(prof)])

    # Intent F: Field or topic search + subgraph summary
    for n, d in G.nodes(data=True):
        if d["type"] == "Field" and d["label"].lower() in q_low:
            ds_list = [u for u, _, k in G.in_edges(n, keys=True) if k == "HAS_THEME"]
            ds_list.sort(key=lambda x: -G.nodes[x].get("n_papers", 0))
            lines = header + [
                f"Matched Field of Study: {d['label']} [{n}] ({len(ds_list)} datasets in DSKG)",
                "",
                f"Top 10 Datasets in {d['label']} by Referencing Papers:",
            ]
            for i, ds in enumerate(ds_list[:10], 1):
                a = G.nodes[ds]
                lines.append(f"  {i:>2}. {a['label'][:42]:<42} [{ds:<12}] "
                             f"{a['n_papers']:>8,} papers  PageRank={a['pagerank']:.5f}  (C{a['community']})")
            return "\n".join(lines)

    # Fallback: Keyword search across dataset titles, keywords, fields, descriptions
    hits = search_datasets(G, question, limit=10)
    if hits:
        lines = header + [f"Top {len(hits)} Datasets Matching Query Semantics:"]
        for i, (ds, score) in enumerate(hits, 1):
            a = G.nodes[ds]
            flds = ", ".join(G.nodes[f]["label"] for f in out_by_rel(G, ds, "HAS_THEME"))
            lines.append(f"  {i:>2}. {a['label'][:40]:<40} [{ds:<12}] "
                         f"{a['n_papers']:>7,} papers  (C{a['community']})  Fields: {flds or 'N/A'}")
        top_prof = analyze_dataset_profile(G, H, hits[0][0])
        lines += ["", "--- Detailed Profile of Best Match ---", format_profile(top_prof)]
        return "\n".join(lines)

    return "\n".join(header + [
        "No matching entities found. Try mentioning a dataset (e.g. 'ImageNet', 'GenBank'), "
        "a field (e.g. 'Computer science', 'Biology'), or asking for 'bridge datasets'."
    ])


# --------------------------------------------------------------------------- #
# Interactive REPL for Live Grading Demo
# --------------------------------------------------------------------------- #
def run_repl(G: nx.MultiDiGraph, H: nx.Graph, out_dir: Path) -> None:
    print("=" * 74)
    print("DSKG INTERACTIVE GRAPH ENGINE SHELL")
    print(f"Loaded G ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges) "
          f"& H ({H.number_of_nodes():,} datasets, {H.number_of_edges():,} co-citations)")
    print("Commands:")
    print('  profile "ImageNet"               | recommend "ImageNet" [--novel]')
    print('  path "ImageNet" "GenBank" [--html]| compare "ImageNet" "CIFAR-10"')
    print('  bridges [FieldName]              | ask <any natural language question>')
    print("  quit / exit")
    print("=" * 74)

    while True:
        try:
            raw = input("\ndskg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            break
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = raw.split()
        cmd = parts[0].lower()
        rest = parts[1:]

        try:
            if cmd == "profile" and rest:
                ds = resolve_dataset(G, " ".join(rest))
                print(format_profile(analyze_dataset_profile(G, H, ds)))
            elif cmd == "recommend" and rest:
                novel = "--novel" in rest
                q_parts = [p for p in rest if p != "--novel"]
                ds = resolve_dataset(G, " ".join(q_parts))
                recs = recommend_datasets(G, H, ds, top=10, novel_only=novel)
                print(format_recommendations(G, ds, recs, novel_only=novel))
            elif cmd == "path" and len(rest) >= 2:
                do_html = "--html" in rest
                q_parts = [p for p in rest if p != "--html"]
                d1 = resolve_dataset(G, q_parts[0])
                d2 = resolve_dataset(G, q_parts[1])
                res = find_cross_domain_paths(G, H, d1, d2)
                print(format_path_result(G, res))
                if do_html:
                    sc = scene_path(G, H, res)
                    fname = f"path-{_slug(G.nodes[d1]['label'])}-to-{_slug(G.nodes[d2]['label'])}.html"
                    p = render(sc, out_dir / fname)
                    print(f"Wrote interactive path graph to {p}")
            elif cmd == "compare" and len(rest) >= 2:
                d1 = resolve_dataset(G, rest[0])
                d2 = resolve_dataset(G, rest[1])
                print(compare_datasets(G, H, d1, d2))
            elif cmd == "bridges":
                fq = " ".join(rest) if rest else None
                print(format_bridges(find_bridge_datasets(G, H, field_query=fq, top=15), fq))
            elif cmd == "ask" and rest:
                print(answer_question(G, H, " ".join(rest)))
            else:
                # Treat any free-form input as a natural-language Graph-RAG question
                print(answer_question(G, H, raw))
        except Exception as exc:
            print(f"Error: {exc}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    def add_globals(parser: argparse.ArgumentParser, defaults: bool) -> None:
        kw = (lambda d: {"default": d}) if defaults else (lambda d: {})
        parser.add_argument("--graph", type=Path, **kw(Path("data/processed/kg.pkl")))
        parser.add_argument("--out", type=Path, **kw(Path("output")))

    ap = argparse.ArgumentParser(
        prog="python -m src.graph_engine",
        description="Pillar 3: Query, recommendation, pathfinding and Graph-RAG engine.")
    add_globals(ap, defaults=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def new(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, argument_default=argparse.SUPPRESS)
        add_globals(sp, defaults=False)
        return sp

    p = new("profile", "inspect a dataset's full semantic profile & removal impact")
    p.add_argument("query", help="dataset title, number, or dataset:ID")

    p = new("recommend", "recommend complementary datasets (co-citation + Adamic-Adar + metadata)")
    p.add_argument("query", help="dataset title, number, or dataset:ID")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--novel", action="store_true", default=False,
                   help="only recommend datasets not yet directly co-cited with the query")

    p = new("path", "find shortest co-citation & semantic paths between two datasets")
    p.add_argument("source", help="first dataset title or dataset:ID")
    p.add_argument("target", help="second dataset title or dataset:ID")
    p.add_argument("--html", action="store_true", default=False,
                   help="render the discovered paths to an interactive HTML file in output/")
    p.add_argument("--open", action="store_true", default=False,
                   help="open the rendered HTML file in the browser")

    p = new("bridges", "rank interdisciplinary bridge datasets by betweenness centrality")
    p.add_argument("--field", type=str, default=None, help="optional field filter (e.g. 'Biology')")
    p.add_argument("--top", type=int, default=15)

    p = new("compare", "compare two datasets side by side")
    p.add_argument("dataset_a", help="first dataset")
    p.add_argument("dataset_b", help="second dataset")

    p = new("ask", "answer a natural-language question using Graph-RAG subgraph traversal")
    p.add_argument("question", help="natural language question in quotes")

    new("repl", "launch an interactive shell (loads kg.pkl once for instant demo queries)")
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
        if args.cmd == "profile":
            ds = resolve_dataset(G, args.query)
            print(format_profile(analyze_dataset_profile(G, H, ds)))
        elif args.cmd == "recommend":
            ds = resolve_dataset(G, args.query)
            recs = recommend_datasets(G, H, ds, top=args.top, novel_only=args.novel)
            print(format_recommendations(G, ds, recs, novel_only=args.novel))
        elif args.cmd == "path":
            d1 = resolve_dataset(G, args.source)
            d2 = resolve_dataset(G, args.target)
            res = find_cross_domain_paths(G, H, d1, d2)
            print(format_path_result(G, res))
            if args.html or args.open:
                sc = scene_path(G, H, res)
                fname = f"path-{_slug(G.nodes[d1]['label'])}-to-{_slug(G.nodes[d2]['label'])}.html"
                out_file = render(sc, args.out / fname)
                print(f"\nWrote interactive path view to {out_file} "
                      f"({len(sc.nodes)} nodes, {len(sc.edges)} edges)")
                if args.open:
                    webbrowser.open(out_file.resolve().as_uri())
        elif args.cmd == "bridges":
            rows = find_bridge_datasets(G, H, field_query=args.field, top=args.top)
            print(format_bridges(rows, field_query=args.field))
        elif args.cmd == "compare":
            d1 = resolve_dataset(G, args.dataset_a)
            d2 = resolve_dataset(G, args.dataset_b)
            print(compare_datasets(G, H, d1, d2))
        elif args.cmd == "ask":
            print(answer_question(G, H, args.question))
        elif args.cmd == "repl":
            run_repl(G, H, args.out)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())