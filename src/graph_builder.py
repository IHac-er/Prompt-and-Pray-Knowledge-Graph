"""Pillar 1 - Knowledge graph construction from the DSKG N-Triples dump.

Reads ``DSKG.nt`` (one triple per line, so it is streamed with a plain parser
instead of loading everything into rdflib) and builds:

* ``G``  - a typed ``networkx.MultiDiGraph`` (the full knowledge graph)
* ``H``  - a weighted dataset-to-dataset co-citation graph (two datasets are
  linked when papers mention both). Communities and centrality are computed
  on ``H`` because papers only connect to datasets in the raw data.

Node types : Dataset, Paper, Field, Person, Organization, Keyword
Relations  : REFERENCED_BY (Dataset->Paper), HAS_THEME (Dataset->Field),
             CREATED_BY, PUBLISHED_BY, CONTRIBUTED_BY (Dataset->Person/Org),
             HAS_KEYWORD (Dataset->Keyword).  The relation is the edge key.

Normalization / deduplication
* keywords are case-folded, unescaped, de-underscored and de-duplicated;
* foaf:Organization and foaf:Agent records are merged by normalized name;
* person records with identical normalized names are reported, and merged
  only with ``--merge-persons`` (DSKG already ran author disambiguation).

Usage (from the project root):
    python -m src.graph_builder
    python -m src.graph_builder --max-datasets 300      # quick dev build
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterator

import networkx as nx

log = logging.getLogger("graph_builder")

DSKG_PREFIX = "http://dskg.org/entity/"
SCHEMA_VERSION = 1

PREDICATES: dict[str, str] = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "type",
    "http://purl.org/dc/terms/isReferencedBy": "isReferencedBy",
    "http://purl.org/dc/terms/title": "title",
    "http://purl.org/dc/terms/description": "description",
    "http://purl.org/dc/terms/alternative": "alternative",
    "http://purl.org/dc/terms/modified": "modified",
    "http://purl.org/dc/terms/issued": "issued",
    "http://purl.org/dc/terms/creator": "creator",
    "http://purl.org/dc/terms/publisher": "publisher",
    "http://purl.org/dc/terms/contributor": "contributor",
    "http://purl.org/dc/terms/source": "source",
    "http://purl.org/dc/terms/language": "language",
    "http://purl.org/dc/terms/identifier": "identifier",
    "http://purl.org/dc/terms/accessRights": "accessRights",
    "http://purl.org/dc/terms/licence": "licence",
    "http://purl.org/dc/terms/format": "format",
    "http://purl.org/dc/terms/hasVersion": "hasVersion",
    "http://www.w3.org/ns/dcat#keyword": "keyword",
    "http://www.w3.org/ns/dcat#landingPage": "landingPage",
    "http://www.w3.org/ns/dcat#theme": "theme",
    "http://www.w3.org/ns/dcat#distribution": "distribution",
    "http://www.w3.org/ns/dcat#accessURL": "accessURL",
    "http://www.w3.org/ns/dcat#byteSize": "byteSize",
    "http://www.w3.org/ns/dcat#contactPoint": "contactPoint",
    "http://xmlns.com/foaf/0.1/name": "name",
    "https://www.w3.org/ns/adms#identifier": "admsIdentifier",
    "http://www.w3.org/2006/vcard/ns#fn": "fn",
}

# --------------------------------------------------------------------------- #
# N-Triples parsing
# --------------------------------------------------------------------------- #
_LINE_RE = re.compile(r'^(<[^>]*>|_:\S+)\s+<([^>]*)>\s+(.+?)\s*\.\s*$')
_URI_RE = re.compile(r"^<([^>]*)>$")
_LIT_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"(?:\^\^<[^>]*>|@[A-Za-z0-9-]+)?$', re.S)
_ESC_RE = re.compile(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)", re.S)
_SIMPLE_ESC = {"t": "\t", "n": "\n", "r": "\r", "b": "\b", "f": "\f",
               '"': '"', "'": "'", "\\": "\\"}


def unescape(text: str) -> str:
    """Decode N-Triples string escapes (\\n, \\", \\uXXXX, ...)."""
    if "\\" not in text:
        return text

    def _rep(m: re.Match[str]) -> str:
        g = m.group(1)
        if len(g) > 1 and g[0] in "uU":
            try:
                return chr(int(g[1:], 16))
            except (ValueError, OverflowError):
                return m.group(0)
        return _SIMPLE_ESC.get(g, m.group(0))

    return _ESC_RE.sub(_rep, text)


def iter_triples(path: Path) -> Iterator[tuple[str, str, str, bool]]:
    """Yield ``(subject_uri, predicate_uri, value, value_is_uri)``.

    Blank-node subjects and malformed lines are skipped (and counted by the
    caller through the difference between lines read and triples yielded).
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _LINE_RE.match(line)
            if not m or not m.group(1).startswith("<"):
                continue
            subj, pred, obj = m.groups()
            um = _URI_RE.match(obj)
            if um:
                yield subj[1:-1], pred, um.group(1), True
                continue
            lm = _LIT_RE.match(obj)
            if lm:
                yield subj[1:-1], pred, unescape(lm.group(1)), False


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Case-fold, strip accents-insensitive punctuation and collapse spaces."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, limit: int | None = None) -> str:
    """Collapse whitespace; optionally truncate (with an ellipsis)."""
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def uri_tail(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]


def dskg_entity(uri: str) -> tuple[str, str] | None:
    """('dataset', '1') for http://dskg.org/entity/dataset/1, else None."""
    if not uri.startswith(DSKG_PREFIX):
        return None
    parts = uri[len(DSKG_PREFIX):].split("/")
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def to_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Pass 1: read the dump into plain Python structures
# --------------------------------------------------------------------------- #
@dataclass
class RawData:
    """Everything extracted from the dump, before graph construction."""
    entities: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    paper_links: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    unknown_predicates: Counter = field(default_factory=Counter)
    n_triples: int = 0


def read_dump(nt_path: Path) -> RawData:
    """Stream the .nt file once and group triples by subject."""
    if not nt_path.exists():
        raise FileNotFoundError(
            f"{nt_path} not found. Copy DSKG.nt into data/raw/ (see README).")
    raw = RawData()
    t0 = time.time()
    for subj, pred, value, _is_uri in iter_triples(nt_path):
        raw.n_triples += 1
        name = PREDICATES.get(pred)
        if name is None:
            raw.unknown_predicates[pred] += 1
            continue
        if name == "isReferencedBy":
            raw.paper_links[subj].append(value)
        else:
            raw.entities[subj][name].append(value)
    log.info("Read %s triples in %.1fs (%s entities, %s paper-link subjects)",
             f"{raw.n_triples:,}", time.time() - t0,
             f"{len(raw.entities):,}", f"{len(raw.paper_links):,}")
    if not raw.paper_links and not raw.entities:
        raise ValueError("No usable triples found - is this the DSKG .nt file?")
    return raw


def load_field_labels(path: Path | None) -> dict[str, str]:
    """Optional CSV (id,label) mapping MAG field ids to readable names."""
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["id"].strip(): row["label"].strip()
                for row in csv.DictReader(fh) if row.get("id")}


# --------------------------------------------------------------------------- #
# Pass 2: graph construction
# --------------------------------------------------------------------------- #
def _origin(sources: list[str]) -> str:
    joined = " ".join(sources).lower()
    if "wikidata.org" in joined:
        return "Wikidata"
    if "zenodo.org" in joined:
        return "OpenAIRE"
    return "Unknown"


def build_graph(raw: RawData, field_labels: dict[str, str] | None = None,
                merge_persons: bool = False, max_datasets: int | None = None
                ) -> tuple[nx.MultiDiGraph, dict]:
    """Build the typed multigraph. Returns ``(G, stats)``."""
    field_labels = field_labels or {}
    G = nx.MultiDiGraph()
    stats: dict = {"dedup": {}}

    # (distributions are folded into their dataset as attributes below)

    dataset_uris = sorted(
        (u for u in raw.entities if (e := dskg_entity(u)) and e[0] == "dataset"),
        key=lambda u: int(dskg_entity(u)[1]) if dskg_entity(u)[1].isdigit() else 0)
    if max_datasets:
        dataset_uris = dataset_uris[:max_datasets]
    if not dataset_uris:
        raise ValueError("No datasets found in the dump.")

    # ---- agents (person / organization / agent) are created lazily
    org_by_name: dict[str, str] = {}
    person_by_name: dict[str, str] = {}
    person_name_nodes: defaultdict[str, set] = defaultdict(set)
    agent_node: dict[str, str] = {}
    merged_org_records = 0

    def get_agent(uri: str) -> str | None:
        nonlocal merged_org_records
        if uri in agent_node:
            return agent_node[uri]
        ent = dskg_entity(uri)
        if not ent or ent[0] not in {"person", "organization", "agent"}:
            return None
        kind, local = ent
        props = raw.entities.get(uri, {})
        name = clean_text((props.get("name") or [""])[0]) or f"{kind} {local}"
        norm = normalize_text(name)
        if kind == "person":
            nid = f"person:{local}"
            if merge_persons and norm in person_by_name:
                nid = person_by_name[norm]
            else:
                person_by_name.setdefault(norm, nid)
            person_name_nodes[norm].add(nid)
            if nid not in G:
                G.add_node(nid, type="Person", label=name,
                           orcid=next((i for i in props.get("admsIdentifier", [])
                                       if "orcid" in i.lower()), None))
        else:
            if norm in org_by_name:
                nid = org_by_name[norm]
                merged_org_records += 1
                G.nodes[nid]["source_kinds"] = sorted(
                    set(G.nodes[nid]["source_kinds"]) | {kind})
            else:
                nid = f"org:{local}"
                org_by_name[norm] = nid
                G.add_node(nid, type="Organization", label=name,
                           source_kinds=[kind])
        agent_node[uri] = nid
        return nid

    kw_seen: Counter[str] = Counter()
    kw_raw_count = 0
    edge_counts: Counter[str] = Counter()

    def link(u: str, v: str, rel: str) -> None:
        if not G.has_edge(u, v, key=rel):
            G.add_edge(u, v, key=rel)
            edge_counts[rel] += 1

    for ds_uri in dataset_uris:
        props = raw.entities[ds_uri]
        local = dskg_entity(ds_uri)[1]
        node = f"dataset:{local}"

        # distribution attributes
        formats: set[str] = set()
        licences: set[str] = set()
        sizes: list[float] = []
        access: list[str] = []
        for d_uri in props.get("distribution", []):
            dp = raw.entities.get(d_uri, {})
            formats.update(f.strip().lower() for f in dp.get("format", []))
            licences.update(l.strip() for l in dp.get("licence", []))
            sizes += [s for s in map(to_float, dp.get("byteSize", [])) if s is not None]
            access += dp.get("accessURL", [])

        ids = props.get("admsIdentifier", [])
        doi = next((i for i in ids if "doi.org" in i.lower()), ids[0] if ids else None)
        contact = None
        for k in props.get("contactPoint", []):
            fn = raw.entities.get(k, {}).get("fn")
            if fn:
                contact = clean_text(fn[0])
                break

        G.add_node(
            node, type="Dataset",
            label=clean_text((props.get("title") or [f"Dataset {local}"])[0]),
            description=clean_text((props.get("description") or [""])[0], 1500),
            origin=_origin(props.get("source", [])),
            identifier=doi,
            issued=(props.get("issued") or [None])[0],
            modified=(props.get("modified") or [None])[0],
            language=(props.get("language") or [None])[0],
            access_rights=(props.get("accessRights") or [None])[0],
            aliases=[clean_text(a) for a in props.get("alternative", [])][:5],
            formats=sorted(formats), licences=sorted(licences),
            byte_size=max(sizes) if sizes else None,
            landing_pages=(props.get("landingPage", []) + access)[:3],
            contact=contact,
        )

        for paper_uri in props.get("isReferencedBy", []) + raw.paper_links.get(ds_uri, []):
            pid = f"paper:{uri_tail(paper_uri)}"
            if pid not in G:
                G.add_node(pid, type="Paper")
            link(node, pid, "REFERENCED_BY")

        for theme in props.get("theme", []):
            fid_local = uri_tail(theme)
            fid = f"field:{fid_local}"
            if fid not in G:
                G.add_node(fid, type="Field",
                           label=field_labels.get(fid_local, f"Field {fid_local}"))
            link(node, fid, "HAS_THEME")

        for pred, rel in (("creator", "CREATED_BY"), ("publisher", "PUBLISHED_BY"),
                          ("contributor", "CONTRIBUTED_BY")):
            for a_uri in props.get(pred, []):
                a_node = get_agent(a_uri)
                if a_node:
                    link(node, a_node, rel)

        for kw in props.get("keyword", []):
            kw_raw_count += 1
            norm = normalize_text(kw)
            if not norm:
                continue
            kw_seen[norm] += 1
            kid = f"keyword:{norm}"
            if kid not in G:
                G.add_node(kid, type="Keyword", label=norm)
            link(node, kid, "HAS_KEYWORD")

    # DSKG has no explicit paper nodes; datasets' paper counts become attributes
    for ds_uri in dataset_uris:
        node = f"dataset:{dskg_entity(ds_uri)[1]}"
        G.nodes[node]["n_papers"] = sum(
            1 for _, _, k in G.out_edges(node, keys=True) if k == "REFERENCED_BY")

    person_collisions = sum(1 for nodes in person_name_nodes.values() if len(nodes) > 1)
    stats["dedup"] = {
        "keyword_mentions": kw_raw_count,
        "unique_keywords_after_normalization": len(kw_seen),
        "organization_agent_records_merged": merged_org_records,
        "person_names_shared_by_multiple_records": person_collisions,
        "persons_merged": merge_persons,
    }
    stats["edge_counts"] = dict(edge_counts)
    log.info("Graph: %s nodes, %s edges", f"{G.number_of_nodes():,}",
             f"{G.number_of_edges():,}")
    return G, stats


# --------------------------------------------------------------------------- #
# Helpers over the built graph
# --------------------------------------------------------------------------- #
def out_by_rel(G: nx.MultiDiGraph, node: str, rel: str) -> list[str]:
    """Targets of ``node``'s outgoing edges with relation ``rel``."""
    return [v for _, v, k in G.out_edges(node, keys=True) if k == rel]


def dataset_nodes(G: nx.MultiDiGraph) -> list[str]:
    return [n for n, d in G.nodes(data=True) if d.get("type") == "Dataset"]


# --------------------------------------------------------------------------- #
# Metrics: projection, centrality, communities
# --------------------------------------------------------------------------- #
def build_dataset_projection(G: nx.MultiDiGraph) -> tuple[nx.Graph, Counter]:
    """Weighted dataset co-citation graph.

    Edge weight = number of papers that reference both datasets;
    ``jaccard`` = weight / papers referencing either.
    Also returns the distribution ``{datasets cited by a paper: n_papers}``.
    """
    datasets = sorted(dataset_nodes(G))
    index = {d: i for i, d in enumerate(datasets)}
    n = len(datasets)
    paper_to_ds: dict[str, list[int]] = defaultdict(list)
    for d in datasets:
        for p in out_by_rel(G, d, "REFERENCED_BY"):
            paper_to_ds[p].append(index[d])

    pair_counts: Counter[int] = Counter()
    for dss in paper_to_ds.values():
        if len(dss) < 2:
            continue
        dss.sort()
        for a, b in combinations(dss, 2):
            pair_counts[a * n + b] += 1

    H = nx.Graph()
    for d in datasets:
        H.add_node(d, label=G.nodes[d]["label"], n_papers=G.nodes[d].get("n_papers", 0))
    for key, shared in pair_counts.items():
        a, b = divmod(key, n)
        na, nb = datasets[a], datasets[b]
        union = H.nodes[na]["n_papers"] + H.nodes[nb]["n_papers"] - shared
        H.add_edge(na, nb, weight=shared, jaccard=shared / union if union else 0.0)

    papers_per_cite = Counter(len(v) for v in paper_to_ds.values())
    log.info("Projection: %s dataset nodes, %s co-citation edges",
             f"{H.number_of_nodes():,}", f"{H.number_of_edges():,}")
    return H, papers_per_cite


def weighted_pagerank(H: nx.Graph, alpha: float = 0.85, tol: float = 1e-8,
                      max_iter: int = 100) -> dict[str, float]:
    """Small dependency-free weighted PageRank (power iteration)."""
    nodes = list(H)
    n = len(nodes)
    if n == 0:
        return {}
    strength = {u: sum(d["weight"] for d in H[u].values()) for u in nodes}
    pr = {u: 1.0 / n for u in nodes}
    for _ in range(max_iter):
        dangling = sum(pr[u] for u in nodes if strength[u] == 0)
        new = {u: (1 - alpha) / n + alpha * dangling / n for u in nodes}
        for u in nodes:
            if strength[u] == 0:
                continue
            share = alpha * pr[u] / strength[u]
            for v, d in H[u].items():
                new[v] += share * d["weight"]
        err = sum(abs(new[u] - pr[u]) for u in nodes)
        pr = new
        if err < n * tol:
            break
    return pr


def detect_communities(H: nx.Graph, seed: int = 42) -> tuple[dict[str, int], float]:
    """Louvain communities on ``H``. Singletons get id -1 ("isolated")."""
    if H.number_of_edges() == 0:
        return {n: -1 for n in H}, 0.0
    comms = nx.community.louvain_communities(H, weight="weight", seed=seed)
    modularity = nx.community.modularity(H, comms, weight="weight")
    comms = sorted(comms, key=len, reverse=True)
    assign: dict[str, int] = {}
    next_id = 0
    for c in comms:
        if len(c) == 1:
            for n in c:
                assign[n] = -1
        else:
            for n in c:
                assign[n] = next_id
            next_id += 1
    return assign, modularity


def compute_metrics(G: nx.MultiDiGraph, H: nx.Graph, use_betweenness: bool = False
                    ) -> dict:
    """Annotate nodes with centrality/community and return summary stats."""
    pagerank = weighted_pagerank(H)
    communities, modularity = detect_communities(H)
    w_degree = {u: sum(d["weight"] for d in H[u].values()) for u in H}
    betweenness: dict[str, float] = {}
    if use_betweenness and H.number_of_edges():
        k = min(200, H.number_of_nodes())
        betweenness = nx.betweenness_centrality(H, k=k, seed=42)

    n_total = max(G.number_of_nodes() - 1, 1)
    for node, data in G.nodes(data=True):
        if data["type"] != "Paper":
            data["degree"] = G.degree(node)
            data["degree_centrality"] = G.degree(node) / n_total
    for d in H:
        attrs = G.nodes[d]
        attrs["pagerank"] = pagerank[d]
        attrs["weighted_degree"] = w_degree[d]
        attrs["community"] = communities[d]
        attrs["betweenness"] = betweenness.get(d)
        H.nodes[d].update(community=communities[d], pagerank=pagerank[d])

    comp_sizes = sorted((len(c) for c in nx.connected_components(H)), reverse=True)
    sizes = Counter(c for c in communities.values() if c >= 0)
    return {
        "projection_nodes": H.number_of_nodes(),
        "projection_edges": H.number_of_edges(),
        "projection_components": len(comp_sizes),
        "projection_largest_component": comp_sizes[0] if comp_sizes else 0,
        "projection_density": nx.density(H),
        "projection_avg_weighted_degree":
            sum(w_degree.values()) / len(w_degree) if w_degree else 0.0,
        "n_communities": len(sizes),
        "isolated_datasets": sum(1 for c in communities.values() if c == -1),
        "modularity": modularity,
        "community_sizes": dict(sorted(sizes.items())),
    }


# --------------------------------------------------------------------------- #
# Output: pickle, CSVs and a human-readable report
# --------------------------------------------------------------------------- #
def save_graph(G: nx.MultiDiGraph, H: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"version": SCHEMA_VERSION, "graph": G, "dataset_graph": H},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: Path) -> tuple[nx.MultiDiGraph, nx.Graph]:
    """Load the pickle written by :func:`save_graph`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the build step first.")
    with path.open("rb") as fh:
        blob = pickle.load(fh)
    if blob.get("version") != SCHEMA_VERSION:
        raise ValueError("Graph file was built by an incompatible version; rebuild.")
    return blob["graph"], blob["dataset_graph"]


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def write_outputs(G: nx.MultiDiGraph, H: nx.Graph, stats: dict,
                  papers_per_cite: Counter, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(dataset_nodes(G),
                      key=lambda n: G.nodes[n]["n_papers"], reverse=True)

    rows = []
    for d in datasets:
        a = G.nodes[d]
        fields = [G.nodes[f]["label"] for f in out_by_rel(G, d, "HAS_THEME")]
        rows.append([d, a["label"], a["origin"], a["n_papers"], "; ".join(fields),
                     len(out_by_rel(G, d, "CREATED_BY")), a["community"],
                     f'{a["pagerank"]:.6f}', a["weighted_degree"],
                     H.degree(d) if d in H else 0])
    _write_csv(out_dir / "dataset_metrics.csv",
               ["node", "title", "origin", "n_papers", "fields", "n_creators",
                "community", "pagerank", "weighted_degree", "co_cited_datasets"], rows)

    _write_csv(out_dir / "degree_distribution_papers.csv",
               ["datasets_referenced_by_paper", "n_papers"],
               sorted(papers_per_cite.items()))

    # community summaries
    by_comm: defaultdict[int, list[str]] = defaultdict(list)
    for d in datasets:
        by_comm[G.nodes[d]["community"]].append(d)
    comm_rows = []
    for cid, members in sorted(by_comm.items(), key=lambda kv: -len(kv[1])):
        if cid == -1:
            continue
        fields = Counter(G.nodes[f]["label"] for d in members
                         for f in out_by_rel(G, d, "HAS_THEME"))
        kws = Counter(G.nodes[k]["label"] for d in members
                      for k in out_by_rel(G, d, "HAS_KEYWORD"))
        top = sorted(members, key=lambda d: -G.nodes[d]["n_papers"])[:3]
        comm_rows.append([cid, len(members),
                          "; ".join(f"{n} ({c})" for n, c in fields.most_common(3)),
                          "; ".join(k for k, _ in kws.most_common(5)),
                          " | ".join(G.nodes[d]["label"] for d in top)])
    _write_csv(out_dir / "community_summary.csv",
               ["community", "n_datasets", "top_fields", "top_keywords",
                "top_datasets_by_papers"], comm_rows)

    # report
    type_counts = Counter(d["type"] for _, d in G.nodes(data=True))
    lines = ["KNOWLEDGE GRAPH BUILD REPORT", "=" * 60,
             f"Nodes: {G.number_of_nodes():,}   Edges: {G.number_of_edges():,}",
             "", "Nodes by type:"]
    lines += [f"  {t:<14}{c:>10,}" for t, c in type_counts.most_common()]
    lines += ["", "Edges by relation:"]
    lines += [f"  {r:<16}{c:>10,}" for r, c in
              sorted(stats["edge_counts"].items(), key=lambda kv: -kv[1])]
    lines += ["", "Deduplication:"]
    lines += [f"  {k}: {v}" for k, v in stats["dedup"].items()]
    lines += ["", "Dataset co-citation projection:"]
    for k in ("projection_nodes", "projection_edges", "projection_components",
              "projection_largest_component", "n_communities", "isolated_datasets"):
        lines.append(f"  {k}: {stats[k]:,}")
    lines.append(f"  modularity: {stats['modularity']:.3f}")
    lines.append(f"  density: {stats['projection_density']:.4f}")

    lines += ["", "Top 15 datasets by number of referencing papers:"]
    for d in datasets[:15]:
        a = G.nodes[d]
        lines.append(f"  {a['n_papers']:>7,}  {a['label'][:60]}  (community {a['community']})")

    lines += ["", "Papers by number of datasets they reference (top 8 buckets):"]
    for k, v in sorted(papers_per_cite.items())[:8]:
        lines.append(f"  {k} dataset(s): {v:,} papers")

    top_kw = sorted(((G.degree(n), G.nodes[n]["label"]) for n in G
                     if G.nodes[n]["type"] == "Keyword"), reverse=True)[:15]
    lines += ["", "Top keywords: " + ", ".join(f"{l} ({c})" for c, l in top_kw)]

    lines += ["", "Field sanity check (verify labels against sample titles):"]
    fields_sorted = sorted((n for n in G if G.nodes[n]["type"] == "Field"),
                           key=lambda n: -G.in_degree(n))
    for f in fields_sorted[:12]:
        titles = [G.nodes[u]["label"][:35] for u, _, k in G.in_edges(f, keys=True)
                  if k == "HAS_THEME"][:3]
        lines.append(f"  {G.nodes[f]['label']} [{f}] - {G.in_degree(f)} datasets, "
                     f"e.g. {' | '.join(titles)}")

    (out_dir / "build_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "build_stats.json").write_text(json.dumps(
        {"node_types": dict(type_counts), **stats}, indent=2, default=str),
        encoding="utf-8")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_build(nt: Path, out_dir: Path, field_labels_path: Path | None,
              merge_persons: bool = False, max_datasets: int | None = None,
              betweenness: bool = False) -> None:
    raw = read_dump(nt)
    G, stats = build_graph(raw, load_field_labels(field_labels_path),
                           merge_persons=merge_persons, max_datasets=max_datasets)
    del raw
    H, papers_per_cite = build_dataset_projection(G)
    stats.update(compute_metrics(G, H, use_betweenness=betweenness))
    save_graph(G, H, out_dir / "kg.pkl")
    write_outputs(G, H, stats, papers_per_cite, out_dir)
    print(f"\nSaved graph and reports to {out_dir.resolve()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the DSKG knowledge graph.")
    ap.add_argument("--nt", type=Path, default=Path("data/raw/DSKG.nt"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--field-labels", type=Path, default=Path("data/field_labels.csv"))
    ap.add_argument("--merge-persons", action="store_true",
                    help="merge person records with identical normalized names")
    ap.add_argument("--max-datasets", type=int, default=None,
                    help="build from only the first N datasets (fast dev build)")
    ap.add_argument("--betweenness", action="store_true",
                    help="also compute sampled betweenness centrality (slower)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    try:
        run_build(args.nt, args.out, args.field_labels, args.merge_persons,
                  args.max_datasets, args.betweenness)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
