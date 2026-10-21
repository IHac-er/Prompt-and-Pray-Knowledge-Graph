# DSKG Knowledge Graph (course project)

## Setup (Windows, PowerShell, from the project folder)

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    # copy DSKG.nt to data\raw\DSKG.nt

## Stage 1 - profile the data (done)

    python scripts\explore_nt.py

## Stage 2 - build the knowledge graph (done)

    python -m src.graph_builder                       # full build (~1 min)
    python -m src.graph_builder --max-datasets 200    # quick dev build

Outputs in data\processed\: kg.pkl, build_report.txt, build_stats.json,
dataset_metrics.csv, community_summary.csv, degree_distribution_papers.csv

## Stage 3 - interactive visualizations

Each command writes a self-contained HTML file to output\ (add --open to
open it in the browser). Loading the graph takes a few seconds per command.

    python -m src.visualizer overview
    python -m src.visualizer map                      # every dataset + connectors (heavy)
    python -m src.visualizer dataset "ImageNet"
    python -m src.visualizer community 2
    python -m src.visualizer field "Economics"
    python -m src.visualizer search "protein structure"

Layout tuning: add `--spacing 1.5` (or 2) to spread nodes further apart, or
`--spacing 0.8` to pull them closer. It works on every view, e.g.

    python -m src.visualizer map --spacing 1.5 --open

If PowerShell reports an encoding error while writing the page, run
`$env:PYTHONUTF8=1` first, then repeat the command.

## Stage 4 - query engine

    python -m src.graph_engine profile "ImageNet"
    python -m src.graph_engine recommend "ImageNet" --top 10
    python -m src.graph_engine recommend "GenBank" --novel
    python -m src.graph_engine path "ImageNet" "GenBank" --html --open
    python -m src.graph_engine bridges --field "Computer science"
    python -m src.graph_engine compare "ImageNet" "CIFAR-10"
    python -m src.graph_engine ask "How are ImageNet and CIFAR-10 connected?"
    python -m src.graph_engine repl        # loads the graph once, then instant queries
