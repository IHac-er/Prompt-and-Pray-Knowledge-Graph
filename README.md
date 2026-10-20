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

If PowerShell reports an encoding error while writing the page, run
`$env:PYTHONUTF8=1` first, then repeat the command.

## Tests

    python tests\test_graph_builder.py
    python tests\test_visualizer.py
