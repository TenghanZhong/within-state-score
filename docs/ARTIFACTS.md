# Artifact Guide

## Code

- `src/CON2.py`: builds rolling graph-risk states, trains contract variants,
  saves source checkpoints, and performs frozen checkpoint re-evaluation.
- `src/Plot.py`: rebuilds aggregate paper figures from multi-seed CSV outputs.
- `src/NodeMechanism.py`: recomputes asset-level optimal exposure under saved
  contract coefficients and summarizes exposure changes by graph-risk quintile.

## Main Results

- `results/frozen500/multi_seed_paired_significance.csv`: seed-level paired
  tests and main pooled CVaR95 rows.
- `results/frozen500/multi_seed_cluster_episode_bootstrap.csv`: clustered
  bootstrap intervals for episode-averaged and state-averaged metrics.
- `results/frozen500/claim_diagnostics.csv`: compact diagnostic checks.
- `results/frozen500/method_design_audit.csv`: method-design audit
  table for matched architecture, feasibility, and control settings.

## Source Checkpoints

- `checkpoints/source_50path/seed_*/tables/*_model.pt`: saved PyTorch state
  dictionaries for all neural variants.
- `checkpoints/source_50path/seed_*/run_summary.json`: sanitized source-run
  configuration used by frozen re-evaluation.

The source checkpoints are included so the frozen 500-path evaluation can be
re-run without retraining.

## Figures

- `figures/frozen500/fig1_risk_frontier_core.*`: crowding versus pooled test-set
  CVaR95 loss.
- `figures/frozen500/*_plot_data.csv`: plot data used for aggregate figures.
- `figures/mechanism_source/fig_mechanism_graph_minus_feature.*`: asset-level
  mechanism diagnostic.
- `figures/mechanism_source/fig_mechanism_graph_minus_feature_bins.csv`:
  seed-level bin means used for the mechanism figure.
