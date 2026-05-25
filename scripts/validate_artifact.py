#!/usr/bin/env python3
"""Validate the public artifact layout and headline frozen-evaluation values."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
FORBIDDEN_PATTERNS = (
    r"[A-Z]:\\",
    "Baidu" + "Syncdisk",
    "Users" + r"\\",
    "Chat" + "GP" + "T",
    "Open" + "A" + "I",
    "Gen" + "A" + "I",
)


REQUIRED_FILES = [
    "src/CON2.py",
    "src/Plot.py",
    "src/NodeMechanism.py",
    "results/frozen500/multi_seed_paired_significance.csv",
    "results/frozen500/multi_seed_cluster_episode_bootstrap.csv",
    "results/frozen500/frozen_reevaluation_summary.txt",
    "figures/frozen500/fig1_risk_frontier_core.pdf",
    "figures/frozen500/fig1_risk_frontier_core_plot_data.csv",
    "figures/mechanism_source/fig_mechanism_graph_minus_feature.pdf",
    "figures/mechanism_source/fig_mechanism_graph_minus_feature_bins.csv",
]


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def check_required_files() -> None:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).exists()]
    if missing:
        fail("missing required files: " + ", ".join(missing))


def check_checkpoints() -> None:
    checkpoint_root = ROOT / "checkpoints" / "source_50path"
    if not checkpoint_root.exists():
        fail("missing checkpoints/source_50path")
    model_count = len(list(checkpoint_root.glob("seed_*/tables/*_model.pt")))
    if model_count != 120:
        fail(f"expected 120 saved model checkpoints, found {model_count}")
    missing_summary = [
        str(checkpoint_root / f"seed_{seed}" / "run_summary.json")
        for seed in SEEDS
        if not (checkpoint_root / f"seed_{seed}" / "run_summary.json").exists()
    ]
    if missing_summary:
        fail("missing sanitized source summaries: " + ", ".join(missing_summary))


def check_forbidden_text() -> None:
    patterns = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_PATTERNS]
    text_suffixes = {".md", ".txt", ".json", ".csv", ".py", ".yml", ".yaml"}
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                hits.append(f"{rel}: {pattern.pattern}")
                break
    if hits:
        fail("forbidden local/private text found: " + "; ".join(hits[:20]))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_headline_cvar() -> None:
    path = ROOT / "results" / "frozen500" / "multi_seed_paired_significance.csv"
    rows = read_csv(path)
    wanted = {
        ("industry", "placement_vs_feature_only", "CVaR95_loss"): (-0.0095, -0.0109, -0.0079),
        ("industry", "learned_signal_vs_random_signal", "CVaR95_loss"): (0.0023, 0.0009, 0.0039),
        ("etf", "placement_vs_feature_only", "CVaR95_loss"): (-0.0084, -0.0094, -0.0074),
        ("etf", "learned_signal_vs_random_signal", "CVaR95_loss"): (0.0084, 0.0075, 0.0093),
    }
    index = {
        (r["dataset"], r["comparison_type"], r["metric"]): r
        for r in rows
    }
    for key, expected in wanted.items():
        if key not in index:
            fail(f"missing headline CVaR row: {key}")
        row = index[key]
        got = (
            round(float(row["delta"]), 4),
            round(float(row["ci_low"]), 4),
            round(float(row["ci_high"]), 4),
        )
        rounded_expected = tuple(round(x, 4) for x in expected)
        if got != rounded_expected:
            fail(f"unexpected values for {key}: got {got}, expected {rounded_expected}")


def main() -> None:
    check_required_files()
    check_checkpoints()
    check_forbidden_text()
    check_headline_cvar()
    print("Artifact validation passed.")
    print("Saved model checkpoints: 120")
    print("Frozen 500-path CVaR rows match expected rounded values.")


if __name__ == "__main__":
    main()
