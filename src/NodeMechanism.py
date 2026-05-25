"""Node-level exposure mechanism check for the graph-penalty contract.

This script uses existing multi-seed outputs. It rebuilds the public test
states, reads saved learned coefficients, recomputes asset-level optimal
exposures, and plots exposure changes by graph-risk-score quintile. The
mechanism diagnostic uses saved trained coefficients from the source multi-seed
run and is invariant to the frozen 500-path payoff re-evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import CON2 as con


DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "raw"
DEFAULT_BASE_OUT = PROJECT_DIR / "checkpoints" / "source_50path"
DEFAULT_FIG_DIR = PROJECT_DIR / "figures" / "mechanism_rebuilt"
DEFAULT_SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def config_from_summary(base_out: Path, seed: int) -> con.ExperimentConfig:
    summary_path = base_out / f"seed_{seed}" / "run_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        cfg_dict = json.load(f)["config"]
    cfg_dict["random_seed"] = seed
    fields = {f.name for f in dataclasses.fields(con.ExperimentConfig)}
    clean = {k: v for k, v in cfg_dict.items() if k in fields}
    return con.ExperimentConfig(**clean)


def build_tests(data_dir: Path, cfg: con.ExperimentConfig) -> Dict[str, con.EpisodeSet]:
    rng = np.random.default_rng(cfg.random_seed)
    data = con.load_all_data(data_dir)

    returns, factor, vix = con.prepare_industry_returns(data, cfg)
    industry_es = con.build_episode_set(
        name="industry49",
        returns=returns,
        factor=factor,
        vix=vix,
        cfg=cfg,
        start_date=cfg.industry_start,
        end_date=cfg.test_end,
        rng=rng,
    )
    train = industry_es.subset_by_date(cfg.industry_start, cfg.train_end, "industry_train")
    val = industry_es.subset_by_date(cfg.val_start, cfg.val_end, "industry_val")
    test = industry_es.subset_by_date(cfg.test_start, cfg.test_end, "industry_test")
    _, _, industry_test, _ = con.standardize_feature_sets(train, val, test)

    out = {"industry": industry_test}
    etf_prepared = con.prepare_etf_returns(data, cfg)
    if etf_prepared is not None:
        etf_returns, etf_factor, etf_vix = etf_prepared
        etf_es = con.build_episode_set(
            name="ETF",
            returns=etf_returns,
            factor=etf_factor,
            vix=etf_vix,
            cfg=cfg,
            start_date=cfg.etf_start,
            end_date=cfg.etf_test_end,
            rng=rng,
        )
        etf_train = etf_es.subset_by_date(cfg.etf_start, cfg.etf_train_end, "etf_train")
        etf_val = etf_es.subset_by_date(cfg.etf_val_start, cfg.etf_val_end, "etf_val")
        etf_test = etf_es.subset_by_date(cfg.etf_test_start, cfg.etf_test_end, "etf_test")
        _, _, etf_test, _ = con.standardize_feature_sets(etf_train, etf_val, etf_test)
        out["etf"] = etf_test
    return out


def read_loadings(base_out: Path, seed: int, dataset: str) -> pd.DataFrame:
    name = "industry_episode_details_neural.csv" if dataset == "industry" else "etf_episode_details_neural.csv"
    path = base_out / f"seed_{seed}" / "tables" / name
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def method_names(dataset: str) -> Tuple[str, str]:
    if dataset == "industry":
        return "GraphSignal-ICL-NoPenalty", "GraphSignal-ICL"
    return "ETF_GraphSignal-ICL-NoPenalty", "ETF_GraphSignal-ICL"


def solve_with_episode_loadings(
    es: con.EpisodeSet,
    cfg: con.ExperimentConfig,
    b: int,
    row: pd.Series,
    m: np.ndarray,
    lambda_override: float | None = None,
) -> np.ndarray:
    n = es.N
    theta = np.full(n, float(row["theta"]))
    eta = np.full(n, float(row["eta"]))
    lambd_value = float(row["lambda"]) if lambda_override is None else float(lambda_override)
    lambd = np.full(n, lambd_value)
    _, s_vec = con.solve_best_response_np(theta, eta, lambd, es.W_gls[b], m, es, b, cfg)
    return s_vec


def node_rows_for_seed(base_out: Path, data_dir: Path, seed: int) -> List[Dict[str, object]]:
    cfg = config_from_summary(base_out, seed)
    tests = build_tests(data_dir, cfg)
    rows: List[Dict[str, object]] = []

    for dataset, es in tests.items():
        details = read_loadings(base_out, seed, dataset)
        feature_name, graph_name = method_names(dataset)
        feature = details[details["method"] == feature_name].set_index("date")
        graph = details[details["method"] == graph_name].set_index("date")

        for b, date in enumerate(es.dates):
            f_row = feature.loc[pd.Timestamp(date)]
            g_row = graph.loc[pd.Timestamp(date)]
            m = es.m_tail_cvar[b]
            s_feature = solve_with_episode_loadings(es, cfg, b, f_row, m, lambda_override=0.0)
            s_graph = solve_with_episode_loadings(es, cfg, b, g_row, m)
            s_graph_no_penalty = solve_with_episode_loadings(es, cfg, b, g_row, m, lambda_override=0.0)

            for j, symbol in enumerate(es.symbols):
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "date": pd.Timestamp(date).date().isoformat(),
                        "node": j,
                        "symbol": symbol,
                        "m_tail_cvar": float(m[j]),
                        "s_feature_only": float(s_feature[j]),
                        "s_graph_penalty": float(s_graph[j]),
                        "s_graph_no_penalty": float(s_graph_no_penalty[j]),
                        "delta_graph_minus_feature": float(s_graph[j] - s_feature[j]),
                        "delta_penalty_channel": float(s_graph[j] - s_graph_no_penalty[j]),
                    }
                )
    return rows


def assign_bins(df: pd.DataFrame, value_col: str = "m_tail_cvar", bins: int = 5) -> pd.DataFrame:
    out = df.copy()
    labels = [f"Q{i}" for i in range(1, bins + 1)]
    for dataset in out["dataset"].unique():
        mask = out["dataset"] == dataset
        ranks = out.loc[mask, value_col].rank(method="first")
        out.loc[mask, "score_bin"] = pd.qcut(ranks, bins, labels=labels)
    return out


def summarize_bins(df: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    rows = []
    for (dataset, score_bin, seed), g in df.groupby(["dataset", "score_bin", "seed"], observed=True):
        rows.append(
            {
                "dataset": dataset,
                "score_bin": str(score_bin),
                "seed": int(seed),
                "mean_delta": float(g[delta_col].mean()),
                "n": int(len(g)),
            }
        )
    seed_df = pd.DataFrame(rows)
    out_rows = []
    for (dataset, score_bin), g in seed_df.groupby(["dataset", "score_bin"], observed=True):
        vals = g["mean_delta"].to_numpy(float)
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out_rows.append(
            {
                "dataset": dataset,
                "score_bin": score_bin,
                "mean_delta": mean,
                "ci_low": mean - 1.96 * se,
                "ci_high": mean + 1.96 * se,
                "seed_count": int(len(vals)),
                "node_episode_count": int(g["n"].sum()),
            }
        )
    order = {"industry": 0, "etf": 1}
    return pd.DataFrame(out_rows).sort_values(["dataset", "score_bin"], key=lambda s: s.map(order).fillna(s))


def plot_mechanism(summary: pd.DataFrame, fig_dir: Path, stem: str, title_delta: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d for d in ["industry", "etf"] if d in set(summary["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(8.2, 3.1), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    colors = {"industry": "#2F6B9A", "etf": "#B45F3C"}

    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset].copy()
        sub["x"] = sub["score_bin"].str.extract(r"Q(\d+)").astype(int)
        sub = sub.sort_values("x")
        y = sub["mean_delta"].to_numpy(float)
        lo = y - sub["ci_low"].to_numpy(float)
        hi = sub["ci_high"].to_numpy(float) - y
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.errorbar(
            sub["x"],
            y,
            yerr=np.vstack([lo, hi]),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors.get(dataset, "#444444"),
        )
        ax.set_title("(a) Industry" if dataset == "industry" else "(b) ETF")
        ax.set_xlabel("Graph-risk score quintile")
        ax.set_xticks(sub["x"])
        ax.set_xticklabels(sub["score_bin"])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel(title_delta)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def parse_seeds(text: str) -> Tuple[int, ...]:
    if not text.strip():
        return DEFAULT_SEEDS
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Export node-level mechanism checks.")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--base_out", type=Path, default=DEFAULT_BASE_OUT)
    p.add_argument("--fig_dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    all_rows: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"Exporting node mechanism rows for seed {seed}...")
        all_rows.extend(node_rows_for_seed(args.base_out, args.data_dir, seed))

    node_df = assign_bins(pd.DataFrame(all_rows))
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    node_path = args.fig_dir / "fig_mechanism_node_exposure_data.csv"
    node_df.to_csv(node_path, index=False)

    main_summary = summarize_bins(node_df, "delta_graph_minus_feature")
    main_summary.to_csv(args.fig_dir / "fig_mechanism_graph_minus_feature_bins.csv", index=False)
    plot_mechanism(
        main_summary,
        args.fig_dir,
        "fig_mechanism_graph_minus_feature",
        "Delta exposure",
    )

    channel_summary = summarize_bins(node_df, "delta_penalty_channel")
    channel_summary.to_csv(args.fig_dir / "fig_mechanism_penalty_channel_bins.csv", index=False)
    plot_mechanism(
        channel_summary,
        args.fig_dir,
        "fig_mechanism_penalty_channel",
        "Delta exposure",
    )

    print(f"Wrote node-level data: {node_path}")
    print(f"Wrote figures to: {args.fig_dir}")


if __name__ == "__main__":
    main()
