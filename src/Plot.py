#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figures for the risk-signal placement version of Contract Learning.

The current paper claim is signal placement, not universal graph-score dominance.
The figures therefore prioritize:
    1. Graph-penalty contract vs feature-only contract placement effects.
    2. Graph-penalty contract vs randomized-score contract boundary.
    3. A clean CVaR/crowding frontier with the randomized-score control included.

Example:
    python src/Plot.py --results_dir results/frozen500 --out_dir figures/frozen500_rebuilt
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.4,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.22,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "mathtext.fontset": "dejavusans",
    })


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_results_root(results_dir: str | Path) -> Path:
    root = Path(results_dir)
    candidates = [
        root,
        root / "results" / "frozen500",
        root / "outputs_randomsignal_10seed",
        root / "outputs_randomsignal_frozen500",
        root / "outputs_nopenalty_multiseed",
        root / "outputs_nopenalty_full",
        root / "outputs_graphsignal_full",
        root / "outputs_graphsignal_multiseed",
        root / "outputs_graphstrong_multiseed",
        root / "outputs_plus_multiseed",
    ]
    for c in candidates:
        if (c / "multi_seed_industry_main_results_all_summary.csv").exists():
            return c
        if (c / "tables" / "industry_main_results_all.csv").exists():
            return c
    raise FileNotFoundError(
        "Could not locate industry results under --results_dir. Expected either "
        "multi_seed_industry_main_results_all_summary.csv or tables/industry_main_results_all.csv."
    )


def short_method(name: str) -> str:
    mapping = {
        "FixedSalary": "FixedSalary",
        "LinearPnL": "Linear",
        "EqualRPE": "Equal-RPE",
        "GLS_RPE": "GLS-RPE",
        "EqualRPE_TailCVaRPenalty": "Equal-RPE+Tail",
        "GLS_RPE_TailCVaRPenalty": "GLS+Tail",
        "FullFixed_GLSBenchmark_GraphTailCVaR": "GLS+Tail",
        "RawNodeMLP_GLS_NoGraph_HardEffort": "RawNode",
        "GraphSignal-ICL-NoPenalty": "GS-NoPenalty",
        "GraphSignal-ICL": "GraphSignal-ICL",
        "RandomSignal-ICL": "RandomSignal-ICL",
        "GraphScoreMLP_GLS_TailCVaR_HardEffort": "GraphSignal-ICL",
        "FullGNN_TailMessage_GLSBenchmark_HardEffort": "FullGNN",
        "RandomGraphGNN_GLSBenchmark_HardEffort": "RandomGNN",
        "ETF_RawNodeMLP_GLS_NoGraph_HardEffort": "ETF-RawNode",
        "ETF_GraphSignal-ICL-NoPenalty": "ETF-GraphSignal-NoPenalty",
        "ETF_GraphSignal-ICL": "ETF-GraphSignal-ICL",
        "ETF_RandomSignal-ICL": "ETF-RandomSignal",
        "ETF_FullGNN_TailMessage_GLSBenchmark_HardEffort": "ETF-FullGNN",
        "ETF_RandomGraphGNN_GLSBenchmark_HardEffort": "ETF-RandomGNN",
    }
    return mapping.get(str(name).strip(), str(name).strip())


PAPER_METHOD_LABELS = {
    "RawNode": "No-graph neural baseline",
    "GS-NoPenalty": "Feature-only contract",
    "GraphSignal-ICL": "Graph-penalty contract",
    "RandomSignal-ICL": "Randomized-score contract",
    "FullGNN": "Message-passing graph learner",
    "RandomGNN": "Random-graph learner",
    "FixedSalary": "Fixed-salary contract",
    "Linear": "Linear output-sharing contract",
    "Equal-RPE": "Equal-weight RPE contract",
    "GLS-RPE": "GLS peer-benchmark contract",
    "ETF-RawNode": "ETF raw node-feature learner",
    "ETF-GraphSignal-NoPenalty": "ETF feature-only contract",
    "ETF-GraphSignal-ICL": "ETF graph-penalty contract",
    "ETF-RandomSignal": "ETF randomized-score contract",
    "ETF-FullGNN": "ETF message-passing graph learner",
    "ETF-RandomGNN": "ETF random-graph learner",
}


def paper_method_label(name: str) -> str:
    return PAPER_METHOD_LABELS.get(str(name).strip(), str(name).strip())


def metric_col(df: pd.DataFrame, base: str) -> str:
    if f"{base}_mean" in df.columns:
        return f"{base}_mean"
    if base in df.columns:
        return base
    raise KeyError(f"Missing metric {base}; available columns: {list(df.columns)}")


def optional_std_col(df: pd.DataFrame, base: str) -> Optional[str]:
    if f"{base}_std" in df.columns:
        return f"{base}_std"
    return None


METRIC_LABELS = {
    "CVaR95_loss": "CVaR95",
    "CVaR99_loss": "CVaR99",
    "crowding": "Crowding",
    "co_crash_freq": "Co-crash",
    "principal_CE": "CE",
    "policy_objective": "Policy obj.",
}


MAIN_RISK_METRICS = ["CVaR95_loss", "CVaR99_loss", "crowding", "co_crash_freq"]


def read_required_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return pd.read_csv(path)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def signed_errorbars(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df["delta"].astype(float).to_numpy()
    lo = df["ci_low"].astype(float).to_numpy()
    hi = df["ci_high"].astype(float).to_numpy()
    return np.vstack([np.maximum(x - lo, 0.0), np.maximum(hi - x, 0.0)]), x


def significance_marker(row: pd.Series) -> str:
    if "bh_reject_05" in row.index and str(row["bh_reject_05"]).lower() in {"true", "1"}:
        return "*"
    return ""


def make_placement_effect_plot(results_root: Path, out_dir: Path) -> Path:
    """Main reviewer-facing figure: explicit penalty placement effects."""
    paired = read_required_csv(results_root / "multi_seed_paired_significance.csv", "paired significance table")
    sub = paired[
        (paired["comparison_type"] == "placement_vs_feature_only")
        & (paired["metric"].isin(MAIN_RISK_METRICS))
    ].copy()
    if sub.empty:
        raise ValueError("No placement_vs_feature_only rows found for main risk metrics.")
    sub["metric_label"] = sub["metric"].map(METRIC_LABELS)
    sub["dataset_label"] = sub["dataset"].map({"industry": "Industry", "etf": "ETF"}).fillna(sub["dataset"])
    sub["metric_order"] = sub["metric"].apply(lambda m: MAIN_RISK_METRICS.index(m))
    sub["dataset_order"] = sub["dataset"].map({"industry": 0, "etf": 1}).fillna(2)
    sub = sub.sort_values(["dataset_order", "metric_order"]).reset_index(drop=True)
    sub.to_csv(out_dir / "fig1_placement_effects_plot_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)
    colors = {"Industry": "#f97316", "ETF": "#2563eb"}
    for ax, (dataset, g) in zip(axes, sub.groupby("dataset_label", sort=False)):
        g = g.sort_values("metric_order").reset_index(drop=True)
        y = np.arange(len(g))
        xerr, x = signed_errorbars(g)
        ax.barh(y, x, color=colors.get(dataset, "#6b7280"), alpha=0.86, height=0.62)
        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#111827", elinewidth=0.85, capsize=2.5)
        ax.axvline(0.0, color="#111827", linewidth=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(g["metric_label"])
        ax.invert_yaxis()
        ax.set_title(dataset, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.22)
        for yi, row in enumerate(g.itertuples(index=False)):
            marker = significance_marker(pd.Series(row._asdict()))
            if marker:
                ax.text(float(row.delta), yi, "  *", va="center", ha="left" if float(row.delta) >= 0 else "right", fontsize=10)

    fig.supxlabel("Graph-penalty contract minus feature-only contract (negative is lower risk)", y=0.02)
    fig.suptitle("Explicit Risk-Penalty Placement Effects", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=1.8)
    out_base = out_dir / "fig1_placement_effects_v4"
    save_figure(fig, out_base)
    plt.close(fig)
    return out_base


def make_random_signal_boundary_plot(results_root: Path, out_dir: Path) -> Path:
    """Boundary figure: learned graph-risk score versus randomized same-architecture score."""
    paired = read_required_csv(results_root / "multi_seed_paired_significance.csv", "paired significance table")
    sub = paired[
        (paired["comparison_type"] == "learned_signal_vs_random_signal")
        & (paired["metric"].isin(MAIN_RISK_METRICS))
    ].copy()
    if sub.empty:
        raise ValueError("No learned_signal_vs_random_signal rows found.")
    sub["metric_label"] = sub["metric"].map(METRIC_LABELS)
    sub["dataset_label"] = sub["dataset"].map({"industry": "Industry", "etf": "ETF"}).fillna(sub["dataset"])
    sub["metric_order"] = sub["metric"].apply(lambda m: MAIN_RISK_METRICS.index(m))
    sub["dataset_order"] = sub["dataset"].map({"industry": 0, "etf": 1}).fillna(2)
    sub = sub.sort_values(["dataset_order", "metric_order"]).reset_index(drop=True)
    sub.to_csv(out_dir / "fig2_random_signal_boundary_plot_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True)
    colors = {"supports_treatment": "#16a34a", "supports_baseline": "#9ca3af", "mixed_or_not_significant": "#f59e0b"}
    for ax, (dataset, g) in zip(axes, sub.groupby("dataset_label", sort=False)):
        g = g.sort_values("metric_order").reset_index(drop=True)
        y = np.arange(len(g))
        xerr, x = signed_errorbars(g)
        bar_colors = [colors.get(str(s), "#9ca3af") for s in g["claim_support"]]
        ax.barh(y, x, color=bar_colors, alpha=0.88, height=0.62)
        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#111827", elinewidth=0.85, capsize=2.5)
        ax.axvline(0.0, color="#111827", linewidth=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(g["metric_label"])
        ax.invert_yaxis()
        ax.set_title(dataset, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.22)

    fig.supxlabel("Graph-penalty contract minus randomized-score contract (negative is better for risk)", y=0.02)
    fig.suptitle("Learned Signal Versus Same-Architecture Random Signal", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=1.8)
    out_base = out_dir / "fig2_random_signal_boundary_v4"
    save_figure(fig, out_base)
    plt.close(fig)
    return out_base


def read_industry_results(results_root: Path) -> pd.DataFrame:
    main_path = first_existing([
        results_root / "multi_seed_industry_main_results_all_summary.csv",
        results_root / "tables" / "industry_main_results_all.csv",
    ])
    if main_path is None:
        raise FileNotFoundError("Industry main results table not found.")

    main = pd.read_csv(main_path)

    # Append methods from neural summary if they are absent in the main table.
    neural_path = first_existing([
        results_root / "multi_seed_industry_neural_results_summary.csv",
        results_root / "tables" / "industry_neural_results.csv",
    ])
    if neural_path is not None and "method" in main.columns:
        neural = pd.read_csv(neural_path)
        missing = set(neural["method"].astype(str)) - set(main["method"].astype(str))
        if missing:
            main = pd.concat([main, neural[neural["method"].astype(str).isin(missing)]], ignore_index=True)

    return main


def make_frontier_plot(
    results_root: Path,
    out_dir: Path,
    include_nopenalty: bool,
    single_column: bool = False,
    show_error_bars: bool = False,
    force_nopenalty: bool = False,
) -> Optional[Path]:
    # CIKM short version: Figure 1 is a four-method placement frontier.
    # If ETF summaries are available, draw Industry and ETF as two panels.
    keep = ["RawNode"]
    if include_nopenalty:
        keep.append("GS-NoPenalty")
    keep += ["GraphSignal-ICL", "RandomSignal-ICL"]

    styles = {
        "RawNode": dict(marker="^", s=105, color="#4E79A7", edgecolor="#111827", label=paper_method_label("RawNode")),
        "GS-NoPenalty": dict(marker="v", s=115, color="#F2B84B", edgecolor="#111827", label=paper_method_label("GS-NoPenalty")),
        "GraphSignal-ICL": dict(marker="*", s=210, color="#F47C20", edgecolor="#111827", label=paper_method_label("GraphSignal-ICL")),
        "RandomSignal-ICL": dict(marker="X", s=125, color="#8FA1BD", edgecolor="#111827", label=paper_method_label("RandomSignal-ICL")),
    }

    def read_etf_results() -> Optional[pd.DataFrame]:
        main_path = first_existing([
            results_root / "multi_seed_etf_main_results_all_summary.csv",
            results_root / "tables" / "etf_main_results_all.csv",
        ])
        if main_path is None:
            return None
        main = pd.read_csv(main_path)
        neural_path = first_existing([
            results_root / "multi_seed_etf_neural_results_summary.csv",
            results_root / "tables" / "etf_neural_results.csv",
        ])
        if neural_path is not None and "method" in main.columns:
            neural = pd.read_csv(neural_path)
            missing = set(neural["method"].astype(str)) - set(main["method"].astype(str))
            if missing:
                main = pd.concat([main, neural[neural["method"].astype(str).isin(missing)]], ignore_index=True)
        return main

    def prepare_panel(raw: pd.DataFrame, dataset_label: str, etf: bool = False) -> pd.DataFrame:
        raw = raw.copy()
        raw["short_method"] = raw["method"].map(short_method)
        if etf:
            raw["short_method"] = raw["short_method"].replace({
                "ETF-RawNode": "RawNode",
                "ETF-GraphSignal-NoPenalty": "GS-NoPenalty",
                "ETF-GraphSignal-ICL": "GraphSignal-ICL",
                "ETF-RandomSignal": "RandomSignal-ICL",
            })

        if include_nopenalty and "GS-NoPenalty" not in set(raw["short_method"]) and not force_nopenalty:
            raise ValueError(f"No feature-only contract found for {dataset_label} frontier panel.")

        crowd_col = metric_col(raw, "crowding")
        cvar_col = metric_col(raw, "CVaR95_loss")
        crowd_std = optional_std_col(raw, "crowding")
        cvar_std = optional_std_col(raw, "CVaR95_loss")

        cols = ["short_method", crowd_col, cvar_col]
        for c in [crowd_std, cvar_std]:
            if c and c not in cols:
                cols.append(c)

        df = raw[raw["short_method"].isin(keep)][cols].copy()
        if df.empty:
            raise ValueError(f"No requested methods found in {dataset_label} results table.")

        df = df.groupby("short_method", as_index=False).mean()
        df["order"] = df["short_method"].apply(lambda x: keep.index(x))
        df = df.sort_values("order")
        df["display_method"] = df["short_method"].map(paper_method_label)
        df["dataset"] = dataset_label
        df = df.rename(columns={
            crowd_col: "crowding_mean",
            cvar_col: "CVaR95_loss_mean",
        })
        if crowd_std:
            df = df.rename(columns={crowd_std: "crowding_std"})
        else:
            df["crowding_std"] = np.nan
        if cvar_std:
            df = df.rename(columns={cvar_std: "CVaR95_loss_std"})
        else:
            df["CVaR95_loss_std"] = np.nan
        return df[[
            "dataset", "short_method", "display_method", "crowding_mean",
            "CVaR95_loss_mean", "crowding_std", "CVaR95_loss_std", "order",
        ]]

    try:
        industry_df = prepare_panel(read_industry_results(results_root), "Industry")
    except ValueError as exc:
        print(exc)
        return None

    panels = [industry_df]
    if not single_column:
        etf_raw = read_etf_results()
        if etf_raw is not None:
            try:
                panels.append(prepare_panel(etf_raw, "ETF", etf=True))
            except ValueError as exc:
                print(f"Skipping ETF frontier panel: {exc}")

    df_all = pd.concat(panels, ignore_index=True)
    stem = "fig1_risk_frontier_core" if include_nopenalty else "fig1_risk_frontier_core_main"
    df_all.to_csv(out_dir / f"{stem}_plot_data.csv", index=False)

    if single_column or len(panels) == 1:
        figsize = (3.55, 3.18) if single_column else (5.0, 3.45)
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
        legend_kwargs = dict(loc="best", frameon=False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25))
        legend_kwargs = dict(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=4,
            frameon=False,
            handletextpad=0.35,
            columnspacing=0.85,
        )

    for panel_index, (ax, panel_df) in enumerate(zip(axes, panels)):
        dataset_label = str(panel_df["dataset"].iloc[0])
        for _, row in panel_df.iterrows():
            m = row["short_method"]
            st = styles[m]
            x = float(row["crowding_mean"])
            y = float(row["CVaR95_loss_mean"])

            if show_error_bars and np.isfinite(row["crowding_std"]) and np.isfinite(row["CVaR95_loss_std"]):
                ax.errorbar(
                    x, y,
                    xerr=float(row["crowding_std"]),
                    yerr=float(row["CVaR95_loss_std"]),
                    fmt="none",
                    ecolor="#9ca3af",
                    elinewidth=0.65,
                    capsize=1.8,
                    alpha=0.35,
                    zorder=1,
                )

            ax.scatter(
                x, y,
                marker=st["marker"],
                s=st["s"],
                color=st["color"],
                edgecolor=st["edgecolor"],
                linewidth=1.15,
                label=st["label"],
                zorder=4 if m == "GraphSignal-ICL" else 3,
            )

        if include_nopenalty:
            full = panel_df[panel_df["short_method"] == "GraphSignal-ICL"]
            no = panel_df[panel_df["short_method"] == "GS-NoPenalty"]
            if not full.empty and not no.empty and not single_column:
                x1, y1 = float(no["crowding_mean"].iloc[0]), float(no["CVaR95_loss_mean"].iloc[0])
                x2, y2 = float(full["crowding_mean"].iloc[0]), float(full["CVaR95_loss_mean"].iloc[0])
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.55, color="#9A4E17", shrinkA=6, shrinkB=7),
                    zorder=2,
                )

        xs = panel_df["crowding_mean"].astype(float).to_numpy()
        ys = panel_df["CVaR95_loss_mean"].astype(float).to_numpy()
        xpad = max((xs.max() - xs.min()) * 0.18, 1e-4)
        ypad = max((ys.max() - ys.min()) * 0.22, 1e-4)
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
        ax.set_ylim(ys.min() - ypad, ys.max() + ypad)
        panel_title = f"({chr(ord('a') + panel_index)}) {dataset_label}" if len(panels) > 1 else dataset_label
        ax.set_title(panel_title)
        ax.set_xlabel("Crowding")
        if ax is axes[0]:
            ax.set_ylabel("CVaR95 loss")
        ax.grid(True)
        ax.text(
            0.035, 0.94, "lower-left is better",
            transform=ax.transAxes,
            fontsize=8.2,
            color="#374151",
            va="top",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#CBD5E1", alpha=0.88),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if len(panels) == 1:
        axes[0].legend(**legend_kwargs)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.legend(handles, labels, **legend_kwargs)
        fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.7)

    out_base = out_dir / stem
    save_figure(fig, out_base)
    plt.close(fig)
    return out_base
def load_synthetic_sweeps(results_root: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(results_root / "seed_*" / "tables" / "synthetic_mechanism_sweeps.csv")))

    if not paths:
        fallbacks = [
            results_root / "tables" / "synthetic_mechanism_sweeps.csv",
            results_root / "synthetic_mechanism_sweeps.csv",
        ]
        paths = [str(p) for p in fallbacks if p.exists()]

    if not paths:
        raise FileNotFoundError("synthetic_mechanism_sweeps.csv not found.")

    frames = []
    for p in paths:
        pth = Path(p)
        seed = 0
        for part in pth.parts:
            if part.startswith("seed_"):
                try:
                    seed = int(part.replace("seed_", ""))
                except Exception:
                    seed = 0
        tmp = pd.read_csv(p)
        tmp["seed"] = seed
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def relative_sweep(df: pd.DataFrame, sweep_name: str, xcol: str) -> pd.DataFrame:
    sub = df[df["sweep"].astype(str).str.lower() == sweep_name].copy()
    if sub.empty:
        raise ValueError(f"Missing sweep={sweep_name}")

    rows = []
    for seed, g in sub.groupby("seed"):
        g = g.sort_values(xcol)
        baseline = float(g["crowding"].iloc[0])
        baseline = baseline if abs(baseline) > 1e-12 else 1.0
        g = g.copy()
        g["crowding_change_pct"] = (g["crowding"] / baseline - 1.0) * 100.0
        rows.append(g[[xcol, "crowding_change_pct", "seed"]])

    rel = pd.concat(rows, ignore_index=True)
    agg = rel.groupby(xcol, as_index=False).agg(
        mean_change=("crowding_change_pct", "mean"),
        std_change=("crowding_change_pct", "std"),
        n=("crowding_change_pct", "count"),
    )
    agg["std_change"] = agg["std_change"].fillna(0.0)
    return agg


def make_figure3(results_root: Path, out_dir: Path, show_band: bool = False) -> Path:
    df = load_synthetic_sweeps(results_root)
    eta = relative_sweep(df, "eta", "eta")
    lam = relative_sweep(df, "lambda", "lambda")

    eta.to_csv(out_dir / "fig3_eta_plot_data.csv", index=False)
    lam.to_csv(out_dir / "fig3_lambda_plot_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), sharey=False)
    panels = [
        (axes[0], eta, "eta", r"RPE intensity $\eta$", r"(a) RPE intensity"),
        (axes[1], lam, "lambda", r"Systemic penalty $\lambda$", r"(b) Systemic penalty"),
    ]

    for ax, dat, xcol, xlabel, title in panels:
        x = dat[xcol].values
        y = dat["mean_change"].values
        ax.plot(x, y, marker="o", markersize=4.0, linewidth=1.55, color="#1f77b4")
        if show_band:
            sd = dat["std_change"].values
            ax.fill_between(x, y - sd, y + sd, color="#1f77b4", alpha=0.07, linewidth=0)
        ax.axhline(0.0, color="#6b7280", linewidth=0.75, linestyle="--", alpha=0.65)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel("Crowding change vs baseline (%)")

    for ax, dat in [(axes[0], eta), (axes[1], lam)]:
        ymin = float(dat["mean_change"].min())
        ymax = float(dat["mean_change"].max())
        pad = max((ymax - ymin) * 0.25, 0.05)
        ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout(w_pad=2.0)
    out_base = out_dir / "fig4_synthetic_sweeps_v4"
    save_figure(fig, out_base)
    plt.close(fig)
    return out_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate risk-signal placement paper figures.")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--single_column_fig2", action="store_true")
    parser.add_argument("--show_fig2_error_bars", action="store_true")
    parser.add_argument("--show_fig3_band", action="store_true")
    parser.add_argument("--skip_diagnostic_figures", action="store_true", help="Skip placement and randomized-score diagnostic figures.")
    parser.add_argument("--only_frontier", action="store_true", help="Only generate the clean frontier figure.")
    parser.add_argument("--force_nopenalty_figure", action="store_true")
    args = parser.parse_args()

    set_paper_style()
    results_root = find_results_root(args.results_dir)
    out_dir = ensure_dir(args.out_dir or (results_root / "paper_figures_randomsignal"))

    generated = []
    if not args.skip_diagnostic_figures and not args.only_frontier:
        generated.append(make_placement_effect_plot(results_root, out_dir))
        generated.append(make_random_signal_boundary_plot(results_root, out_dir))

    frontier = make_frontier_plot(
        results_root,
        out_dir,
        include_nopenalty=True,
        single_column=args.single_column_fig2,
        show_error_bars=args.show_fig2_error_bars,
        force_nopenalty=args.force_nopenalty_figure,
    )
    if frontier is not None:
        generated.append(frontier)

    if not args.only_frontier:
        try:
            generated.append(make_figure3(results_root, out_dir, show_band=args.show_fig3_band))
        except FileNotFoundError as exc:
            print(f"Skipping synthetic-sweep figure: {exc}")

    print("Generated:")
    for stem in generated:
        print(f"  {stem}.png / .pdf / .svg")
    print(f"  Plot-data CSVs saved to: {out_dir}")


if __name__ == "__main__":
    main()
