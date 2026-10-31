"""Train and evaluate the constant-score explicit-penalty baseline.

This runner adds the ablation

    m_feature = m_graph,  m_penalty = 1,  lambda active

without changing the existing frozen 500-path result directory.  For each seed
it trains only the constant-score neural contract on the 50-path source setup,
re-evaluates that checkpoint with the requested frozen-evaluation path count,
then combines the new rows with a reference result directory so the standard
paired diagnostics can compare:

    Constant - Feature, Graph - Constant, Graph - Random.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import CON2  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the constant-score penalty ablation.")
    p.add_argument("--data_dir", required=True, help="Raw-data directory described in docs/DATA.md.")
    p.add_argument("--out_dir", required=True, help="Output directory for constant-score ablation results.")
    p.add_argument("--reference_results_dir", default=str(ROOT / "results" / "frozen500"), help="Existing frozen 500-path result directory to combine with.")
    p.add_argument("--seeds", default="7,11,13,17,19,23,29,31,37,41", help="Comma-separated training seeds.")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--mc_paths", type=int, default=50, help="Training bootstrap paths per state.")
    p.add_argument("--frozen_eval_mc_paths", type=int, default=500, help="Held-out evaluation bootstrap paths per state.")
    p.add_argument("--torch_threads", type=int, default=1)
    p.add_argument("--crowding_obj_weight", type=float, default=0.10)
    p.add_argument("--effort_target", type=float, default=0.040)
    p.add_argument("--effort_target_weight", type=float, default=50.0)
    p.add_argument("--graph_propagation_rho", type=float, default=0.25)
    p.add_argument("--hard_effort_z_min", type=float, default=0.80)
    p.add_argument("--hard_effort_z_cap", type=float, default=1.20)
    p.add_argument("--neural_peer_benchmark", default="gls", choices=["gls", "equal"])
    p.add_argument("--skip_etf", action="store_true")
    p.add_argument("--quick", action="store_true", help="Smoke-test mode with reduced epochs, scenario paths, and episode frequency.")
    return p.parse_args()


def make_cfg(args: argparse.Namespace, seed: int, mc_paths: int) -> CON2.ExperimentConfig:
    cfg = CON2.ExperimentConfig(
        quick=bool(args.quick),
        epochs=int(args.epochs),
        random_seed=int(seed),
        mc_paths=int(mc_paths),
        torch_threads=int(args.torch_threads),
        crowding_obj_weight=float(args.crowding_obj_weight),
        effort_target=float(args.effort_target),
        effort_target_weight=float(args.effort_target_weight),
        graph_propagation_rho=float(args.graph_propagation_rho),
        hard_effort_z_min=float(args.hard_effort_z_min),
        hard_effort_z_cap=float(args.hard_effort_z_cap),
        neural_peer_benchmark=str(args.neural_peer_benchmark),
    )
    cfg.apply_quick()
    return cfg


def build_source_splits(
    data: Dict[str, Any],
    cfg: CON2.ExperimentConfig,
    include_etf: bool,
) -> Tuple[CON2.EpisodeSet, CON2.EpisodeSet, CON2.EpisodeSet, Optional[Tuple[CON2.EpisodeSet, CON2.EpisodeSet, CON2.EpisodeSet]]]:
    rng = CON2.np.random.default_rng(cfg.random_seed)
    returns, factor, vix = CON2.prepare_industry_returns(data, cfg)
    industry_es = CON2.build_episode_set("industry49", returns, factor, vix, cfg, cfg.industry_start, cfg.test_end, rng)
    train = industry_es.subset_by_date(cfg.industry_start, cfg.train_end, "industry_train")
    val = industry_es.subset_by_date(cfg.val_start, cfg.val_end, "industry_val")
    test = industry_es.subset_by_date(cfg.test_start, cfg.test_end, "industry_test")
    train, val, test, _ = CON2.standardize_feature_sets(train, val, test)

    etf_splits = None
    if include_etf:
        etf_prepared = CON2.prepare_etf_returns(data, cfg)
        if etf_prepared is not None:
            etf_returns, etf_factor, etf_vix = etf_prepared
            etf_es = CON2.build_episode_set("ETF", etf_returns, etf_factor, etf_vix, cfg, cfg.etf_start, cfg.etf_test_end, rng)
            etf_train = etf_es.subset_by_date(cfg.etf_start, cfg.etf_train_end, "etf_train")
            etf_val = etf_es.subset_by_date(cfg.etf_val_start, cfg.etf_val_end, "etf_val")
            etf_test = etf_es.subset_by_date(cfg.etf_test_start, cfg.etf_test_end, "etf_test")
            etf_splits = CON2.standardize_feature_sets(etf_train, etf_val, etf_test)[:3]
    return train, val, test, etf_splits


def constant_spec(dataset: str, benchmark_kind: str = "gls") -> Dict[str, Any]:
    if dataset == "industry":
        return dict(
            label="GraphSignal-ICL-ConstantPenalty",
            use_graph=False,
            graph_kind="equal",
            benchmark_kind=benchmark_kind,
            penalty_kind="constant",
            feature_mode="graph_scores",
        )
    if dataset == "etf":
        return dict(
            label="ETF_GraphSignal-ICL-ConstantPenalty",
            use_graph=False,
            graph_kind="equal",
            benchmark_kind=benchmark_kind,
            penalty_kind="constant",
            feature_mode="graph_scores",
        )
    raise ValueError(dataset)


def read_reference_table(reference_seed_dir: pathlib.Path, dataset: str, table: str) -> pd.DataFrame:
    path = reference_seed_dir / "tables" / f"{dataset}_{table}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def sanitize_workspace_paths(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    root_win = str(ROOT)
    root_posix = ROOT.as_posix()

    def clean(value: object) -> object:
        if not isinstance(value, str):
            return value
        value = value.replace(root_win + "\\", "")
        value = value.replace(root_win + "/", "")
        value = value.replace(root_posix + "/", "")
        return value

    object_cols = out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        out[col] = out[col].map(clean)
    return out


def write_combined_tables(
    seed_dir: pathlib.Path,
    reference_seed_dir: pathlib.Path,
    dataset: str,
    metrics: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    tables_dir = CON2.ensure_dir(seed_dir / "tables")
    ref_metrics = read_reference_table(reference_seed_dir, dataset, "neural_results")
    combined_metrics = pd.concat([ref_metrics, metrics], ignore_index=True) if not ref_metrics.empty else metrics
    combined_metrics = sanitize_workspace_paths(combined_metrics)
    CON2.write_display_csv(combined_metrics, tables_dir / f"{dataset}_neural_results.csv")
    CON2.write_display_csv(combined_metrics, tables_dir / f"{dataset}_main_results_all.csv")

    ref_details = read_reference_table(reference_seed_dir, dataset, "episode_details_neural")
    for frame in (ref_details, details):
        if "date" in frame.columns:
            frame["date"] = CON2.normalize_date_series(frame["date"])
    combined_details = pd.concat([ref_details, details], ignore_index=True) if not ref_details.empty else details
    CON2.write_display_csv(combined_details, tables_dir / f"{dataset}_episode_details_neural.csv")


def run_seed(args: argparse.Namespace, data: Dict[str, Any], seed: int) -> None:
    out_seed = CON2.ensure_dir(pathlib.Path(args.out_dir).expanduser().resolve() / f"seed_{seed}")
    source_tables = CON2.ensure_dir(out_seed / "source_50path_tables")
    reference_seed_dir = pathlib.Path(args.reference_results_dir).expanduser().resolve() / f"seed_{seed}"
    cfg_source = make_cfg(args, seed, args.mc_paths)
    cfg_eval = dataclasses.replace(cfg_source, mc_paths=int(args.frozen_eval_mc_paths))
    if CON2.TORCH_AVAILABLE and cfg_source.torch_threads and cfg_source.torch_threads > 0:
        try:
            CON2.torch.set_num_threads(int(cfg_source.torch_threads))
            CON2.torch.set_num_interop_threads(1)
        except Exception:
            pass
    CON2.set_seed(seed)

    train, val, test, etf_splits = build_source_splits(data, cfg_source, include_etf=not args.skip_etf)
    print(f"Training industry constant-score baseline for seed {seed}")
    CON2.train_neural_contract(train, val, test, cfg_source, source_tables, **constant_spec("industry", cfg_source.neural_peer_benchmark))

    industry_eval, etf_eval = CON2.build_eval_tests_with_frozen_random_controls(data, cfg_source, cfg_eval)
    print(f"Evaluating industry constant-score baseline for seed {seed} with {cfg_eval.mc_paths} paths")
    metrics, details = CON2.evaluate_saved_neural_contract(industry_eval, cfg_eval, source_tables, **constant_spec("industry", cfg_eval.neural_peer_benchmark))
    metrics["dataset"] = "industry"
    write_combined_tables(out_seed, reference_seed_dir, "industry", pd.DataFrame([metrics]), details)

    if not args.skip_etf and etf_splits is not None and etf_eval is not None:
        etf_train, etf_val, etf_test = etf_splits
        if etf_train.B > 5 and etf_val.B > 2:
            print(f"Training ETF constant-score baseline for seed {seed}")
            CON2.train_neural_contract(etf_train, etf_val, etf_test, cfg_source, source_tables, **constant_spec("etf", cfg_source.neural_peer_benchmark))
            print(f"Evaluating ETF constant-score baseline for seed {seed} with {cfg_eval.mc_paths} paths")
            etf_metrics, etf_details = CON2.evaluate_saved_neural_contract(etf_eval, cfg_eval, source_tables, **constant_spec("etf", cfg_eval.neural_peer_benchmark))
            etf_metrics["dataset"] = "etf"
            write_combined_tables(out_seed, reference_seed_dir, "etf", pd.DataFrame([etf_metrics]), etf_details)


def main() -> None:
    args = parse_args()
    seeds = CON2.parse_seed_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    out_dir = CON2.ensure_dir(pathlib.Path(args.out_dir).expanduser().resolve())
    data = CON2.load_all_data(pathlib.Path(args.data_dir).expanduser().resolve())
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"Constant-score ablation seed {seed}")
        print("=" * 80)
        run_seed(args, data, int(seed))
    CON2.run_artifact_diagnostics(out_dir, seeds)
    print(f"Constant-score ablation diagnostics written under {out_dir}")


if __name__ == "__main__":
    main()
