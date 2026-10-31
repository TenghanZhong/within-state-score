"""Check the layout and selected values of the included frozen artifacts."""

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


class ArtifactIntegrityTest(unittest.TestCase):
    def test_required_artifacts_exist(self):
        required = (
            "src/CON2.py",
            "src/Plot.py",
            "src/NodeMechanism.py",
            "results/frozen500/multi_seed_paired_significance.csv",
            "results/frozen500/multi_seed_cluster_episode_bootstrap.csv",
            "results/frozen500/frozen_reevaluation_summary.txt",
            "results/constant_score_500/multi_seed_paired_significance.csv",
            "results/constant_score_500/multi_seed_cluster_episode_bootstrap.csv",
            "figures/frozen500/fig1_risk_frontier_core.pdf",
            "figures/frozen500/fig1_risk_frontier_core_plot_data.csv",
            "figures/mechanism_source/fig_mechanism_graph_minus_feature.pdf",
            "figures/mechanism_source/fig_mechanism_graph_minus_feature_bins.csv",
        )
        for relative_path in required:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

    def test_source_checkpoints_and_run_summaries_exist(self):
        checkpoint_root = ROOT / "checkpoints" / "source_50path"
        self.assertEqual(len(list(checkpoint_root.glob("seed_*/tables/*_model.pt"))), 120)
        for seed in SEEDS:
            with self.subTest(seed=seed):
                seed_dir = checkpoint_root / f"seed_{seed}"
                self.assertEqual(len(list((seed_dir / "tables").glob("*_model.pt"))), 12)
                self.assertTrue((seed_dir / "run_summary.json").is_file())

    def assert_comparison_values(self, relative_path, expected):
        with (ROOT / relative_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        index = {(row["dataset"], row["comparison_type"], row["metric"]): row for row in rows}
        for key, values in expected.items():
            with self.subTest(comparison=key):
                self.assertIn(key, index)
                actual = tuple(round(float(index[key][column]), 4) for column in ("delta", "ci_low", "ci_high"))
                self.assertEqual(actual, tuple(round(value, 4) for value in values))

    def test_frozen_cvar_comparisons_match_included_values(self):
        self.assert_comparison_values(
            "results/frozen500/multi_seed_paired_significance.csv",
            {
                ("industry", "placement_vs_feature_only", "CVaR95_loss"): (-0.0095, -0.0109, -0.0079),
                ("industry", "learned_signal_vs_random_signal", "CVaR95_loss"): (0.0023, 0.0009, 0.0039),
                ("etf", "placement_vs_feature_only", "CVaR95_loss"): (-0.0084, -0.0094, -0.0074),
                ("etf", "learned_signal_vs_random_signal", "CVaR95_loss"): (0.0084, 0.0075, 0.0093),
            },
        )

    def test_constant_score_comparisons_match_included_values(self):
        self.assert_comparison_values(
            "results/constant_score_500/multi_seed_paired_significance.csv",
            {
                ("industry", "constant_penalty_vs_feature_only", "CVaR95_loss"): (-0.0093, -0.0106, -0.0081),
                ("industry", "graph_penalty_vs_constant_penalty", "crowding"): (-0.0079, -0.0109, -0.0051),
                ("etf", "constant_penalty_vs_feature_only", "CVaR95_loss"): (-0.0163, -0.0175, -0.0152),
                ("etf", "graph_penalty_vs_constant_penalty", "crowding"): (-0.0043, -0.0070, -0.0019),
            },
        )


if __name__ == "__main__":
    unittest.main()
