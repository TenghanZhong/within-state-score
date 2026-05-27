import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import CON2  # noqa: E402


class ConstantScoreVariantTest(unittest.TestCase):
    def test_numpy_constant_penalty_is_unscored(self):
        m_graph = np.array([[0.7, 1.2, 1.8]], dtype=float)
        es = SimpleNamespace(
            m_marginal=np.full_like(m_graph, 9.0),
            m_tail_cvar=m_graph,
            m_centrality=np.full_like(m_graph, 3.0),
            m_random=np.full_like(m_graph, 5.0),
        )

        got = CON2.get_penalty_score(es, "constant")

        np.testing.assert_allclose(got, np.ones_like(m_graph))

    @unittest.skipUnless(CON2.TORCH_AVAILABLE, "PyTorch unavailable")
    def test_torch_constant_penalty_is_unscored(self):
        torch = CON2.torch
        m_graph = torch.tensor([[0.7, 1.2, 1.8]], dtype=torch.float32)
        data = {
            "m_tail_cvar": m_graph,
            "m_marginal": torch.full_like(m_graph, 9.0),
            "m_centrality": torch.full_like(m_graph, 3.0),
            "m_random": torch.full_like(m_graph, 5.0),
        }

        got = CON2.choose_torch_m(data, "constant")

        torch.testing.assert_close(got, torch.ones_like(m_graph))

    def test_constant_baseline_specs_keep_graph_features(self):
        cfg = CON2.ExperimentConfig()

        industry_specs = CON2.neural_contract_specs(cfg, "industry")
        etf_specs = CON2.neural_contract_specs(cfg, "etf")

        industry = next(s for s in industry_specs if s["label"] == "GraphSignal-ICL-ConstantPenalty")
        etf = next(s for s in etf_specs if s["label"] == "ETF_GraphSignal-ICL-ConstantPenalty")
        self.assertEqual(industry["feature_mode"], "graph_scores")
        self.assertEqual(industry["penalty_kind"], "constant")
        self.assertEqual(etf["feature_mode"], "graph_scores")
        self.assertEqual(etf["penalty_kind"], "constant")


if __name__ == "__main__":
    unittest.main()
