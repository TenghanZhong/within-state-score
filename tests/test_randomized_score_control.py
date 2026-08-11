import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import CON2  # noqa: E402


class RandomizedScoreControlTest(unittest.TestCase):
    def test_numpy_penalty_uses_the_stored_permutation(self):
        tail_score = np.array(
            [[0.7, 0.9, 1.1, 1.3], [0.6, 0.8, 1.2, 1.4]],
            dtype=float,
        )
        random_perm = np.array(
            [[2, 0, 3, 1], [1, 3, 0, 2]],
            dtype=np.int64,
        )
        es = SimpleNamespace(
            m_tail_cvar=tail_score,
            random_perm=random_perm,
        )

        got = CON2.get_penalty_score(es, "random")

        np.testing.assert_allclose(
            got,
            np.take_along_axis(tail_score, random_perm, axis=1),
        )

    @unittest.skipUnless(CON2.TORCH_AVAILABLE, "PyTorch unavailable")
    def test_only_standardized_tail_score_is_permuted(self):
        torch = CON2.torch
        features = torch.arange(2 * 4 * 9, dtype=torch.float32).reshape(2, 4, 9)
        random_perm = torch.tensor(
            [[2, 0, 3, 1], [1, 3, 0, 2]],
            dtype=torch.long,
        )
        data = {
            "features": features,
            "random_perm": random_perm,
        }

        randomized = CON2.select_torch_features(data, "random_scores")
        expected = features.clone()
        expected[..., 7] = torch.gather(features[..., 7], 1, random_perm)

        torch.testing.assert_close(randomized, expected)
        unchanged_columns = [0, 1, 2, 3, 4, 5, 6, 8]
        torch.testing.assert_close(
            randomized[..., unchanged_columns],
            features[..., unchanged_columns],
        )
        for state in range(features.shape[0]):
            torch.testing.assert_close(
                torch.sort(randomized[state, :, 7]).values,
                torch.sort(features[state, :, 7]).values,
            )

    @unittest.skipUnless(CON2.TORCH_AVAILABLE, "PyTorch unavailable")
    def test_feature_and_penalty_use_the_same_permutation(self):
        torch = CON2.torch
        features = torch.zeros((2, 4, 9), dtype=torch.float32)
        features[..., 7] = torch.tensor(
            [[-1.5, -0.5, 0.5, 1.5], [-1.2, -0.4, 0.4, 1.2]],
            dtype=torch.float32,
        )
        raw_tail_score = torch.tensor(
            [[0.7, 0.9, 1.1, 1.3], [0.6, 0.8, 1.2, 1.4]],
            dtype=torch.float32,
        )
        random_perm = torch.tensor(
            [[2, 0, 3, 1], [1, 3, 0, 2]],
            dtype=torch.long,
        )
        data = {
            "features": features,
            "m_tail_cvar": raw_tail_score,
            "random_perm": random_perm,
        }

        randomized_features = CON2.select_torch_features(data, "random_scores")
        randomized_penalty = CON2.choose_torch_m(data, "random")

        torch.testing.assert_close(
            randomized_features[..., 7],
            torch.gather(features[..., 7], 1, random_perm),
        )
        torch.testing.assert_close(
            randomized_penalty,
            torch.gather(raw_tail_score, 1, random_perm),
        )

    def test_frozen_evaluation_reuses_the_source_permutation(self):
        dates = pd.DatetimeIndex(["2024-01-01", "2024-02-01"])
        source = SimpleNamespace(
            name="source",
            dates=dates,
            symbols=["A", "B", "C"],
            N=3,
            B=2,
            random_perm=np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64),
            W_random=np.arange(18, dtype=float).reshape(2, 3, 3),
        )
        target = SimpleNamespace(
            name="target",
            dates=dates,
            symbols=["A", "B", "C"],
            N=3,
            B=2,
            m_tail_cvar=np.array(
                [[0.8, 1.0, 1.2], [0.7, 1.1, 1.2]],
                dtype=float,
            ),
            m_random=np.zeros((2, 3), dtype=float),
            random_perm=np.zeros((2, 3), dtype=np.int64),
            W_random=np.zeros((2, 3, 3), dtype=float),
        )

        result = CON2.copy_random_controls(source, target)

        np.testing.assert_array_equal(result.random_perm, source.random_perm)
        np.testing.assert_allclose(
            result.m_random,
            np.take_along_axis(
                target.m_tail_cvar,
                source.random_perm,
                axis=1,
            ),
        )
        np.testing.assert_allclose(result.W_random, source.W_random)
        self.assertFalse(np.shares_memory(result.random_perm, source.random_perm))
        self.assertFalse(np.shares_memory(result.W_random, source.W_random))

    def test_frozen_evaluation_rejects_symbol_reordering(self):
        dates = pd.DatetimeIndex(["2024-01-01"])
        source = SimpleNamespace(
            name="source",
            dates=dates,
            symbols=["A", "B"],
            N=2,
            B=1,
            random_perm=np.array([[1, 0]], dtype=np.int64),
            W_random=np.zeros((1, 2, 2), dtype=float),
        )
        target = SimpleNamespace(
            name="target",
            dates=dates,
            symbols=["B", "A"],
            N=2,
            B=1,
            m_tail_cvar=np.array([[0.8, 1.2]], dtype=float),
            m_random=np.zeros((1, 2), dtype=float),
            random_perm=np.zeros((1, 2), dtype=np.int64),
            W_random=np.zeros((1, 2, 2), dtype=float),
        )

        with self.assertRaisesRegex(ValueError, "Symbol-order mismatch"):
            CON2.copy_random_controls(source, target)


if __name__ == "__main__":
    unittest.main()
