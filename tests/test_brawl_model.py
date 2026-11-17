from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.brawl_model import BrawlMatch, build_model, predict_batch


class BrawlModelTests(unittest.TestCase):
    def test_predict_batch_returns_probability(self) -> None:
        predictions, latency_ms = predict_batch(
            [
                BrawlMatch(
                    mode="brawlBall",
                    map="Sneaky Fields",
                    team_1_avg_elo=12.5,
                    team_2_avg_elo=11.0,
                    team_1_avg_power=11.0,
                    team_2_avg_power=11.0,
                )
            ]
        )

        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0].label, (0, 1))
        self.assertGreaterEqual(predictions[0].confidence, 0.0)
        self.assertLessEqual(predictions[0].confidence, 1.0)
        self.assertGreaterEqual(latency_ms, 0.0)

    def test_missing_training_data_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                build_model(missing_path)


if __name__ == "__main__":
    unittest.main()
