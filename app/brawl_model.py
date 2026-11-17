from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(
    os.getenv(
        "BRAWL_DATA_PATH",
        str(PROJECT_ROOT / "data/brawlstars_ranked_sample.csv"),
    )
)

CATEGORICAL_FEATURES = [0, 1]
NUMERIC_FEATURES = [2, 3, 4, 5]


@dataclass(frozen=True)
class BrawlMatch:
    mode: str
    map: str
    team_1_avg_elo: float
    team_2_avg_elo: float
    team_1_avg_power: float
    team_2_avg_power: float


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float


def _read_training_rows(path: Path) -> tuple[list[list[object]], list[int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. Run "
            "`python scripts/prepare_brawl_data.py --limit 100000` first."
        )

    rows: list[list[object]] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                [
                    row["mode"],
                    row["map"],
                    float(row["team_1_avg_elo"]),
                    float(row["team_2_avg_elo"]),
                    float(row["team_1_avg_power"]),
                    float(row["team_2_avg_power"]),
                ]
            )
            labels.append(int(row["team_1_win"]))
    if len(set(labels)) < 2:
        raise ValueError("Training data must contain both winning classes")
    return rows, labels


def build_model(path: Path = DATA_PATH) -> Pipeline:
    rows, labels = _read_training_rows(path)
    model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            StandardScaler(),
                            NUMERIC_FEATURES,
                        ),
                        (
                            "categorical",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(rows, labels)
    return model


_MODEL: Pipeline | None = None
_MODEL_LOCK = Lock()


def get_model() -> Pipeline:
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = build_model()
    return _MODEL


def predict_batch(
    instances: list[BrawlMatch],
) -> tuple[list[Prediction], float]:
    started = perf_counter()
    probabilities = get_model().predict_proba(
        [
            [
                instance.mode,
                instance.map,
                instance.team_1_avg_elo,
                instance.team_2_avg_elo,
                instance.team_1_avg_power,
                instance.team_2_avg_power,
            ]
            for instance in instances
        ]
    )
    elapsed_ms = (perf_counter() - started) * 1000
    labels = np.argmax(probabilities, axis=1)
    return [
        Prediction(
            label=int(label),
            confidence=float(probabilities[index, label]),
        )
        for index, label in enumerate(labels)
    ], elapsed_ms
