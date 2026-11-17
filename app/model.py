from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float


def build_model() -> Pipeline:
    data = load_iris()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    model.fit(data.data, data.target)
    return model


MODEL = build_model()


def predict_batch(instances: list[list[float]]) -> tuple[list[Prediction], float]:
    values = np.asarray(instances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Each instance must contain exactly four numeric features")

    started = perf_counter()
    probabilities = MODEL.predict_proba(values)
    labels = np.argmax(probabilities, axis=1)
    elapsed_ms = (perf_counter() - started) * 1000
    predictions = [
        Prediction(label=int(label), confidence=float(probabilities[index, label]))
        for index, label in enumerate(labels)
    ]
    return predictions, elapsed_ms
