from __future__ import annotations

from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import predict_batch

app = FastAPI(
    title="P10 IA as a Service",
    version="0.1.0",
    description="Baseline d'inférence Iris pour mesurer l'overhead d'une API.",
)


class PredictionRequest(BaseModel):
    instances: list[
        Annotated[list[float], Field(min_length=4, max_length=4)]
    ] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    label: int
    confidence: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
    model_latency_ms: float


class MetricsResponse(BaseModel):
    prediction_requests_total: int
    prediction_errors_total: int
    model_latency_total_ms: float
    model_latency_average_ms: float


_metrics_lock = Lock()
_prediction_requests_total = 0
_prediction_errors_total = 0
_model_latency_total_ms = 0.0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    with _metrics_lock:
        requests_total = _prediction_requests_total
        errors_total = _prediction_errors_total
        latency_total_ms = _model_latency_total_ms

    successful_requests = requests_total - errors_total
    average_latency_ms = (
        latency_total_ms / successful_requests if successful_requests else 0.0
    )
    return MetricsResponse(
        prediction_requests_total=requests_total,
        prediction_errors_total=errors_total,
        model_latency_total_ms=latency_total_ms,
        model_latency_average_ms=average_latency_ms,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    global _prediction_requests_total, _prediction_errors_total
    global _model_latency_total_ms

    with _metrics_lock:
        _prediction_requests_total += 1

    try:
        predictions, model_latency_ms = predict_batch(request.instances)
    except (TypeError, ValueError) as error:
        with _metrics_lock:
            _prediction_errors_total += 1
        raise HTTPException(status_code=422, detail=str(error)) from error

    with _metrics_lock:
        _model_latency_total_ms += model_latency_ms

    return PredictionResponse(
        predictions=[
            PredictionItem(label=item.label, confidence=item.confidence)
            for item in predictions
        ],
        model_latency_ms=model_latency_ms,
    )
