from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import predict_batch

app = FastAPI(
    title="P10 IA as a Service",
    version="0.1.0",
    description="Baseline d'inférence Iris pour mesurer l'overhead d'une API.",
)


class PredictionRequest(BaseModel):
    instances: list[list[float]] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    label: int
    confidence: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]
    model_latency_ms: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        predictions, model_latency_ms = predict_batch(request.instances)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictionResponse(
        predictions=[
            PredictionItem(label=item.label, confidence=item.confidence)
            for item in predictions
        ],
        model_latency_ms=model_latency_ms,
    )
