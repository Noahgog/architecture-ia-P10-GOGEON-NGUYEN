from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import psutil
import requests

from app.brawl_model import BrawlMatch, predict_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark modèle seul et API HTTP")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--batch-sizes", default="1,4,16")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Nombre de requêtes HTTP exécutées simultanément",
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


SAMPLE = BrawlMatch(
    mode="brawlBall",
    map="Sneaky Fields",
    team_1_avg_elo=12.5,
    team_2_avg_elo=11.0,
    team_1_avg_power=11.0,
    team_2_avg_power=11.0,
)


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * value)))
    return ordered[index]


def run_model(batch: list[BrawlMatch], repetitions: int) -> list[float]:
    latencies = []
    for _ in range(repetitions):
        started = time.perf_counter()
        predict_batch(batch)
        latencies.append((time.perf_counter() - started) * 1000)
    return latencies


def request_api(
    url: str,
    batch: list[BrawlMatch],
    timeout: float,
) -> tuple[float, bool]:
    started = time.perf_counter()
    try:
        response = requests.post(
            url,
            json={"instances": [instance.__dict__ for instance in batch]},
            timeout=timeout,
        )
        failed = response.status_code >= 400
    except requests.RequestException:
        failed = True
    return (time.perf_counter() - started) * 1000, failed


def run_api(
    url: str,
    batch: list[BrawlMatch],
    repetitions: int,
    concurrency: int,
    timeout: float,
) -> tuple[list[float], int, float]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(
            executor.map(
                lambda _: request_api(url, batch, timeout),
                range(repetitions),
            )
        )
    wall_time_seconds = time.perf_counter() - started
    latencies = [latency for latency, _ in results]
    errors = sum(failed for _, failed in results)
    return latencies, errors, wall_time_seconds


def main() -> None:
    args = parse_args()
    if args.requests < 1:
        raise ValueError("--requests must be positive")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []

    for batch_size in (int(value) for value in args.batch_sizes.split(",")):
        batch = [SAMPLE] * batch_size
        model_latencies = run_model(batch, args.requests)
        api_latencies, errors, api_wall_time_seconds = run_api(
            args.url,
            batch,
            args.requests,
            args.concurrency,
            args.timeout,
        )
        model_median = statistics.median(model_latencies)
        api_median = statistics.median(api_latencies)
        rows.append(
            {
                "batch_size": batch_size,
                "requests": args.requests,
                "model_median_ms": round(model_median, 4),
                "model_p95_ms": round(percentile(model_latencies, 0.95), 4),
                "api_median_ms": round(api_median, 4),
                "api_p95_ms": round(percentile(api_latencies, 0.95), 4),
                "overhead_median_ms": round(api_median - model_median, 4),
                "throughput_req_s": round(
                    args.requests / api_wall_time_seconds, 4
                ),
                "error_rate": round(errors / args.requests, 4),
                "concurrency": args.concurrency,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
            }
        )

    output = results_dir / f"benchmark_{timestamp}.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "timestamp_utc": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "url": args.url,
        "requests": args.requests,
        "batch_sizes": args.batch_sizes,
        "concurrency": args.concurrency,
    }
    (results_dir / f"benchmark_{timestamp}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Résultats écrits dans {output}")


if __name__ == "__main__":
    main()
