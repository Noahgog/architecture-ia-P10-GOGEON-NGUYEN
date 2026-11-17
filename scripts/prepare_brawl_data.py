from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from datasets import load_dataset


OUTPUT_COLUMNS = [
    "mode",
    "map",
    "team_1_avg_elo",
    "team_2_avg_elo",
    "team_1_avg_power",
    "team_2_avg_power",
    "team_1_win",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prépare un échantillon du dataset Brawl Stars ranked"
    )
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--output", default="data/brawlstars_ranked_sample.csv")
    parser.add_argument("--split", default="season55")
    return parser.parse_args()


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    if args.limit < 2:
        raise ValueError("--limit doit être supérieur ou égal à 2")

    dataset = load_dataset(
        "EliF77/brawlstars-ranked",
        split=args.split,
        streaming=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source in itertools.islice(dataset, args.limit):
        record = str(source["record"])
        if not record.startswith(("T1", "T2")):
            continue

        team_1_elos = [
            float(source[f"t1_b{i}_elo"]) for i in range(3)
        ]
        team_2_elos = [
            float(source[f"t2_b{i}_elo"]) for i in range(3)
        ]
        team_1_powers = [
            float(source[f"t1_b{i}_power"]) for i in range(3)
        ]
        team_2_powers = [
            float(source[f"t2_b{i}_power"]) for i in range(3)
        ]
        rows.append(
            {
                "mode": source["mode"],
                "map": source["map"],
                "team_1_avg_elo": average(team_1_elos),
                "team_2_avg_elo": average(team_2_elos),
                "team_1_avg_power": average(team_1_powers),
                "team_2_avg_power": average(team_2_powers),
                "team_1_win": int(record.startswith("T1")),
            }
        )

    if len({row["team_1_win"] for row in rows}) < 2:
        raise ValueError("L'échantillon préparé ne contient pas les deux classes")

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} lignes écrites dans {output}")


if __name__ == "__main__":
    main()
