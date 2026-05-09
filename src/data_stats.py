"""Стадия DVC: статистика по сырому датасету (для прозрачности изменений данных).

Считает по `data/raw/flight_delays_ru_synthetic_2023_2025.csv` сводную статистику
и сохраняет в `reports/metrics/data_stats.json`. Эта метрика отслеживается DVC,
поэтому каждый запуск `dvc repro` обновляет её, а `dvc metrics show` отображает.

Зачем это нужно: показать в DVC, что данные действительно меняются между итерациями.
git history файла reports/metrics/data_stats.json + scripts/show_data_evolution.py
дают полную картину эволюции датасета.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_params, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data_stats")


def file_sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    params = load_params()
    raw_path = resolve_path(params["paths"]["raw_csv"])
    metrics_dir = resolve_path(params["paths"]["metrics_dir"])

    log.info("Читаю: %s", raw_path)
    df = pd.read_csv(raw_path, encoding="utf-8-sig")

    file_size = raw_path.stat().st_size
    sha = file_sha256_short(raw_path)

    positive_count = int(df["is_significant_delay"].sum())
    positive_share = float(df["is_significant_delay"].mean())
    reason_counts = df["delay_reason"].value_counts().to_dict()
    reason_counts = {str(k): int(v) for k, v in reason_counts.items()}

    delay_only = df.loc[df["is_significant_delay"] == 1, "delay_minutes"]

    stats = {
        "file": {
            "path": str(raw_path.relative_to(resolve_path("."))),
            "size_bytes": file_size,
            "sha256_16": sha,
        },
        "rows_total": len(df),
        "duplicate_flight_id": int(df["flight_id"].duplicated().sum()),
        "positive_class": {
            "count": positive_count,
            "share": round(positive_share, 6),
        },
        "delay_reason_counts": reason_counts,
        "delay_minutes_when_significant": {
            "mean": round(float(delay_only.mean()), 2) if len(delay_only) else None,
            "median": float(delay_only.median()) if len(delay_only) else None,
            "p95": float(delay_only.quantile(0.95)) if len(delay_only) else None,
            "max": int(delay_only.max()) if len(delay_only) else None,
        },
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "data_stats.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log.info(
        "rows=%d positive=%d (%.4f) sha=%s size=%.1fMB",
        stats["rows_total"],
        positive_count,
        positive_share,
        sha,
        file_size / 1e6,
    )
    log.info("Сохранено: %s", out_path)


if __name__ == "__main__":
    main()
