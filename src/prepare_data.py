"""Очистка сырого датасета: удаление дублей, парсинг дат."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import load_params, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prepare_data")


def main() -> None:
    params = load_params()
    raw_path = resolve_path(params["paths"]["raw_csv"])
    cleaned_path = resolve_path(params["paths"]["cleaned_csv"])
    dedup_key = params["prepare"]["drop_duplicates_on"]

    log.info("Читаю сырой CSV: %s", raw_path)
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    log.info("Загружено строк: %d, колонок: %d", len(df), df.shape[1])

    before = len(df)
    df = df.drop_duplicates(subset=[dedup_key], keep="first").reset_index(drop=True)
    removed = before - len(df)
    log.info("Удалено дублей по %s: %d", dedup_key, removed)

    if "flight_date" in df.columns:
        df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    if "scheduled_departure_local" in df.columns:
        df["scheduled_departure_local"] = pd.to_datetime(
            df["scheduled_departure_local"], errors="coerce"
        )
    if "scheduled_arrival_local" in df.columns:
        df["scheduled_arrival_local"] = pd.to_datetime(
            df["scheduled_arrival_local"], errors="coerce"
        )

    significant = int(df["is_significant_delay"].sum())
    share = significant / len(df) if len(df) else 0.0
    log.info("Значимых задержек: %d (%.2f%%)", significant, share * 100)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_path, index=False)
    log.info("Сохранено: %s (%d строк)", cleaned_path, len(df))


if __name__ == "__main__":
    main()
