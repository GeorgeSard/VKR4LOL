"""Формирование итогового набора признаков для обучения."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import load_params, resolve_path
from src.preprocessing import (
    DROP_COLS,
    FEATURE_COLS,
    RAW_TARGET_NUMERIC,
    TARGET_DELAY,
    TARGET_REASON,
    add_derived_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("make_features")


def main() -> None:
    params = load_params()
    cleaned_path = resolve_path(params["paths"]["cleaned_csv"])
    features_path = resolve_path(params["paths"]["features_csv"])

    log.info("Читаю очищенный CSV: %s", cleaned_path)
    df = pd.read_csv(cleaned_path)
    log.info("Загружено строк: %d, колонок: %d", len(df), df.shape[1])

    df = add_derived_features(df)
    log.info("Добавлено производных признаков (флаги погоды/нагрузки/каскада)")

    keep_cols = FEATURE_COLS + [TARGET_DELAY, TARGET_REASON, RAW_TARGET_NUMERIC]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В очищенном датасете не хватает колонок: {missing}")

    df_features = df[keep_cols].copy()

    dropped = [c for c in DROP_COLS if c in df.columns]
    log.info("Отброшено служебных колонок: %d (%s)", len(dropped), ", ".join(dropped))

    features_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(features_path, index=False)
    log.info(
        "Сохранено: %s (%d строк, %d колонок)",
        features_path,
        len(df_features),
        df_features.shape[1],
    )


if __name__ == "__main__":
    main()
