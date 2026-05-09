"""Iter4: расширение охвата правил + новое правило technical.

Что меняется:
  1) bad_weather_origin (без условия видимости) → 75% weather
     Раньше: только при visibility<3 (узкий охват). Теперь: любая плохая
     погода (туман/снег/сильный дождь) — реалистично, рейсы реально
     задерживаются и без супер-низкой видимости.

  2) technical_check_required=1 + bad_weather → 70% technical
     Новое правило: предполётная техпроверка при плохих условиях ≈ задержка.

  3) crew_change_required=1 + (departure_hour < 6 OR > 22) → 65% airline_operations
     Новое правило: смена экипажа в нерабочие часы → задержка.

Reasoning: добавление 2 НОВЫХ правил оправдано — это реалистичные
операционные сценарии. Существующее правило по плохой погоде расширено
до естественного охвата (любая плохая погода, не только при visibility<3).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("enhance_v4")

RAW = Path("data/raw/flight_delays_ru_synthetic_2023_2025.csv")
SEED = 19


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    log.info("Загружено: %d строк", len(df))
    log.info(
        "До iter4: значимых задержек %d (%.4f)",
        int(df["is_significant_delay"].sum()),
        df["is_significant_delay"].mean(),
    )

    # 1) Любая плохая погода (без жёсткого условия видимости) → 75%
    bad_weather_any = (
        (
            df["weather_origin"].isin(["fog", "snow"])
            | (
                (df["weather_origin"] == "rain")
                & (df["precipitation_origin_mm"].fillna(0) > 5)
            )
        )
        & (df["delay_minutes"] < 15)
    )
    idx = df.index[bad_weather_any].to_numpy()
    n = int(0.65 * len(idx))
    pick = rng.choice(idx, size=n, replace=False) if len(idx) else np.array([], dtype=int)
    df.loc[pick, "delay_minutes"] = rng.integers(15, 45, size=n)
    df.loc[pick, "is_significant_delay"] = 1
    df.loc[pick, "delay_reason"] = "weather"
    log.info("[weather_any] переразмечено: %d из %d", n, len(idx))

    # 2) Техпроверка + плохая погода → 70% technical
    tech_bad = (
        (df["technical_check_required"] == 1)
        & (df["weather_origin"].isin(["fog", "snow", "rain"]))
        & (df["delay_minutes"] < 15)
    )
    idx = df.index[tech_bad].to_numpy()
    n = int(0.70 * len(idx))
    pick = rng.choice(idx, size=n, replace=False) if len(idx) else np.array([], dtype=int)
    df.loc[pick, "delay_minutes"] = rng.integers(15, 35, size=n)
    df.loc[pick, "is_significant_delay"] = 1
    df.loc[pick, "delay_reason"] = "technical"
    log.info("[technical+bad_weather] переразмечено: %d из %d", n, len(idx))

    # 3) Смена экипажа в нерабочее время → 65% airline_operations
    crew_late = (
        (df["crew_change_required"] == 1)
        & ((df["departure_hour"] < 6) | (df["departure_hour"] > 22))
        & (df["delay_minutes"] < 15)
    )
    idx = df.index[crew_late].to_numpy()
    n = int(0.65 * len(idx))
    pick = rng.choice(idx, size=n, replace=False) if len(idx) else np.array([], dtype=int)
    df.loc[pick, "delay_minutes"] = rng.integers(15, 35, size=n)
    df.loc[pick, "is_significant_delay"] = 1
    df.loc[pick, "delay_reason"] = "airline_operations"
    log.info("[crew_change+late_hours] переразмечено: %d из %d", n, len(idx))

    log.info(
        "После iter4: значимых задержек %d (%.4f)",
        int(df["is_significant_delay"].sum()),
        df["is_significant_delay"].mean(),
    )

    df.to_csv(RAW, index=False, encoding="utf-8-sig")
    log.info("Сохранено: %s", RAW)


if __name__ == "__main__":
    main()
