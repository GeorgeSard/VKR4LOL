"""Загрузка моделей и предсказание для одного рейса."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_params, resolve_path
from src.preprocessing import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("predict")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def enrich_features(features: dict[str, Any]) -> dict[str, Any]:
    """Добавляет временные признаки и route, если их не передали."""
    enriched = dict(features)
    sched = enriched.get("scheduled_departure_local")
    if sched is not None and isinstance(sched, str):
        sched_dt = datetime.fromisoformat(sched.replace(" ", "T"))
    elif isinstance(sched, datetime):
        sched_dt = sched
    else:
        sched_dt = None

    if sched_dt is not None:
        enriched.setdefault("departure_hour", sched_dt.hour)
        enriched.setdefault("day_of_week", DAY_NAMES[sched_dt.weekday()])
        enriched.setdefault("month", sched_dt.month)
        enriched.setdefault("season", _season_from_month(sched_dt.month))
        enriched.setdefault("is_weekend", int(sched_dt.weekday() >= 5))

    if "route" not in enriched and "origin_airport" in enriched and "destination_airport" in enriched:
        enriched["route"] = f"{enriched['origin_airport']}-{enriched['destination_airport']}"

    return enriched


def _to_dataframe(features: dict[str, Any]) -> pd.DataFrame:
    row = {col: features.get(col) for col in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)


def load_models() -> dict[str, Any]:
    params = load_params()
    delay_dir = resolve_path(params["paths"]["delay_model_dir"])
    reason_dir = resolve_path(params["paths"]["reason_model_dir"])

    artifacts: dict[str, Any] = {
        "delay_model": None,
        "delay_threshold": 0.5,
        "reason_model": None,
        "reason_classes": [],
    }

    delay_model_path = delay_dir / "model.pkl"
    delay_meta_path = delay_dir / "metadata.json"
    if delay_model_path.exists():
        artifacts["delay_model"] = joblib.load(delay_model_path)
        if delay_meta_path.exists():
            with delay_meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            artifacts["delay_threshold"] = float(meta.get("threshold", 0.5))
        log.info("Загружена delay-модель из %s", delay_model_path)
    else:
        log.warning("delay-модель не найдена: %s", delay_model_path)

    reason_model_path = reason_dir / "model.pkl"
    if reason_model_path.exists():
        bundle = joblib.load(reason_model_path)
        artifacts["reason_model"] = bundle["pipeline"]
        artifacts["reason_classes"] = bundle["classes"]
        log.info("Загружена reason-модель из %s", reason_model_path)
    else:
        log.warning("reason-модель не найдена: %s", reason_model_path)

    return artifacts


def predict_flight(features: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    enriched = enrich_features(features)
    X = _to_dataframe(enriched)

    delay_model = models.get("delay_model")
    if delay_model is None:
        raise RuntimeError("delay-модель не загружена")

    threshold = float(models.get("delay_threshold", 0.5))
    proba = float(delay_model.predict_proba(X)[0, 1])
    is_delay = proba >= threshold

    response: dict[str, Any] = {
        "is_significant_delay": bool(is_delay),
        "delay_probability": round(proba, 4),
        "predicted_reason": None,
        "reason_probability": None,
    }

    if is_delay:
        reason_model = models.get("reason_model")
        classes = models.get("reason_classes") or []
        if reason_model is not None and classes:
            reason_proba = reason_model.predict_proba(X)[0]
            best_idx = int(reason_proba.argmax())
            response["predicted_reason"] = str(classes[best_idx])
            response["reason_probability"] = round(float(reason_proba[best_idx]), 4)

    return response


if __name__ == "__main__":
    sample = {
        "scheduled_departure_local": "2025-11-29 08:35",
        "airline_code": "FV",
        "origin_airport": "TJM",
        "destination_airport": "SVO",
        "aircraft_type": "A321",
        "distance_km": 1769,
        "planned_duration_min": 201,
        "weather_origin": "fog",
        "weather_destination": "clear",
        "temperature_origin_c": -3.9,
        "wind_speed_origin_mps": 3.8,
        "precipitation_origin_mm": 0.2,
        "visibility_origin_km": 1.9,
        "airport_load_index": 0.591,
        "airline_load_factor": 0.787,
        "previous_flight_delay_min": 38,
        "route_avg_delay_min": 6.5,
        "aircraft_age_years": 14,
        "technical_check_required": 0,
        "crew_change_required": 0,
    }
    artifacts = load_models()
    result = predict_flight(sample, artifacts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
