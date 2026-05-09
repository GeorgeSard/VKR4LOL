"""Препроцессор, списки колонок и feature engineering."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

NUMERIC_COLS: list[str] = [
    "distance_km",
    "planned_duration_min",
    "temperature_origin_c",
    "wind_speed_origin_mps",
    "precipitation_origin_mm",
    "visibility_origin_km",
    "airport_load_index",
    "airline_load_factor",
    "previous_flight_delay_min",
    "route_avg_delay_min",
    "aircraft_age_years",
    "weather_severity_score",
]

INT_PASSTHROUGH_COLS: list[str] = [
    "departure_hour",
    "month",
    "is_weekend",
    "technical_check_required",
    "crew_change_required",
    "bad_weather_origin",
    "low_visibility_origin",
    "wind_strong_origin",
    "precip_heavy_origin",
    "airport_high_load",
    "airline_high_load",
    "combined_high_load",
    "prev_delay_high",
    "prev_delay_med",
    "peak_hour",
    "cascade_risk",
    "is_winter",
]

CATEGORICAL_COLS: list[str] = [
    "airline_code",
    "origin_airport",
    "destination_airport",
    "route",
    "aircraft_type",
    "weather_origin",
    "weather_destination",
    "season",
    "day_of_week",
]

FEATURE_COLS: list[str] = NUMERIC_COLS + INT_PASSTHROUGH_COLS + CATEGORICAL_COLS

DROP_COLS: list[str] = [
    "flight_id",
    "flight_date",
    "scheduled_departure_local",
    "scheduled_arrival_local",
    "airline_name",
    "flight_number",
    "origin_city",
    "destination_city",
]

TARGET_DELAY = "is_significant_delay"
TARGET_REASON = "delay_reason"
RAW_TARGET_NUMERIC = "delay_minutes"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет интерпретируемые булевы флаги поверх сырых признаков.

    Все флаги основаны на операционных порогах (плохая погода, низкая
    видимость, высокая загрузка, каскадная задержка от предыдущего рейса).
    LightGBM ловит линейные пороги, но явные флаги резко уменьшают шум
    и ускоряют сходимость дерева на нужный сплит.
    """
    df = df.copy()

    weather_origin = df["weather_origin"].fillna("unknown") if "weather_origin" in df else pd.Series([""] * len(df))
    precipitation = df.get("precipitation_origin_mm", pd.Series([0] * len(df))).fillna(0)
    visibility = df.get("visibility_origin_km", pd.Series([10] * len(df))).fillna(10)
    wind = df.get("wind_speed_origin_mps", pd.Series([0] * len(df))).fillna(0)
    airport_load = df.get("airport_load_index", pd.Series([0] * len(df))).fillna(0)
    airline_load = df.get("airline_load_factor", pd.Series([0] * len(df))).fillna(0)
    prev_delay = df.get("previous_flight_delay_min", pd.Series([0] * len(df))).fillna(0)
    departure_hour = df.get("departure_hour", pd.Series([0] * len(df))).fillna(0).astype(int)
    season = df.get("season", pd.Series(["unknown"] * len(df))).fillna("unknown")

    df["bad_weather_origin"] = (
        weather_origin.isin(["fog", "snow"])
        | ((weather_origin == "rain") & (precipitation > 5))
    ).astype(int)
    df["low_visibility_origin"] = (visibility < 3).astype(int)
    df["wind_strong_origin"] = (wind > 12).astype(int)
    df["precip_heavy_origin"] = (precipitation > 5).astype(int)

    df["airport_high_load"] = (airport_load > 0.92).astype(int)
    df["airline_high_load"] = (airline_load > 0.85).astype(int)
    df["combined_high_load"] = (df["airport_high_load"] & df["airline_high_load"]).astype(int)

    df["prev_delay_high"] = (prev_delay > 60).astype(int)
    df["prev_delay_med"] = ((prev_delay > 30) & (prev_delay <= 60)).astype(int)
    df["cascade_risk"] = (df["prev_delay_high"] & df["combined_high_load"]).astype(int)

    df["peak_hour"] = departure_hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(int)
    df["is_winter"] = (season == "winter").astype(int)

    df["weather_severity_score"] = (
        df["bad_weather_origin"]
        + df["low_visibility_origin"]
        + df["wind_strong_origin"]
        + df["precip_heavy_origin"]
    ).astype(int)

    return df


def categorical_feature_indices() -> list[int]:
    """Индексы категориальных колонок в выходе ColumnTransformer.

    Порядок выхода: NUMERIC → INT_PASSTHROUGH → CATEGORICAL.
    Эти индексы передаются в LightGBM через `model__categorical_feature`.
    """
    start = len(NUMERIC_COLS) + len(INT_PASSTHROUGH_COLS)
    return list(range(start, start + len(CATEGORICAL_COLS)))


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    int_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype="int32",
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("int", int_pipe, INT_PASSTHROUGH_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
