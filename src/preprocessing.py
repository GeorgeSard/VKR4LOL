"""Препроцессор и списки колонок."""

from __future__ import annotations

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
]

INT_PASSTHROUGH_COLS: list[str] = [
    "departure_hour",
    "month",
    "is_weekend",
    "technical_check_required",
    "crew_change_required",
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
