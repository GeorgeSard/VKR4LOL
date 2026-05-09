"""FastAPI приложение: предсказание задержки + причины, переобучение, мониторинг."""

from __future__ import annotations

import logging
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.config import get_mlflow_tracking_uri
from src.predict import enrich_features, load_models, predict_flight

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("api")

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/app")

state: dict[str, Any] = {
    "models": None,
    "models_loaded_at": None,
    "training_in_progress": False,
    "last_training_status": None,
}


def _refresh_models() -> None:
    log.info("Загружаю модели в память")
    artifacts = load_models()
    state["models"] = artifacts
    state["models_loaded_at"] = datetime.utcnow().isoformat(timespec="seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _refresh_models()
    except Exception as exc:
        log.warning("Не удалось загрузить модели при старте: %s", exc)
        state["models"] = {"delay_model": None, "reason_model": None}
    yield


app = FastAPI(
    title="Flight Delay Prediction API (ВКР)",
    description="Прогноз значимой задержки авиарейса и классификация причины.",
    version="1.0.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


class PredictRequest(BaseModel):
    scheduled_departure_local: str = Field(..., examples=["2025-11-29 08:35"])
    airline_code: str
    origin_airport: str
    destination_airport: str
    aircraft_type: str
    distance_km: float
    planned_duration_min: float
    weather_origin: Optional[str] = None
    weather_destination: Optional[str] = None
    temperature_origin_c: Optional[float] = None
    wind_speed_origin_mps: Optional[float] = None
    precipitation_origin_mm: Optional[float] = None
    visibility_origin_km: Optional[float] = None
    airport_load_index: Optional[float] = None
    airline_load_factor: Optional[float] = None
    previous_flight_delay_min: Optional[float] = None
    route_avg_delay_min: Optional[float] = None
    aircraft_age_years: Optional[float] = None
    technical_check_required: Optional[int] = 0
    crew_change_required: Optional[int] = 0


class PredictResponse(BaseModel):
    is_significant_delay: bool
    delay_probability: float
    predicted_reason: Optional[str] = None
    reason_probability: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    delay_model: bool
    reason_model: bool
    models_loaded_at: Optional[str] = None
    training_in_progress: bool


class TrainResponse(BaseModel):
    status: str
    message: str


class ExperimentRun(BaseModel):
    run_id: str
    run_name: Optional[str]
    experiment_id: str
    status: Optional[str]
    start_time: Optional[str]
    metrics: dict
    params: dict


class ContainerInfo(BaseModel):
    id: str
    name: str
    image: str
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models = state.get("models") or {}
    return HealthResponse(
        status="ok",
        delay_model=models.get("delay_model") is not None,
        reason_model=models.get("reason_model") is not None,
        models_loaded_at=state.get("models_loaded_at"),
        training_in_progress=state.get("training_in_progress", False),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    models = state.get("models") or {}
    if models.get("delay_model") is None:
        raise HTTPException(
            status_code=503,
            detail="delay-модель не загружена. Запустите обучение через POST /train.",
        )
    features = enrich_features(req.model_dump())
    result = predict_flight(features, models)
    return PredictResponse(**result)


def _run_dvc_repro() -> None:
    state["training_in_progress"] = True
    state["last_training_status"] = "running"
    log.info("Запускаю dvc repro в %s", PROJECT_ROOT)
    try:
        proc = subprocess.run(
            ["dvc", "repro"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        log.info("dvc stdout:\n%s", proc.stdout)
        if proc.returncode != 0:
            log.error("dvc вернул код %d, stderr:\n%s", proc.returncode, proc.stderr)
            state["last_training_status"] = f"failed (code {proc.returncode})"
            return
        state["last_training_status"] = "ok"
        _refresh_models()
    except FileNotFoundError:
        log.error("Команда dvc не найдена в PATH")
        state["last_training_status"] = "dvc not found"
    except Exception as exc:
        log.exception("Ошибка обучения: %s", exc)
        state["last_training_status"] = f"error: {exc}"
    finally:
        state["training_in_progress"] = False


@app.post("/train", response_model=TrainResponse)
def train(background_tasks: BackgroundTasks) -> TrainResponse:
    if state.get("training_in_progress"):
        return TrainResponse(status="already_running", message="Обучение уже идёт")
    background_tasks.add_task(_run_dvc_repro)
    return TrainResponse(status="started", message="dvc repro запущен в фоне")


@app.get("/experiments", response_model=list[ExperimentRun])
def experiments() -> list[ExperimentRun]:
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=get_mlflow_tracking_uri())
        results: list[ExperimentRun] = []
        for exp in client.search_experiments():
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=50,
                order_by=["attribute.start_time DESC"],
            )
            for run in runs:
                start_ms = run.info.start_time
                start_str = (
                    datetime.utcfromtimestamp(start_ms / 1000).isoformat(timespec="seconds")
                    if start_ms
                    else None
                )
                results.append(
                    ExperimentRun(
                        run_id=run.info.run_id,
                        run_name=run.data.tags.get("mlflow.runName"),
                        experiment_id=exp.experiment_id,
                        status=run.info.status,
                        start_time=start_str,
                        metrics=dict(run.data.metrics),
                        params=dict(run.data.params),
                    )
                )
        return results
    except Exception as exc:
        log.warning("MLflow недоступен: %s", exc)
        raise HTTPException(status_code=503, detail=f"MLflow недоступен: {exc}")


@app.get("/system/containers", response_model=list[ContainerInfo])
def system_containers() -> list[ContainerInfo]:
    try:
        import docker

        client = docker.from_env()
        containers = client.containers.list(all=True)
        return [
            ContainerInfo(
                id=c.short_id,
                name=c.name,
                image=(c.image.tags[0] if c.image.tags else c.image.short_id),
                status=c.status,
            )
            for c in containers
        ]
    except Exception as exc:
        log.warning("Docker socket недоступен: %s", exc)
        raise HTTPException(status_code=503, detail=f"Docker недоступен: {exc}")
