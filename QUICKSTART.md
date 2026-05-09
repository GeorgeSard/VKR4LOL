# Быстрый старт — Docker Compose

Весь проект (FastAPI, MLflow, Prometheus, Grafana, cAdvisor + автоматическое обучение моделей через DVC) поднимается одной командой.

## Запуск с нуля

```bash
docker compose up --build
```

Что произойдёт:

1. Поднимется **MLflow** на `:5001` (внутри контейнера — порт 5000).
2. Запустится **trainer** (one-shot): выполнит `dvc init --no-scm` (если нужно) → `dvc repro` → `dvc metrics show`. Если данные/код/параметры не менялись с прошлого запуска, DVC ничего не пересчитывает и trainer завершается за пару секунд.
3. После успешного завершения trainer стартует **FastAPI** на `:8000`.
4. Параллельно поднимаются **Prometheus** (`:9090`), **Grafana** (`:3000`, admin/admin) и **cAdvisor** (`:8080`).

После завершения trainer в `docker compose logs trainer` будут видны метрики DVC:

```
--- DVC METRICS ---
Path                                  f1     macro_f1   ...
reports/metrics/delay_metrics.json    0.62   -          ...
reports/metrics/reason_metrics.json   -      0.42       ...
```

## Проверка API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "crew_change_required": 0
  }'
```

Swagger UI: <http://localhost:8000/docs>

## URL'ы сервисов

| Сервис | URL | Назначение |
|--------|-----|------------|
| FastAPI | <http://localhost:8000> | API + Swagger |
| FastAPI metrics | <http://localhost:8000/metrics> | Prometheus метрики |
| MLflow | <http://localhost:5001> | Эксперименты, runs, артефакты |
| Prometheus | <http://localhost:9090> | Targets, графики метрик |
| Grafana | <http://localhost:3000> | Дашборды (admin/admin) |
| cAdvisor | <http://localhost:8080> | Метрики Docker-контейнеров |

## Workflow «изменил данные → переобучил → увидел разницу в DVC»

```bash
# 1. Вносим правки в датасет
nano data/raw/flight_delays_ru_synthetic_2023_2025.csv
# ...или подкладываем новый CSV в data/raw/

# 2. Перезапускаем trainer (DVC сам пересчитает только то, что зависит от изменений)
docker compose run --rm trainer dvc repro

# 3. Смотрим метрики
docker compose run --rm trainer dvc metrics show

# 4. Сравниваем с прошлым git-коммитом
docker compose run --rm trainer dvc metrics diff

# 5. Перезапускаем FastAPI с новыми моделями
docker compose restart fastapi
```

То же сработает при изменении `params.yaml` (например, `n_estimators`) — DVC увидит изменение параметра и пересчитает только стадии `train_*`, не трогая `prepare`/`featurize`.

## Просмотр графа DVC pipeline

```bash
docker compose run --rm trainer dvc dag
```

## Импорт Grafana-дашборда

После `docker compose up`:

1. Открой <http://localhost:3000>, войди как `admin/admin`.
2. Configuration → Data Sources → Add → **Prometheus**, URL: `http://prometheus:9090`, Save & Test.
3. Dashboards → Import → загрузи файл `monitoring/grafana-dashboard.json`.

## Остановка

```bash
docker compose down            # сохраняет volumes (MLflow runs, Grafana, Prometheus)
docker compose down -v         # удаляет всё, включая volumes
```

## Локальный запуск без Docker (для отладки)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.prepare_data
python -m src.make_features
python -m src.train_delay
python -m src.train_reason
uvicorn app.main:app --reload
```
