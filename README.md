# Прогнозирование задержек авиарейсов и классификация причин

Учебный ML-проект для ВКР: бинарная классификация значимых задержек (`is_significant_delay`) + многоклассовая классификация причины задержки (`delay_reason`). LightGBM, scikit-learn pipelines, DVC, MLflow, FastAPI, Docker Compose, Prometheus + Grafana + cAdvisor.

## Запуск

```bash
docker compose up --build
```

Подробности — в [QUICKSTART.md](QUICKSTART.md). Полная спецификация и контекст проекта — в [CLAUDE.md](CLAUDE.md).

## Что внутри

```
src/                ML-пайплайн (prepare → featurize → train_delay/train_reason)
app/main.py         FastAPI: /health, /predict, /train, /experiments, /metrics, /system/containers
dvc.yaml            DVC pipeline (4 стадии)
params.yaml         все гиперпараметры
docker-compose.yml  6 сервисов: mlflow, trainer (one-shot), fastapi, prometheus, grafana, cadvisor
monitoring/         Prometheus config + Grafana dashboard
data/raw/           синтетический датасет 2023-2025 (120 360 рейсов)
```

## Workflow «изменил данные — увидел результат в DVC»

```bash
docker compose run --rm trainer dvc repro
docker compose run --rm trainer dvc metrics show
docker compose run --rm trainer dvc metrics diff
docker compose restart fastapi
```

## Данные

Синтетический датасет — `data/raw/flight_delays_ru_synthetic_2023_2025.csv`. Описание — `data/raw/flight_delays_ru_README_2023_2025.md`, словарь колонок — `data/raw/flight_delays_ru_data_dictionary_2023_2025.csv`.
