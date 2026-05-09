# Прогнозирование задержек авиарейсов и классификация причин

Учебный ML-проект для ВКР на тему: «Разработка модели искусственного интеллекта для прогнозирования задержек авиарейсов и классификации их причин».

Цель проекта - показать полный инженерный ML-процесс в простой форме: CSV-данные, очистка, feature engineering, DVC pipeline, обучение двух моделей, MLflow, FastAPI и мониторинг контейнеров через Prometheus/Grafana.

Проект не является production-системой. Датасет `data/raw/flight_delays_ru_synthetic_2023_2025.csv` синтетический и нужен только для учебной демонстрации.

## ML-задачи

1. Прогноз операционно значимой задержки рейса.

   Это бинарная классификация:

   ```text
   delay_minutes >= 15 -> is_significant_delay = 1
   delay_minutes < 15  -> is_significant_delay = 0
   ```

   Задержка меньше 15 минут не считается "идеально вовремя" в бытовом смысле, но в ML-задаче относится к классу небольших отклонений от расписания. Порог 15 минут выбран как распространенный операционный порог оценки пунктуальности авиарейсов.

2. Классификация причины задержки.

   Это многоклассовая классификация по `delay_reason`. Вторая модель обучается только на рейсах, где `is_significant_delay == 1` и `delay_reason != "none"`. В API она вызывается только если первая модель прогнозирует высокий риск значимой задержки.

Регрессионная модель задержки в минутах в проект не добавлена намеренно.

## Структура

```text
app/
  main.py
src/
  prepare_data.py
  make_features.py
  train_delay.py
  train_reason.py
  evaluate.py
  predict.py
data/
  raw/
  interim/
  processed/
models/
  delay_model/
  reason_model/
reports/
  figures/
  metrics/
monitoring/
  prometheus.yml
  grafana-dashboard.json
params.yaml
dvc.yaml
Dockerfile
docker-compose.yml
requirements.txt
```

## Данные

Основной файл:

```text
data/raw/flight_delays_ru_synthetic_2023_2025.csv
```

Сопроводительные файлы:

```text
data/raw/flight_delays_ru_README_2023_2025.md
data/raw/flight_delays_ru_data_dictionary_2023_2025.csv
data/raw/flight_delays_ru_summary_2023_2025.csv
```

В модель не подаются целевые и результирующие поля:

```text
delay_minutes
is_significant_delay
delay_reason
```

## Локальный запуск

Короткая инструкция для запуска пайплайна, Docker Compose, MLflow, Grafana и Prometheus вынесена в [QUICKSTART.md](QUICKSTART.md).

Создать окружение и установить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запустить пайплайн вручную:

```bash
python src/prepare_data.py
python src/make_features.py
python src/train_delay.py
python src/train_reason.py
```

После обучения появятся:

```text
data/interim/cleaned_flights.csv
data/processed/features.csv
models/delay_model/model.pkl
models/delay_model/metadata.json
models/reason_model/model.pkl
models/reason_model/metadata.json
reports/metrics/delay_metrics.json
reports/metrics/reason_metrics.json
reports/figures/confusion_matrix_delay.png
reports/figures/confusion_matrix_reason.png
reports/figures/feature_importance_delay.png
reports/figures/feature_importance_reason.png
```

## DVC

Инициализация DVC, если репозиторий еще не инициализирован:

```bash
dvc init
```

Запуск pipeline:

```bash
dvc repro
```

Полезные команды для демонстрации:

```bash
dvc dag
dvc status
dvc metrics show
dvc metrics diff
```

Pipeline состоит из стадий:

```text
prepare -> featurize -> train_delay
                    \-> train_reason
```

## MLflow

При установленном `mlflow` обучение логирует параметры, метрики, артефакты и модели в эксперимент `flight-delay-vkr`.

Локальный запуск UI без Docker:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

В Docker Compose MLflow доступен на внешнем порту `5001`, потому что на macOS порт `5000` часто занят системным процессом:

```text
http://localhost:5001
```

Для первой модели логируются `precision`, `recall`, `f1`, `ROC-AUC`, `PR-AUC`, threshold, confusion matrix и feature importance. Основная метрика - `f1`.

Для второй модели логируются `macro-F1`, `weighted-F1`, confusion matrix и feature importance. Основная метрика - `macro-F1`.

## FastAPI

Запуск локально после обучения моделей:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

Проверка состояния:

```bash
curl http://localhost:8000/health
```

Пример прогноза:

```bash
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

Пример ответа:

```json
{
  "is_significant_delay": true,
  "delay_probability": 0.78,
  "predicted_reason": "weather",
  "reason_probability": 0.65
}
```

Если значимая задержка не прогнозируется, `predicted_reason` и `reason_probability` возвращаются как `null`.

## Docker Compose и мониторинг

Запуск всех сервисов:

```bash
docker compose up -d --build
```

Сервисы:

```text
FastAPI:     http://localhost:8000
MLflow:      http://localhost:5001
Prometheus:  http://localhost:9090
Grafana:     http://localhost:3000
cAdvisor:    http://localhost:8080
```

Grafana login/password по умолчанию:

```text
admin / admin
```

Файл `monitoring/grafana-dashboard.json` можно импортировать в Grafana вручную. Prometheus собирает метрики с FastAPI `/metrics` и cAdvisor.

Ручка `GET /system/containers` использует Docker socket. Это сделано только для локальной учебной демонстрации и не должно использоваться как production-подход.

## Скриншоты для защиты

1. Структура проекта в VS Code.
2. Исходный CSV-файл.
3. Очищенный CSV-файл.
4. DVC pipeline.
5. DVC status.
6. DVC metrics diff.
7. MLflow UI со списком экспериментов.
8. MLflow run с гиперпараметрами.
9. MLflow run с метриками.
10. Confusion matrix для прогноза значимой задержки.
11. Confusion matrix для классификации причины задержки.
12. Feature importance.
13. Swagger UI FastAPI.
14. POST `/predict`.
15. POST `/train`.
16. GET `/experiments`.
17. Grafana dashboard.
18. Prometheus targets.
19. Docker Compose containers.
