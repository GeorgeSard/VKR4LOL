# DVC для защиты ВКР — что показывать и как

Этот файл — шпаргалка по командам DVC и git, чтобы продемонстрировать процесс улучшения модели в отчёте/презентации.

## Краткая история итераций

| # | Что менялось | F1 (delay) | ROC-AUC | macro_F1 (reason) |
|---|---|---|---|---|
| 0 | Baseline (синтетика как есть) | 0.3274 | 0.5972 | 0.2152 |
| 1 | Улучшение качества данных (`scripts/enhance_data.py`) | 0.5487 | 0.7311 | 0.3099 |
| 2 | Гиперпараметры: больше деревьев, меньше lr | 0.5465 | 0.7260 | 0.2913 |
| 3 | Гиперпараметры: + регуляризация (max_depth, reg_alpha, reg_lambda) | 0.5480 | **0.7327** | **0.3184** |
| 4 | Ещё больше деревьев — без улучшения | 0.5473 | 0.7260 | 0.3057 |
| 5 | **финал**: откат на лучшие params iter 3 (DVC взял из кэша) | 0.5480 | 0.7327 | 0.3184 |

**Главный вывод:** качество данных дало +22 п.п. F1, тюнинг гиперпараметров — ещё +1 п.п. ROC-AUC. Дальнейшие итерации уперлись в плато.

## Как доказать, что модель в коммите N обучалась на других данных, чем в коммите M

Главный инструмент — `scripts/verify_iteration.py`. Берёт SHA коммита, печатает:
- хеш сырого CSV в этом коммите (sha256, первые 16 символов)
- хеш данных, на которых реально обучалась модель (записан в `metadata.json`)
- хеш CSV в HEAD сейчас
- хеш CSV в самом первом коммите (база для сравнения)

```bash
# проверка iter 3
docker compose run --rm trainer python scripts/verify_iteration.py 5916620
```

Пример вывода:

```
=== Коммит 5916620 — iter3: регуляризация ===

📊 Сырые данные в этом коммите:
   sha256_16: c1fcca960cb9a24d
   rows:      120360

🤖 Delay-модель в этом коммите:
   trained_at:  2026-05-09T14:34:01
   F1:          0.5480
   ROC-AUC:     0.7327

📋 Сводка по данным:
   Этот коммит (5916620):    c1fcca960cb9a24d
   Базовый коммит (92e85cd): 21e2c33adca2b7d0 (ДАННЫЕ ОТЛИЧАЮТСЯ)
   HEAD сейчас:              c1fcca960cb9a24d (данные те же)
```

**Что это доказывает для iter 3:**
- В iter 3 данные имели хеш `c1fcca960cb9a24d`
- В baseline (iter 0) хеш был `21e2c33adca2b7d0` — **РАЗНЫЕ**
- Значит модель iter 3 действительно училась на других данных, чем модель iter 0
- F1 модели iter 3 (0.5480) ≠ F1 модели iter 0 (0.3274) — это и подтверждается

Для коммитов начиная с **30b5...** (после добавления training_data в metadata) поле `trained on data` будет содержать конкретный хеш — это даёт абсолютное доказательство привязки модели к данным:

```bash
# для будущих итераций (или после переобучения текущего HEAD)
docker compose run --rm trainer python scripts/verify_iteration.py HEAD
# покажет: trained on data: c1fcca960cb9a24d  ✓ МАТЧ с CSV в коммите
```

### Сценарий проверки на защите

Тебе задают вопрос «А докажи, что в iter 3 модель обучалась не на тех же данных, что в iter 0»:

```bash
# 1. показываешь хеш данных в iter 3
docker compose run --rm trainer python scripts/verify_iteration.py 5916620 | grep -A2 "Сводка"

# 2. показываешь хеш данных в iter 0
docker compose run --rm trainer python scripts/verify_iteration.py 92e85cd | grep -A2 "Сводка"

# 3. сравниваешь хеши: они разные → данные разные → модели обучались на разных данных
```

Дополнительно для подкрепления:
```bash
# показать итоговый F1 в каждом коммите
git show 92e85cd:reports/metrics/delay_metrics.json | grep f1   # 0.3274
git show 5916620:reports/metrics/delay_metrics.json | grep f1   # 0.5480

# показать положительный класс в каждом коммите (требует data_stats, появилось в HEAD)
git show 5916620:data/raw/flight_delays_ru_synthetic_2023_2025.csv | wc -l   # 120361 (с заголовком)
git show 92e85cd:data/raw/flight_delays_ru_synthetic_2023_2025.csv | wc -l   # 120361
```

## Прозрачность изменений данных через DVC

**Проблема:** в репо лежит один файл данных, и без специального инструментария не видно, что он менялся.

**Решение в этом проекте:**

1. **DVC-стадия `data_stats`** считает по `data/raw/*.csv` сводку (rows, positive_share, sha256, mean delay) и сохраняет в `reports/metrics/data_stats.json`. Эта метрика отслеживается DVC.

```bash
# текущая статистика данных (печатает все метрики, включая data_stats)
docker compose run --rm trainer dvc metrics show

# или прямо открыть файл
cat reports/metrics/data_stats.json
```

2. **`scripts/show_data_evolution.py`** проходит по всей git-истории, для каждого коммита извлекает CSV (`git show <sha>:data/raw/...csv`), считает статистику и выдаёт сравнительную таблицу:

```bash
docker compose run --rm trainer python scripts/show_data_evolution.py
```

Результат на текущей истории:

```
commit      rows   positive mean_delay  subject
------------------------------------------------------------------------------------------
92e85cd   120360     18.22%      45.23  init: ВКР проект прогнозирования задержек авиарейс
c117cca   120360     25.07%      40.34  iter1: data quality fix — усиление weather/airport
```

Чётко видно: после iter1 положительный класс вырос с 18.22% до 25.07% — данные действительно изменились. В коммитах iter2–iter5 данные те же (менялись только гиперпараметры), и скрипт это отмечает в логе.

3. **git diff по data_stats.json** показывает изменение метрики данных между коммитами:

```bash
git log --oneline reports/metrics/data_stats.json
git diff <sha1> <sha2> -- reports/metrics/data_stats.json
```

4. **DVC видит изменение сырого файла** через хеш в `dvc.lock`. После `python scripts/enhance_data.py` команда `dvc status` покажет:

```
data_stats:
    changed deps:
        data/raw/flight_delays_ru_synthetic_2023_2025.csv
prepare:
    changed deps:
        data/raw/flight_delays_ru_synthetic_2023_2025.csv
```

Это значит DVC знает о смене данных и понимает, что пайплайн надо пересчитать.

## Команды DVC, которые надо запустить и показать

Все команды выполняются через Docker (стек должен быть поднят):

### 1. Граф пайплайна
```bash
docker compose run --rm trainer dvc dag
```
Показывает структуру: `prepare → featurize → train_delay/train_reason`. На презентации хорошо смотрится.

### 2. Текущие метрики
```bash
docker compose run --rm trainer dvc metrics show
```
Показывает значения из `reports/metrics/*.json` для текущего состояния репо.

### 3. Статус пайплайна
```bash
docker compose run --rm trainer dvc status
```
Если изменили данные/код/params — DVC скажет, какие стадии нуждаются в переобучении.

### 4. Воспроизведение пайплайна
```bash
docker compose run --rm trainer dvc repro
```
Перезапускает только те стадии, которые реально изменились. Если ничего не менялось — печатает "Stage X didn't change, skipping".

### 5. Параметры
```bash
docker compose run --rm trainer dvc params diff HEAD~1 HEAD
```
Покажет, какие параметры менялись в последнем коммите.

### 6. История коммитов с метриками (через git)
DVC `metrics diff` между коммитами требует git-режима, у нас он отключён, поэтому смотрим напрямую через git:

```bash
# показать содержимое метрик в любом коммите
git show HEAD:reports/metrics/delay_metrics.json
git show HEAD~3:reports/metrics/delay_metrics.json

# diff между двумя коммитами
git diff HEAD~3 HEAD -- reports/metrics/

# вся история изменений метрик
git log -p reports/metrics/delay_metrics.json
```

Альтернатива: ручная сводка в `reports/history/iterations.md` (создана автоматически).

## Сценарий демонстрации на защите (5–7 минут)

### Шаг 1. Показать архитектуру пайплайна
```bash
docker compose run --rm trainer dvc dag
```
"Вот четыре стадии моего пайплайна: подготовка данных → формирование признаков → обучение двух моделей."

### Шаг 2. Показать стартовые метрики
```bash
git show 92e85cd:reports/metrics/delay_metrics.json
```
Или открыть `reports/history/iterations.md` — там вся таблица итераций.

### Шаг 3. Показать итерацию с улучшением данных
```bash
git log --oneline | grep iter
git show <SHA коммита iter1> --stat
```
"В iter1 я переразметил 8242 рейса по операционным признакам — это дало основной прирост."

### Шаг 4. Показать DVC «увидел» изменение и пересчитал
```bash
# вернуться на коммит до iter1
git checkout 92e85cd -- params.yaml data/raw/
docker compose run --rm trainer dvc status
```
DVC покажет:
```
prepare:
    changed deps:
        data/raw/...: modified
```

### Шаг 5. Показать переобучение
```bash
docker compose run --rm trainer dvc repro
```
Запустится prepare → featurize → train_delay → train_reason. На реальной демонстрации можно вернуться обратно через `git checkout main -- .`

### Шаг 6. Показать кэш
```bash
docker compose run --rm trainer dvc repro
```
Второй раз сразу скажет "Stage X didn't change, skipping" — DVC не делает лишней работы.

### Шаг 7. Показать MLflow
Открыть http://localhost:5001 — увидеть все 10 runs (5 итераций × 2 модели), сравнить метрики, скачать модели.

## Workflow «изменил данные → переобучил → увидел разницу»

```bash
# 1. Изменили данные (например, новый CSV или скрипт enhance_data.py)
docker compose run --rm trainer python scripts/enhance_data.py

# 2. DVC видит изменение
docker compose run --rm trainer dvc status
# → "prepare: changed deps: data/raw/...: modified"

# 3. Перезапускаем пайплайн
docker compose run --rm trainer dvc repro

# 4. Смотрим обновлённые метрики
docker compose run --rm trainer dvc metrics show

# 5. Сравниваем с предыдущим состоянием через git
git diff HEAD~1 HEAD -- reports/metrics/

# 6. Перезагружаем модели в API
docker compose restart fastapi
```

## Workflow «изменил гиперпараметры → переобучил»

```bash
# 1. Редактируем params.yaml (например, n_estimators)
nano params.yaml

# 2. DVC видит, что изменился train_delay (но не prepare/featurize — они не зависят от этих params)
docker compose run --rm trainer dvc status

# 3. Перезапускаем — DVC пересчитает только train_*
docker compose run --rm trainer dvc repro
# → "Stage 'prepare' didn't change, skipping"
# → "Stage 'featurize' didn't change, skipping"
# → "Running stage 'train_delay'"
# → "Running stage 'train_reason'"
```

## Где искать артефакты

| Что | Путь |
|---|---|
| Метрики по итерациям | `reports/metrics/delay_metrics.json`, `reports/metrics/reason_metrics.json` |
| Confusion matrix | `reports/figures/confusion_matrix_delay.png`, `..._reason.png` |
| Feature importance | `reports/figures/feature_importance_delay.png`, `..._reason.png` |
| Журнал итераций | `reports/history/iterations.md` |
| Hyperparameters текущие | `params.yaml` |
| Hyperparameters в коммите | `git show <SHA>:params.yaml` |
| Всё про MLflow | http://localhost:5001 (после `docker compose up`) |

## Что показывать в Grafana

После `docker compose up` дашборд автоматически провижится:

URL: http://localhost:3000/d/vkr-flight-delay (admin / admin)

Панели:
- FastAPI: суммарный RPS / Всего запросов / p95 latency / Error rate (4 stat-панели сверху)
- FastAPI: RPS по эндпоинтам (timeseries, видно `/predict`, `/health`, `/metrics`)
- FastAPI: p50/p95 latency по эндпоинтам
- FastAPI: статус-коды (2xx/3xx/4xx/5xx)
- cAdvisor: CPU контейнеров
- cAdvisor: память контейнеров
- cAdvisor: сетевой трафик контейнеров

Перед демонстрацией лучше нагенерить трафика:
```bash
for i in {1..200}; do curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @example.json > /dev/null; done
```

Где `example.json` — пример запроса из `CLAUDE.md` (раздел `## FastAPI`).
