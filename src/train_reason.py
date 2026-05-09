"""Обучение модели мультиклассовой классификации причины задержки."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import get_mlflow_tracking_uri, load_params, resolve_path
from src.evaluate import (
    multiclass_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.preprocessing import FEATURE_COLS, TARGET_DELAY, TARGET_REASON, build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train_reason")


def _try_mlflow_log(
    params: dict, metrics: dict, model: Pipeline, classes: list[str], artifacts: list[Path]
) -> None:
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        mlflow.set_experiment(params["mlflow"]["experiment_name"])
        run_name = f"reason_train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({f"reason__{k}": v for k, v in params["train_reason"].items()})
            mlflow.log_param("split.test_size", params["split"]["test_size"])
            mlflow.log_param("split.random_state", params["split"]["random_state"])
            mlflow.log_param("reason__num_classes", len(classes))
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            for cls, vals in metrics.get("per_class", {}).items():
                mlflow.log_metric(f"f1__{cls}", vals["f1"])
            mlflow.sklearn.log_model(model, artifact_path="model")
            for path in artifacts:
                if path.exists():
                    mlflow.log_artifact(str(path))
        log.info("MLflow run сохранён: %s", run_name)
    except Exception as exc:
        log.warning("MLflow не доступен или ошибка логирования: %s", exc)


def main() -> None:
    params = load_params()
    features_path = resolve_path(params["paths"]["features_csv"])
    model_dir = resolve_path(params["paths"]["reason_model_dir"])
    metrics_dir = resolve_path(params["paths"]["metrics_dir"])
    figures_dir = resolve_path(params["paths"]["figures_dir"])

    log.info("Читаю фичи: %s", features_path)
    df = pd.read_csv(features_path)

    mask = (df[TARGET_DELAY].astype(int) == 1) & (df[TARGET_REASON].astype(str) != "none")
    df_filtered = df.loc[mask].reset_index(drop=True)
    log.info("Отобрано рейсов с причиной задержки: %d", len(df_filtered))
    if len(df_filtered) == 0:
        raise RuntimeError("Нет данных для обучения модели причины: проверь is_significant_delay и delay_reason")

    X = df_filtered[FEATURE_COLS]
    y_raw = df_filtered[TARGET_REASON].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes: list[str] = list(label_encoder.classes_)
    log.info("Классы (%d): %s", len(classes), classes)

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info("Train/test: %d / %d", len(X_train), len(X_test))

    cfg = params["train_reason"]
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        max_depth=cfg["max_depth"],
        min_child_samples=cfg["min_child_samples"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        class_weight=cfg["class_weight"],
        random_state=cfg["random_state"],
        n_jobs=-1,
        verbosity=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])

    log.info("Обучаю LightGBM (multiclass, n_estimators=%d)", cfg["n_estimators"])
    pipeline.fit(X_train, y_train)

    pred_test = pipeline.predict(X_test)
    y_test_named = label_encoder.inverse_transform(y_test)
    pred_test_named = label_encoder.inverse_transform(pred_test)
    metrics = multiclass_metrics(y_test_named, pred_test_named, classes=classes)
    log.info(
        "macro_f1=%.4f, weighted_f1=%.4f", metrics["macro_f1"], metrics["weighted_f1"]
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump({"pipeline": pipeline, "classes": classes}, model_path)
    log.info("Модель сохранена: %s", model_path)

    metadata = {
        "task": "multiclass_classification_delay_reason",
        "target": TARGET_REASON,
        "classes": classes,
        "features": FEATURE_COLS,
        "metrics": metrics,
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "library_versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "params": cfg,
    }
    metadata_path = model_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    metrics_path = metrics_dir / "reason_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("Метрики сохранены: %s", metrics_path)

    cm_path = figures_dir / "confusion_matrix_reason.png"
    plot_confusion_matrix(
        y_test_named, pred_test_named, labels=classes, out_path=cm_path,
        title="Confusion matrix — причина задержки",
    )

    importances = pipeline.named_steps["model"].feature_importances_
    fi_path = figures_dir / "feature_importance_reason.png"
    plot_feature_importance(
        importances,
        feature_names=FEATURE_COLS,
        out_path=fi_path,
        title="Feature importance — модель причины задержки",
    )
    log.info("Графики сохранены в %s", figures_dir)

    _try_mlflow_log(params, metrics, pipeline, classes, [cm_path, fi_path, metrics_path, metadata_path])


if __name__ == "__main__":
    main()
