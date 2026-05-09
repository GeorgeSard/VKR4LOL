"""Обучение модели бинарной классификации значимой задержки."""

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

from src.config import file_sha256_short, get_mlflow_tracking_uri, load_params, resolve_path
from src.evaluate import (
    binary_metrics,
    find_best_threshold,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.preprocessing import (
    FEATURE_COLS,
    TARGET_DELAY,
    build_preprocessor,
    categorical_feature_indices,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train_delay")


def _try_mlflow_log(
    params: dict,
    metrics: dict,
    model: Pipeline,
    artifacts: list[Path],
    training_data: dict,
) -> None:
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        mlflow.set_experiment(params["mlflow"]["experiment_name"])
        run_name = f"delay_train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({f"delay__{k}": v for k, v in params["train_delay"].items()})
            mlflow.log_param("split.test_size", params["split"]["test_size"])
            mlflow.log_param("split.random_state", params["split"]["random_state"])
            mlflow.log_params({f"data__{k}": v for k, v in training_data.items()})
            mlflow.set_tag("data_sha256_16", training_data.get("raw_sha256_16", "unknown"))
            mlflow.set_tag("dataset_path", training_data.get("raw_csv_path", "unknown"))
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
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
    model_dir = resolve_path(params["paths"]["delay_model_dir"])
    metrics_dir = resolve_path(params["paths"]["metrics_dir"])
    figures_dir = resolve_path(params["paths"]["figures_dir"])

    log.info("Читаю фичи: %s", features_path)
    df = pd.read_csv(features_path)
    log.info("Строк: %d, колонок-признаков: %d", len(df), len(FEATURE_COLS))

    X = df[FEATURE_COLS]
    y = df[TARGET_DELAY].astype(int).values

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info("Train/test: %d / %d", len(X_train), len(X_test))

    cfg = params["train_delay"]
    model = LGBMClassifier(
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

    cat_indices = categorical_feature_indices()
    log.info("Обучаю LightGBM (n_estimators=%d, native categorical=%s)", cfg["n_estimators"], cat_indices)
    pipeline.fit(X_train, y_train, model__categorical_feature=cat_indices)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    best_t, best_f1 = find_best_threshold(
        y_test, proba_test, cfg["threshold_min"], cfg["threshold_max"], cfg["threshold_step"]
    )
    pred_test = (proba_test >= best_t).astype(int)
    metrics = binary_metrics(y_test, pred_test, proba_test, best_t)
    log.info(
        "Best threshold=%.2f, F1=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f",
        best_t,
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump(pipeline, model_path)
    log.info("Модель сохранена: %s", model_path)

    raw_csv_path = resolve_path(params["paths"]["raw_csv"])
    metadata = {
        "task": "binary_classification_significant_delay",
        "target": TARGET_DELAY,
        "threshold": float(best_t),
        "features": FEATURE_COLS,
        "metrics": metrics,
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "training_data": {
            "raw_csv_path": str(raw_csv_path.relative_to(resolve_path("."))),
            "raw_sha256_16": file_sha256_short(raw_csv_path),
            "raw_size_bytes": raw_csv_path.stat().st_size,
            "features_csv_sha256_16": file_sha256_short(features_path),
            "rows_total": int(len(df)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "positive_share_train": round(float(y_train.mean()), 6),
        },
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

    metrics_path = metrics_dir / "delay_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("Метрики сохранены: %s", metrics_path)

    cm_path = figures_dir / "confusion_matrix_delay.png"
    plot_confusion_matrix(
        y_test,
        pred_test,
        labels=[0, 1],
        display_labels=["norm", "delay>=15"],
        out_path=cm_path,
        title=f"Confusion matrix — задержка (threshold={best_t:.2f})",
    )

    importances = pipeline.named_steps["model"].feature_importances_
    fi_path = figures_dir / "feature_importance_delay.png"
    plot_feature_importance(
        importances,
        feature_names=FEATURE_COLS,
        out_path=fi_path,
        title="Feature importance — модель задержки",
    )
    log.info("Графики сохранены в %s", figures_dir)

    _try_mlflow_log(
        params,
        metrics,
        pipeline,
        [cm_path, fi_path, metrics_path, metadata_path],
        training_data=metadata["training_data"],
    )


if __name__ == "__main__":
    main()
