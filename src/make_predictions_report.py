"""Генерация финального отчёта: прогноз vs факт по тестовым выборкам.

Каждая модель имеет СВОЮ тестовую выборку (разные индексы):

  delay-модель    : 20% от всех 120 000 рейсов (test=24 000)
  reason-модель   : 20% от 44 136 рейсов с реальной значимой задержкой (test ≈ 8 828)

Чтобы метрики не были задвоены leak-ом, оцениваем каждую модель только на
её собственном тестовом сплите.

Файлы на выходе:

  reports/predictions_delay.csv
      24 000 строк: прогноз delay-модели vs реальная значимая задержка.
      Колонки: actual_is_delay, predicted_is_delay, delay_correct,
      delay_probability + ключевые признаки рейса.

  reports/predictions_reason.csv
      ~8 828 строк (только реальные значимые задержки, тест reason-модели):
      actual_reason, predicted_reason, reason_correct, reason_probability +
      ключевые признаки.

  reports/predictions_summary.json
      Сводные метрики обеих моделей (accuracy / F1 / precision / recall /
      ROC-AUC, confusion matrix).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import load_params, resolve_path
from src.preprocessing import (
    FEATURE_COLS,
    TARGET_DELAY,
    TARGET_REASON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("predictions_report")

DESCRIPTIVE_COLS = [
    "airline_code",
    "origin_airport",
    "destination_airport",
    "route",
    "aircraft_type",
    "departure_hour",
    "weather_origin",
    "weather_destination",
    "visibility_origin_km",
    "precipitation_origin_mm",
    "airport_load_index",
    "airline_load_factor",
    "previous_flight_delay_min",
    "technical_check_required",
    "crew_change_required",
]


def main() -> None:
    params = load_params()
    features_path = resolve_path(params["paths"]["features_csv"])
    delay_dir = resolve_path(params["paths"]["delay_model_dir"])
    reason_dir = resolve_path(params["paths"]["reason_model_dir"])
    reports_dir = resolve_path("reports")

    log.info("Читаю фичи: %s", features_path)
    df = pd.read_csv(features_path).reset_index(drop=True)

    delay_pipeline = joblib.load(delay_dir / "model.pkl")
    with (delay_dir / "metadata.json").open("r", encoding="utf-8") as f:
        delay_meta = json.load(f)
    delay_threshold = float(delay_meta["threshold"])
    log.info("Загружена delay-модель, threshold=%.4f", delay_threshold)

    reason_bundle = joblib.load(reason_dir / "model.pkl")
    reason_pipeline = reason_bundle["pipeline"]
    reason_classes: list[str] = reason_bundle["classes"]
    log.info("Загружена reason-модель, классов=%d", len(reason_classes))

    test_size = params["split"]["test_size"]
    rs = params["split"]["random_state"]

    # === DELAY: ровно тот же split, что в src/train_delay.py ===
    y_delay = df[TARGET_DELAY].astype(int).values
    indices_all = np.arange(len(df))
    _, delay_test_idx = train_test_split(
        indices_all, test_size=test_size, random_state=rs, stratify=y_delay
    )
    df_delay_test = df.iloc[delay_test_idx].reset_index(drop=True)
    X_delay_test = df_delay_test[FEATURE_COLS]
    delay_proba = delay_pipeline.predict_proba(X_delay_test)[:, 1]
    delay_pred = (delay_proba >= delay_threshold).astype(int)
    actual_delay = df_delay_test[TARGET_DELAY].astype(int).values

    delay_csv = pd.DataFrame({
        "actual_is_delay": actual_delay,
        "predicted_is_delay": delay_pred,
        "delay_correct": (actual_delay == delay_pred),
        "delay_probability": np.round(delay_proba, 4),
    })
    delay_csv = pd.concat(
        [delay_csv, df_delay_test[DESCRIPTIVE_COLS].reset_index(drop=True)], axis=1
    )
    delay_csv_path = reports_dir / "predictions_delay.csv"
    delay_csv.to_csv(delay_csv_path, index=False)
    log.info("Записано: %s (%d строк)", delay_csv_path, len(delay_csv))

    # === REASON: ровно тот же split, что в src/train_reason.py ===
    mask = (df[TARGET_DELAY].astype(int) == 1) & (df[TARGET_REASON].astype(str) != "none")
    df_reason = df.loc[mask].reset_index(drop=True)
    y_reason_classes = df_reason[TARGET_REASON].astype(str).values
    indices_reason = np.arange(len(df_reason))
    _, reason_test_idx = train_test_split(
        indices_reason, test_size=test_size, random_state=rs, stratify=y_reason_classes
    )
    df_reason_test = df_reason.iloc[reason_test_idx].reset_index(drop=True)
    X_reason_test = df_reason_test[FEATURE_COLS]
    reason_proba_matrix = reason_pipeline.predict_proba(X_reason_test)
    reason_pred_idx = reason_proba_matrix.argmax(axis=1)
    reason_pred_named = np.array([reason_classes[i] for i in reason_pred_idx])
    reason_pred_proba = reason_proba_matrix[np.arange(len(reason_pred_idx)), reason_pred_idx]
    actual_reason = df_reason_test[TARGET_REASON].astype(str).values

    reason_csv = pd.DataFrame({
        "actual_reason": actual_reason,
        "predicted_reason": reason_pred_named,
        "reason_correct": (actual_reason == reason_pred_named),
        "reason_probability": np.round(reason_pred_proba, 4),
    })
    reason_csv = pd.concat(
        [reason_csv, df_reason_test[DESCRIPTIVE_COLS].reset_index(drop=True)], axis=1
    )
    reason_csv_path = reports_dir / "predictions_reason.csv"
    reason_csv.to_csv(reason_csv_path, index=False)
    log.info("Записано: %s (%d строк)", reason_csv_path, len(reason_csv))

    # === Summary JSON ===
    delay_acc = float(accuracy_score(actual_delay, delay_pred))
    delay_prec = float(precision_score(actual_delay, delay_pred, zero_division=0))
    delay_rec = float(recall_score(actual_delay, delay_pred, zero_division=0))
    delay_f1 = float(f1_score(actual_delay, delay_pred, zero_division=0))
    delay_roc = float(roc_auc_score(actual_delay, delay_proba))
    cm_delay = confusion_matrix(actual_delay, delay_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_delay.ravel()

    reason_acc = float(accuracy_score(actual_reason, reason_pred_named))
    reason_macro = float(f1_score(actual_reason, reason_pred_named, average="macro", zero_division=0))
    reason_weighted = float(f1_score(actual_reason, reason_pred_named, average="weighted", zero_division=0))
    reason_report = classification_report(
        actual_reason, reason_pred_named, labels=reason_classes,
        zero_division=0, output_dict=True,
    )
    cm_reason = confusion_matrix(actual_reason, reason_pred_named, labels=reason_classes)

    summary = {
        "split": {"test_size": test_size, "random_state": rs},
        "delay_model": {
            "test_rows": int(len(delay_csv)),
            "threshold": delay_threshold,
            "accuracy": round(delay_acc, 4),
            "precision": round(delay_prec, 4),
            "recall": round(delay_rec, 4),
            "f1": round(delay_f1, 4),
            "roc_auc": round(delay_roc, 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
            "predicted_positives": int(delay_pred.sum()),
            "actual_positives": int(actual_delay.sum()),
        },
        "reason_model": {
            "test_rows": int(len(reason_csv)),
            "scope": "только реальные значимые задержки (is_significant_delay=1, reason!='none')",
            "accuracy": round(reason_acc, 4),
            "macro_f1": round(reason_macro, 4),
            "weighted_f1": round(reason_weighted, 4),
            "per_class": {
                cls: {
                    "precision": round(reason_report[cls]["precision"], 4),
                    "recall": round(reason_report[cls]["recall"], 4),
                    "f1": round(reason_report[cls]["f1-score"], 4),
                    "support": int(reason_report[cls]["support"]),
                }
                for cls in reason_classes if cls in reason_report
            },
            "confusion_matrix": {
                "labels": reason_classes,
                "matrix": cm_reason.tolist(),
            },
            "actual_distribution": dict(Counter(actual_reason.tolist())),
            "predicted_distribution": dict(Counter(reason_pred_named.tolist())),
        },
    }
    summary_path = reports_dir / "predictions_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Записано summary: %s", summary_path)

    # Печать
    print("\n=== ИТОГИ НА ТЕСТОВЫХ ВЫБОРКАХ ===\n")
    print(f"DELAY-модель ({len(delay_csv)} рейсов):")
    print(f"  accuracy  = {delay_acc:.4f}")
    print(f"  F1        = {delay_f1:.4f}")
    print(f"  precision = {delay_prec:.4f}")
    print(f"  recall    = {delay_rec:.4f}")
    print(f"  ROC-AUC   = {delay_roc:.4f}")
    print(f"  threshold = {delay_threshold:.2f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}\n")

    print(f"REASON-модель ({len(reason_csv)} рейсов с реальной задержкой):")
    print(f"  accuracy    = {reason_acc:.4f}")
    print(f"  weighted F1 = {reason_weighted:.4f}")
    print(f"  macro F1    = {reason_macro:.4f}")
    for cls, vals in summary["reason_model"]["per_class"].items():
        print(f"    {cls:30s} F1={vals['f1']:.3f}  support={vals['support']}")


if __name__ == "__main__":
    main()
