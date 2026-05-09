"""Сравнение версий датасета по git history.

Извлекает CSV из каждого указанного git-коммита, считает статистику и сохраняет
сводную таблицу в `reports/history/data_evolution.json` + печатает в консоль.

Использование:

    python scripts/show_data_evolution.py

Зачем нужно: показать, что между итерациями реально менялись данные. DVC-метрика
`reports/metrics/data_stats.json` показывает только текущее состояние —
этот скрипт восстанавливает прошлые состояния из git.
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data_evolution")

CSV_PATH = "data/raw/flight_delays_ru_synthetic_2023_2025.csv"
OUT = Path("reports/history/data_evolution.json")


def _git_log_iterations() -> list[tuple[str, str]]:
    """Возвращает список (sha, subject) коммитов в хронологическом порядке."""
    result = subprocess.run(
        ["git", "log", "--reverse", "--pretty=format:%H|%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    out: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        sha, _, subj = line.partition("|")
        out.append((sha[:7], subj))
    return out


def _read_csv_from_commit(sha: str, path: str) -> pd.DataFrame:
    """git show <sha>:<path> → DataFrame."""
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        capture_output=True,
        check=True,
    )
    return pd.read_csv(io.BytesIO(result.stdout), encoding="utf-8-sig")


def _stats(df: pd.DataFrame) -> dict:
    delay_only = df.loc[df["is_significant_delay"] == 1, "delay_minutes"]
    return {
        "rows": len(df),
        "positive_count": int(df["is_significant_delay"].sum()),
        "positive_share": round(float(df["is_significant_delay"].mean()), 6),
        "delay_reason_counts": {
            str(k): int(v) for k, v in df["delay_reason"].value_counts().items()
        },
        "mean_delay_when_significant": round(float(delay_only.mean()), 2)
        if len(delay_only)
        else None,
    }


def main() -> None:
    log.info("Собираю историю коммитов")
    commits = _git_log_iterations()
    log.info("Найдено коммитов: %d", len(commits))

    evolution: list[dict] = []
    seen: set[str] = set()
    for sha, subj in commits:
        try:
            df = _read_csv_from_commit(sha, CSV_PATH)
        except subprocess.CalledProcessError:
            log.warning("В коммите %s нет файла %s — пропускаю", sha, CSV_PATH)
            continue
        s = _stats(df)
        # пропустить коммиты, где данные не менялись (по содержимому)
        fingerprint = f"{s['rows']}_{s['positive_count']}_{s['mean_delay_when_significant']}"
        if fingerprint in seen:
            log.info("Коммит %s — данные те же, пропускаю в выдаче", sha)
            continue
        seen.add(fingerprint)
        evolution.append({"commit": sha, "subject": subj, "stats": s})
        log.info(
            "%s — rows=%d, positive=%.2f%%, mean_delay=%s",
            sha,
            s["rows"],
            s["positive_share"] * 100,
            s["mean_delay_when_significant"],
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(evolution, f, ensure_ascii=False, indent=2)
    log.info("Сохранено: %s (%d уникальных версий датасета)", OUT, len(evolution))

    # Печать таблицы для копирования в отчёт
    print("\n=== Эволюция датасета по git-истории ===\n")
    print(f"{'commit':<8} {'rows':>7} {'positive':>10} {'mean_delay':>10}  subject")
    print("-" * 90)
    for it in evolution:
        s = it["stats"]
        print(
            f"{it['commit']:<8} {s['rows']:>7} {s['positive_share']*100:>9.2f}% "
            f"{s['mean_delay_when_significant']:>10}  {it['subject'][:50]}"
        )


if __name__ == "__main__":
    main()
