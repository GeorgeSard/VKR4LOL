"""Сравнение исходного датасета и текущего — для слайдов защиты ВКР.

Извлекает версии CSV из git: baseline (init-коммит) vs current (рабочий tree),
считает статистики и сохраняет:

  reports/data_comparison.md           — таблица «было → стало» в markdown
  reports/figures/data_comparison.png  — график долей причин до/после
  reports/figures/data_positive_share.png — bar chart доли задержек

Запуск:
    python scripts/make_data_comparison.py
"""

from __future__ import annotations

import io
import logging
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data_comparison")

CSV_PATH = "data/raw/flight_delays_ru_synthetic_2023_2025.csv"
OUT_MD = Path("reports/data_comparison.md")
OUT_DIST_PNG = Path("reports/figures/data_comparison.png")
OUT_POS_PNG = Path("reports/figures/data_positive_share.png")

# Baseline — init-коммит, до любых правок данных
BASELINE_COMMIT = "92e85cd"
BASELINE_LABEL = "До работы с данными\n(init-коммит)"
CURRENT_LABEL = "После 4 итераций\n(текущий рабочий tree)"


def read_csv_from_commit(sha: str) -> pd.DataFrame:
    proc = subprocess.run(
        ["git", "show", f"{sha}:{CSV_PATH}"], capture_output=True, check=True
    )
    return pd.read_csv(io.BytesIO(proc.stdout), encoding="utf-8-sig")


def stats(df: pd.DataFrame) -> dict:
    delay_only = df.loc[df["is_significant_delay"] == 1, "delay_minutes"]
    reason_counts = df["delay_reason"].value_counts().to_dict()
    return {
        "rows": len(df),
        "positive_count": int(df["is_significant_delay"].sum()),
        "positive_share": float(df["is_significant_delay"].mean()),
        "mean_delay_when_positive": float(delay_only.mean()) if len(delay_only) else 0.0,
        "median_delay_when_positive": float(delay_only.median()) if len(delay_only) else 0.0,
        "p95_delay_when_positive": float(delay_only.quantile(0.95)) if len(delay_only) else 0.0,
        "reason_counts": {k: int(v) for k, v in reason_counts.items()},
    }


def main() -> None:
    log.info("Читаю baseline из коммита %s", BASELINE_COMMIT)
    df_old = read_csv_from_commit(BASELINE_COMMIT)
    log.info("Читаю current из рабочего tree")
    df_new = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    s_old = stats(df_old)
    s_new = stats(df_new)

    # === markdown-таблица ===
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Сравнение датасета: исходные данные vs после работы\n")
    lines.append(
        f"**Baseline:** init-коммит `{BASELINE_COMMIT}` (синтетический генератор как есть)\n"
    )
    lines.append("**Current:** после 4 итераций усиления операционных правил\n")
    lines.append("")
    lines.append("## Общие показатели\n")
    lines.append("| Метрика | Исходные данные | После работы | Изменение |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Всего рейсов | {s_old['rows']:,} | {s_new['rows']:,} | {s_new['rows'] - s_old['rows']:+,} |"
    )
    lines.append(
        f"| Рейсов со значимой задержкой | {s_old['positive_count']:,} | {s_new['positive_count']:,} | "
        f"**+{s_new['positive_count'] - s_old['positive_count']:,}** |"
    )
    lines.append(
        f"| Доля значимых задержек | {s_old['positive_share']*100:.2f}% | {s_new['positive_share']*100:.2f}% | "
        f"**+{(s_new['positive_share']-s_old['positive_share'])*100:.2f} п.п.** |"
    )
    lines.append(
        f"| Средняя задержка (мин) | {s_old['mean_delay_when_positive']:.1f} | {s_new['mean_delay_when_positive']:.1f} | "
        f"{s_new['mean_delay_when_positive']-s_old['mean_delay_when_positive']:+.1f} |"
    )
    lines.append(
        f"| Медиана задержки (мин) | {s_old['median_delay_when_positive']:.0f} | {s_new['median_delay_when_positive']:.0f} | "
        f"{s_new['median_delay_when_positive']-s_old['median_delay_when_positive']:+.0f} |"
    )

    lines.append("\n## Распределение причин задержки\n")
    lines.append("| Причина | Исходные данные | После работы | Изменение |")
    lines.append("|---|---:|---:|---:|")
    all_reasons = sorted(set(s_old["reason_counts"]) | set(s_new["reason_counts"]))
    for r in all_reasons:
        old_c = s_old["reason_counts"].get(r, 0)
        new_c = s_new["reason_counts"].get(r, 0)
        diff = new_c - old_c
        lines.append(f"| `{r}` | {old_c:,} | {new_c:,} | {diff:+,} |")

    lines.append("\n## Что менялось (по итерациям)\n")
    lines.append("| Итерация | Скрипт | Логика |")
    lines.append("|---|---|---|")
    lines.append(
        "| iter1 | `scripts/enhance_data.py` | "
        "70% рейсов с плохой погодой+низкой видимостью → задержка `weather`. "
        "50% рейсов с критической загрузкой → `airport_congestion`. "
        "60% рейсов с задержкой предыдущего >60 мин → `late_aircraft` |"
    )
    lines.append(
        "| iter2 | `scripts/enhance_data_v2.py` | "
        "Усиление трёх правил выше до 80-88% детерминированности |"
    )
    lines.append(
        "| iter3 | `scripts/enhance_data_v3.py` | "
        "Дальнейшее усиление до 92-95%, без новых правил |"
    )
    lines.append(
        "| iter4 | `scripts/enhance_data_v4.py` | "
        "Расширение охвата плохой погоды + 2 новых правила: "
        "технологическая проверка перед вылетом при плохой погоде, "
        "смена экипажа в нерабочие часы |"
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Сохранено: %s", OUT_MD)

    # === Визуализация 1: распределение причин ===
    OUT_DIST_PNG.parent.mkdir(parents=True, exist_ok=True)

    excluded = {"none"}
    reasons = [r for r in all_reasons if r not in excluded]
    old_vals = [s_old["reason_counts"].get(r, 0) for r in reasons]
    new_vals = [s_new["reason_counts"].get(r, 0) for r in reasons]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(reasons))
    width = 0.4
    bars_old = ax.bar(
        x - width / 2, old_vals, width, label="Исходные данные", color="#888888"
    )
    bars_new = ax.bar(
        x + width / 2, new_vals, width, label="После 4 итераций", color="#1f77b4"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=20, ha="right")
    ax.set_ylabel("Количество рейсов")
    ax.set_title(
        "Распределение причин задержки: исходные данные vs после работы",
        fontsize=12,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (bars_old, bars_new):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{int(h):,}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(OUT_DIST_PNG, dpi=120)
    plt.close(fig)
    log.info("Сохранено: %s", OUT_DIST_PNG)

    # === Визуализация 2: доля задержек ===
    OUT_POS_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [BASELINE_LABEL, CURRENT_LABEL]
    shares = [s_old["positive_share"] * 100, s_new["positive_share"] * 100]
    counts = [s_old["positive_count"], s_new["positive_count"]]
    colors = ["#888888", "#1f77b4"]
    bars = ax.bar(labels, shares, color=colors, width=0.55)
    ax.set_ylabel("Доля значимых задержек, %")
    ax.set_title(
        "Доля рейсов со значимой задержкой: до и после работы с данными",
        fontsize=11,
    )
    ax.set_ylim(0, max(shares) * 1.25)
    for bar, share, cnt in zip(bars, shares, counts):
        ax.annotate(
            f"{share:.2f}%\n({cnt:,} рейсов)",
            xy=(bar.get_x() + bar.get_width() / 2, share),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_POS_PNG, dpi=120)
    plt.close(fig)
    log.info("Сохранено: %s", OUT_POS_PNG)

    # Печать в консоль для скриншота терминала
    print("\n" + "=" * 80)
    print("Файлы для скриншота:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_DIST_PNG}")
    print(f"  {OUT_POS_PNG}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
