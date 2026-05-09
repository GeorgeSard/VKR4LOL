"""Проверка, на каких данных обучалась модель в конкретном git-коммите.

Сравнивает:
- хеш сырого CSV в этом коммите
- хеш данных, на которых обучалась модель (записан в metadata.json коммита)
- хеш сырого CSV в HEAD (текущее состояние)

Использование:
    docker compose run --rm trainer python scripts/verify_iteration.py <commit_sha>

Пример:
    docker compose run --rm trainer python scripts/verify_iteration.py 5916620
    docker compose run --rm trainer python scripts/verify_iteration.py 92e85cd

Что показывает:
- Если хеш данных в коммите == хеш в metadata модели → модель действительно
  обучалась на тех данных, которые лежат в этом же коммите.
- Если хеш данных коммита != HEAD → данные изменялись после этого коммита.
- Если хеш данных коммита == другого коммита → между ними данные НЕ менялись
  (возможно, менялись только параметры).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from typing import Any


CSV_PATH = "data/raw/flight_delays_ru_synthetic_2023_2025.csv"
DELAY_META = "models/delay_model/metadata.json"
REASON_META = "models/reason_model/metadata.json"
PARAMS_PATH = "params.yaml"


def git_show_bytes(sha: str, path: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", "show", f"{sha}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None


def sha256_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def commit_subject(sha: str) -> str:
    try:
        return subprocess.run(
            ["git", "log", "-1", "--pretty=format:%s", sha],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "<unknown>"


def fmt_metric(meta: dict[str, Any], path: list[str]) -> str:
    cur: Any = meta
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return "—"
        cur = cur[k]
    if isinstance(cur, float):
        return f"{cur:.4f}"
    return str(cur)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sha = sys.argv[1]

    subj = commit_subject(sha)
    print(f"\n=== Коммит {sha} — {subj} ===\n")

    # 1) данные в этом коммите
    data_bytes = git_show_bytes(sha, CSV_PATH)
    if data_bytes is None:
        print(f"❌ В коммите нет файла {CSV_PATH}")
        sys.exit(1)
    data_hash = sha256_short(data_bytes)
    rows = data_bytes.count(b"\n") - 1
    print(f"📊 Сырые данные (data/raw/...csv) в этом коммите:")
    print(f"   sha256_16: {data_hash}")
    print(f"   rows:      {rows}")
    print(f"   size:      {len(data_bytes) / 1e6:.2f} MB")

    # 2) данные сейчас (HEAD)
    head_bytes = git_show_bytes("HEAD", CSV_PATH)
    head_hash = sha256_short(head_bytes) if head_bytes else None

    # 3) первый коммит для базы сравнения
    first_sha = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        capture_output=True, text=True, check=True
    ).stdout.strip()[:7]
    first_bytes = git_show_bytes(first_sha, CSV_PATH)
    first_hash = sha256_short(first_bytes) if first_bytes else None

    # 4) метаданные delay-модели
    delay_meta_bytes = git_show_bytes(sha, DELAY_META)
    if delay_meta_bytes:
        delay_meta = json.loads(delay_meta_bytes)
        td = delay_meta.get("training_data", {})
        meta_data_hash = td.get("raw_sha256_16")
        print(f"\n🤖 Delay-модель в этом коммите:")
        print(f"   trained_at:        {delay_meta.get('trained_at_utc', '—')}")
        print(f"   F1:                {fmt_metric(delay_meta, ['metrics', 'f1'])}")
        print(f"   ROC-AUC:           {fmt_metric(delay_meta, ['metrics', 'roc_auc'])}")
        print(f"   threshold:         {fmt_metric(delay_meta, ['threshold'])}")
        if meta_data_hash:
            verdict = "✓ МАТЧ" if meta_data_hash == data_hash else "✗ НЕ СОВПАДАЕТ"
            print(f"   trained on data:   {meta_data_hash}  ({verdict} с CSV в коммите)")
            print(f"   rows train/test:   {td.get('rows_train')}/{td.get('rows_test')}")
            print(f"   positive в train:  {td.get('positive_share_train')}")
        else:
            print(f"   trained on data:   <не записано> (модель обучена до добавления training_data в metadata)")
    else:
        print(f"\n🤖 В коммите нет {DELAY_META}")

    # 5) метаданные reason-модели
    reason_meta_bytes = git_show_bytes(sha, REASON_META)
    if reason_meta_bytes:
        reason_meta = json.loads(reason_meta_bytes)
        td = reason_meta.get("training_data", {})
        meta_data_hash = td.get("raw_sha256_16")
        print(f"\n🤖 Reason-модель в этом коммите:")
        print(f"   trained_at:        {reason_meta.get('trained_at_utc', '—')}")
        print(f"   macro_F1:          {fmt_metric(reason_meta, ['metrics', 'macro_f1'])}")
        if meta_data_hash:
            verdict = "✓ МАТЧ" if meta_data_hash == data_hash else "✗ НЕ СОВПАДАЕТ"
            print(f"   trained on data:   {meta_data_hash}  ({verdict} с CSV в коммите)")
            print(f"   rows filtered:     {td.get('rows_filtered_for_reason')}")
        else:
            print(f"   trained on data:   <не записано>")

    # 6) params.yaml в этом коммите
    params_bytes = git_show_bytes(sha, PARAMS_PATH)
    if params_bytes:
        params_hash = sha256_short(params_bytes)
        print(f"\n⚙️  params.yaml в этом коммите:")
        print(f"   sha256_16: {params_hash}")

    # 7) финальная сводка
    print(f"\n📋 Сводка по данным:")
    print(f"   Этот коммит ({sha}):       {data_hash}")
    print(f"   Базовый коммит ({first_sha}): {first_hash} {'(данные те же)' if first_hash == data_hash else '(ДАННЫЕ ОТЛИЧАЮТСЯ)'}")
    if head_hash:
        print(f"   HEAD сейчас:                {head_hash} {'(данные те же)' if head_hash == data_hash else '(ДАННЫЕ ОТЛИЧАЮТСЯ)'}")
    print()


if __name__ == "__main__":
    main()
