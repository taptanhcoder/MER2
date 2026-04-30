from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import json
from sklearn.metrics import classification_report


def build_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(target_names) if target_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    return report


def save_classification_report(report: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)