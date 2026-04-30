from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import json
from sklearn.metrics import confusion_matrix


def build_confusion_matrix_artifact(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
    class_names: Sequence[str] | None = None,
    normalize: str | None = None,
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return {
        "labels": list(labels) if labels is not None else None,
        "class_names": list(class_names) if class_names is not None else None,
        "normalize": normalize,
        "matrix": matrix.tolist(),
    }


def save_confusion_matrix(artifact: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)