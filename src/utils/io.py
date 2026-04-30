from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise TypeError(
                    f"Each JSONL row must be a JSON object. "
                    f"Found {type(parsed)} at {file_path}:{line_no}"
                )
            rows.append(parsed)

    return rows


def write_yaml(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    df.to_csv(file_path, index=False)