from __future__ import annotations

from pathlib import Path


def resolve_project_path(path: str | Path, start: str | Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value

    if start is None:
        base = Path.cwd()
    else:
        base = Path(start)
        if base.is_file():
            base = base.parent

    return (base / value).resolve()