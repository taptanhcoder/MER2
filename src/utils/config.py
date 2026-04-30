from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.utils.paths import resolve_project_path


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")
    return data


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []

    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            deduped.append(resolved)

    return deduped


def _candidate_remap_paths(reference: str | Path, project_root: str | Path) -> list[Path]:
    ref_str = str(reference).replace("\\", "/")
    basename = Path(reference).name
    candidates: list[Path] = []

    # Legacy compatibility remaps
    if "configs/common/label_space/" in ref_str:
        candidates.extend(
            [
                resolve_project_path(Path("configs/label_space") / basename, start=project_root),
                resolve_project_path(Path("configs/dataset/label_space") / basename, start=project_root),
                resolve_project_path(Path("configs/dataset") / basename, start=project_root),
            ]
        )

    if "configs/common/" in ref_str:
        candidates.extend(
            [
                resolve_project_path(Path("configs") / basename, start=project_root),
                resolve_project_path(Path("configs/runtime") / basename, start=project_root),
                resolve_project_path(Path("configs/dataset") / basename, start=project_root),
            ]
        )

    configs_root = resolve_project_path("configs", start=project_root)
    if configs_root.exists():
        matches = sorted([p for p in configs_root.rglob(basename) if p.is_file()])

        if "label_space" in ref_str:
            preferred = [p for p in matches if "label_space" in str(p).replace("\\", "/")]
            matches = preferred or matches

        candidates.extend(matches)

    candidates = [p for p in candidates if p.exists()]
    return _dedupe_paths(candidates)


def resolve_config_reference(reference: str | Path, project_root: str | Path) -> Path:
    direct = resolve_project_path(reference, start=project_root)
    if direct.exists():
        return direct.resolve()

    candidates = _candidate_remap_paths(reference, project_root)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        pretty = "\n".join(f"- {path}" for path in candidates)
        raise FileNotFoundError(
            "Ambiguous legacy config reference could not be resolved uniquely:\n"
            f"reference={reference}\n"
            f"candidates:\n{pretty}"
        )

    raise FileNotFoundError(
        "Config reference could not be resolved.\n"
        f"reference={reference}\n"
        f"project_root={project_root}"
    )