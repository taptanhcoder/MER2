from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit K-fold CV splits for leakage, label balance, and coverage."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cv/kfold_vnemos_mer5.yaml"),
        help="K-fold config YAML.",
    )
    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=None,
        help="Override CV split directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override audit output directory.",
    )
    return parser


def _resolve(path: str | Path) -> Path:
    return resolve_project_path(path, start=PROJECT_ROOT)


def _read_config(path: Path) -> dict[str, Any]:
    cfg_path = _resolve(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"CV config not found: {cfg_path}")
    return load_yaml(cfg_path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _read_original_dataset(cv_cfg: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for item in cv_cfg.get("input_csvs", []):
        path = _resolve(item)
        frame = _read_csv(path)
        frame["_source_csv"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError("cv.input_csvs is empty; cannot audit original dataset coverage.")

    merged = pd.concat(frames, ignore_index=True)
    return merged


def _fold_dirs(cv_dir: Path) -> list[Path]:
    folds = sorted([p for p in cv_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not folds:
        raise RuntimeError(f"No fold_* directories found under {cv_dir}")
    return folds


def _label_counts(frame: pd.DataFrame, label_col: str) -> dict[str, int]:
    if label_col not in frame.columns:
        return {}
    counts = frame[label_col].astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _numeric_stats(frame: pd.DataFrame, col: str, prefix: str) -> dict[str, float]:
    if col not in frame.columns:
        return {}

    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return {}

    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _text_len_stats(frame: pd.DataFrame, text_col: str | None) -> dict[str, float]:
    if not text_col or text_col not in frame.columns:
        return {}

    lengths = frame[text_col].fillna("").astype(str).map(lambda x: len(x.split()))
    return {
        "word_len_mean": float(lengths.mean()),
        "word_len_std": float(lengths.std(ddof=0)),
        "word_len_min": float(lengths.min()),
        "word_len_max": float(lengths.max()),
    }


def _split_stats(
    frame: pd.DataFrame,
    fold: str,
    split: str,
    label_col: str,
    duration_col: str | None,
    text_col: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold,
        "split": split,
        "num_samples": int(len(frame)),
    }

    for label, count in _label_counts(frame, label_col).items():
        row[f"label_{label}"] = int(count)

    if duration_col:
        row.update(_numeric_stats(frame, duration_col, "duration_sec"))

    row.update(_text_len_stats(frame, text_col))
    return row


def _non_empty_values(frame: pd.DataFrame, col: str) -> set[str]:
    if col not in frame.columns:
        return set()
    series = frame[col].dropna().astype(str)
    series = series[series != ""]
    return set(series.tolist())


def _check_within_fold_leakage(
    fold: str,
    split_frames: dict[str, pd.DataFrame],
    leakage_cols: list[str],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for col in leakage_cols:
        values = {split: _non_empty_values(frame, col) for split, frame in split_frames.items()}
        for left, right in [("train", "valid"), ("train", "test"), ("valid", "test")]:
            overlap = values[left] & values[right]
            if overlap:
                issues.append(
                    {
                        "type": "within_fold_overlap",
                        "fold": fold,
                        "column": col,
                        "left": left,
                        "right": right,
                        "num_overlap": int(len(overlap)),
                        "examples": sorted(list(overlap))[:20],
                    }
                )

    return issues


def _check_test_coverage(
    original: pd.DataFrame,
    fold_test_frames: dict[str, pd.DataFrame],
    id_col: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    issues: list[dict[str, Any]] = []

    if id_col not in original.columns:
        raise ValueError(f"Original dataset missing id_col={id_col!r}")

    original_ids = original[id_col].astype(str).tolist()
    original_id_set = set(original_ids)

    rows = []
    seen: dict[str, list[str]] = {}

    for fold, frame in fold_test_frames.items():
        if id_col not in frame.columns:
            raise ValueError(f"{fold}/test.csv missing id_col={id_col!r}")

        for sample_id in frame[id_col].astype(str).tolist():
            seen.setdefault(sample_id, []).append(fold)

    for sample_id in sorted(original_id_set):
        folds = seen.get(sample_id, [])
        rows.append(
            {
                "sample_id": sample_id,
                "num_test_appearances": int(len(folds)),
                "test_folds": ",".join(folds),
            }
        )

    duplicate_test = {sid: folds for sid, folds in seen.items() if len(folds) > 1}
    missing_test = sorted([sid for sid in original_id_set if sid not in seen])

    extra_test = sorted([sid for sid in seen if sid not in original_id_set])

    if duplicate_test:
        issues.append(
            {
                "type": "duplicate_test_coverage",
                "column": id_col,
                "num_issues": int(len(duplicate_test)),
                "examples": [
                    {"sample_id": sid, "folds": folds}
                    for sid, folds in list(duplicate_test.items())[:20]
                ],
            }
        )

    if missing_test:
        issues.append(
            {
                "type": "missing_test_coverage",
                "column": id_col,
                "num_issues": int(len(missing_test)),
                "examples": missing_test[:20],
            }
        )

    if extra_test:
        issues.append(
            {
                "type": "extra_test_ids",
                "column": id_col,
                "num_issues": int(len(extra_test)),
                "examples": extra_test[:20],
            }
        )

    coverage = pd.DataFrame(rows)
    return issues, coverage


def _check_expected_counts(
    original: pd.DataFrame,
    audit_cfg: dict[str, Any],
    label_col: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    expected_total = audit_cfg.get("expected_total_samples", None)
    if expected_total is not None and int(expected_total) != len(original):
        issues.append(
            {
                "type": "unexpected_total_samples",
                "expected": int(expected_total),
                "actual": int(len(original)),
            }
        )

    expected_counts = audit_cfg.get("expected_label_counts", {}) or {}
    if expected_counts and label_col in original.columns:
        actual_counts = _label_counts(original, label_col)
        for label, expected in expected_counts.items():
            actual = int(actual_counts.get(str(label), 0))
            if actual != int(expected):
                issues.append(
                    {
                        "type": "unexpected_label_count",
                        "label": str(label),
                        "expected": int(expected),
                        "actual": actual,
                    }
                )

    return issues


def main() -> int:
    args = build_parser().parse_args()
    cfg = _read_config(args.config)

    cv_cfg = cfg["cv"]
    audit_cfg = cfg.get("audit", {})

    cv_dir = _resolve(args.cv_dir or cv_cfg["output_dir"])
    output_dir = _resolve(args.output_dir or audit_cfg.get("output_dir", "outputs/cv/reports/kfold_audit"))
    output_dir.mkdir(parents=True, exist_ok=True)

    id_col = str(cv_cfg.get("id_col", "sample_id"))
    label_col = str(cv_cfg.get("label_col", "label"))
    duration_col = cv_cfg.get("duration_col", None)
    text_col = cv_cfg.get("text_col", None)
    leakage_cols = list(cv_cfg.get("leakage_check_cols", [id_col]))

    original = _read_original_dataset(cv_cfg)
    if id_col not in original.columns:
        raise ValueError(f"Original dataset missing id_col={id_col!r}")

    if original[id_col].duplicated().any():
        dup_ids = original.loc[original[id_col].duplicated(keep=False), id_col].astype(str).unique().tolist()
        raise ValueError(
            f"Original dataset contains duplicate {id_col}: {dup_ids[:20]} "
            f"(total={len(dup_ids)})"
        )

    fold_dirs = _fold_dirs(cv_dir)

    split_stat_rows: list[dict[str, Any]] = []
    leakage_issues: list[dict[str, Any]] = []
    fold_test_frames: dict[str, pd.DataFrame] = {}

    for fold_dir in fold_dirs:
        fold = fold_dir.name
        train = _read_csv(fold_dir / "train.csv")
        valid = _read_csv(fold_dir / "valid.csv")
        test = _read_csv(fold_dir / "test.csv")

        split_frames = {
            "train": train,
            "valid": valid,
            "test": test,
        }

        for split_name, frame in split_frames.items():
            split_stat_rows.append(
                _split_stats(
                    frame=frame,
                    fold=fold,
                    split=split_name,
                    label_col=label_col,
                    duration_col=duration_col,
                    text_col=text_col,
                )
            )

        leakage_issues.extend(
            _check_within_fold_leakage(
                fold=fold,
                split_frames=split_frames,
                leakage_cols=leakage_cols,
            )
        )

        fold_test_frames[fold] = test

    coverage_issues, coverage = _check_test_coverage(
        original=original,
        fold_test_frames=fold_test_frames,
        id_col=id_col,
    )

    expected_count_issues = _check_expected_counts(
        original=original,
        audit_cfg=audit_cfg,
        label_col=label_col,
    )

    all_issues = leakage_issues + coverage_issues + expected_count_issues

    split_stats = pd.DataFrame(split_stat_rows)
    split_stats.to_csv(output_dir / "cv_split_stats.csv", index=False)
    coverage.to_csv(output_dir / "cv_test_coverage.csv", index=False)

    if all_issues:
        pd.DataFrame(
            [
                {
                    **{k: v for k, v in issue.items() if k != "examples"},
                    "examples": json.dumps(issue.get("examples", []), ensure_ascii=False),
                }
                for issue in all_issues
            ]
        ).to_csv(output_dir / "cv_audit_issues.csv", index=False)

    original_summary = {
        "num_samples": int(len(original)),
        "label_distribution": _label_counts(original, label_col),
    }

    if duration_col:
        original_summary.update(_numeric_stats(original, duration_col, "duration_sec"))
    original_summary.update(_text_len_stats(original, text_col))

    report = {
        "cv_dir": str(cv_dir),
        "output_dir": str(output_dir),
        "num_folds": int(len(fold_dirs)),
        "id_col": id_col,
        "label_col": label_col,
        "leakage_check_cols": leakage_cols,
        "original_summary": original_summary,
        "num_issues": int(len(all_issues)),
        "issues": all_issues,
    }

    (output_dir / "cv_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Wrote CV audit report to: {output_dir}")
    print(json.dumps(
        {
            "num_samples": original_summary["num_samples"],
            "label_distribution": original_summary["label_distribution"],
            "num_folds": len(fold_dirs),
            "num_issues": len(all_issues),
        },
        ensure_ascii=False,
        indent=2,
    ))

    if leakage_issues or coverage_issues:
        print("[ERROR] Found leakage or test coverage issues. Do not train CV until fixed.")
        return 1

    if expected_count_issues:
        print("[WARN] Dataset count differs from configured expectation. Review cv_audit_issues.csv.")
        return 0

    print("[OK] CV splits passed leakage and test coverage audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())