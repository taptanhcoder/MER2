from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare stratified/group-aware K-fold splits for VNEMOS MER5."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cv/kfold_vnemos_mer5.yaml"),
        help="K-fold CV config YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fold CSVs.",
    )
    return parser


def _resolve(path: str | Path) -> Path:
    return resolve_project_path(path, start=PROJECT_ROOT)


def _read_input_csvs(paths: list[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        csv_path = _resolve(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame["_source_csv"] = str(csv_path)
        frames.append(frame)

    if not frames:
        raise ValueError("cv.input_csvs must contain at least one CSV.")

    merged = pd.concat(frames, ignore_index=True)
    return merged


def _deduplicate_rows(frame: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in frame.columns:
        raise ValueError(f"Missing id_col={id_col!r} in input CSVs.")

    duplicated = frame[id_col].duplicated(keep=False)
    if duplicated.any():
        dup_ids = sorted(frame.loc[duplicated, id_col].astype(str).unique().tolist())
        raise ValueError(
            f"Duplicate sample ids found before K-fold split: {dup_ids[:20]} "
            f"(total={len(dup_ids)}). Fix duplicated rows before CV."
        )

    return frame.reset_index(drop=True)


def _label_distribution(frame: pd.DataFrame, label_col: str) -> dict[str, int]:
    if label_col not in frame.columns:
        return {}
    counts = frame[label_col].astype(str).value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _duration_stats(frame: pd.DataFrame, duration_col: str | None) -> dict[str, float]:
    if not duration_col or duration_col not in frame.columns:
        return {}

    values = pd.to_numeric(frame[duration_col], errors="coerce").dropna()
    if values.empty:
        return {}

    return {
        "duration_mean": float(values.mean()),
        "duration_std": float(values.std(ddof=0)),
        "duration_min": float(values.min()),
        "duration_max": float(values.max()),
    }


def _text_length_stats(frame: pd.DataFrame, text_col: str | None) -> dict[str, float]:
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
    split_name: str,
    label_col: str,
    duration_col: str | None,
    text_col: str | None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "split": split_name,
        "num_samples": int(len(frame)),
        "label_distribution": _label_distribution(frame, label_col),
    }
    stats.update(_duration_stats(frame, duration_col))
    stats.update(_text_length_stats(frame, text_col))
    return stats


def _check_no_overlap(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> list[dict[str, Any]]:
    issues = []
    split_frames = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

    for col in cols:
        if col not in train_df.columns or col not in valid_df.columns or col not in test_df.columns:
            continue

        values = {}
        for split, frame in split_frames.items():
            series = frame[col].dropna().astype(str)
            series = series[series != ""]
            values[split] = set(series.tolist())

        pairs = [
            ("train", "valid"),
            ("train", "test"),
            ("valid", "test"),
        ]
        for left, right in pairs:
            overlap = values[left] & values[right]
            if overlap:
                issues.append(
                    {
                        "column": col,
                        "left": left,
                        "right": right,
                        "num_overlap": int(len(overlap)),
                        "examples": sorted(list(overlap))[:20],
                    }
                )

    return issues


def _inner_train_valid_split(
    pool_df: pd.DataFrame,
    label_col: str,
    group_col: str | None,
    use_group: bool,
    valid_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pool_df[label_col].astype(str).to_numpy()

    # For inner validation, prefer stratified random split.
    # Group-aware inner split is hard to guarantee with small folds;
    # therefore we use a conservative fallback:
    # - if group-aware is requested and groups are available, assign whole groups
    #   greedily while preserving label balance approximately.
    # - otherwise use StratifiedShuffleSplit.
    if use_group and group_col and group_col in pool_df.columns:
        groups = pool_df[group_col].fillna("").astype(str)
        if groups.nunique() < len(pool_df):
            return _group_preserving_inner_split(
                pool_df=pool_df,
                label_col=label_col,
                group_col=group_col,
                valid_ratio=valid_ratio,
                seed=seed,
            )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(valid_ratio),
        random_state=int(seed),
    )
    indices = np.arange(len(pool_df))
    train_idx, valid_idx = next(splitter.split(indices, labels))

    train_df = pool_df.iloc[train_idx].reset_index(drop=True)
    valid_df = pool_df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


def _group_preserving_inner_split(
    pool_df: pd.DataFrame,
    label_col: str,
    group_col: str,
    valid_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))

    groups = pool_df[group_col].fillna("").astype(str).to_numpy()
    labels = pool_df[label_col].astype(str).to_numpy()

    group_frame = (
        pd.DataFrame({"group": groups, "label": labels})
        .groupby("group")
        .agg(
            num_samples=("label", "size"),
            main_label=("label", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )

    target_valid_size = max(1, int(round(len(pool_df) * float(valid_ratio))))
    valid_groups: list[str] = []
    current_size = 0

    # Stratify approximately by sorting groups within labels and sampling.
    for _, label_group in group_frame.groupby("main_label"):
        label_group = label_group.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 10_000_000)),
        )
        label_target = max(
            1,
            int(round(label_group["num_samples"].sum() * float(valid_ratio))),
        )
        label_size = 0
        for _, row in label_group.iterrows():
            if label_size >= label_target:
                break
            valid_groups.append(str(row["group"]))
            label_size += int(row["num_samples"])
            current_size += int(row["num_samples"])

    # If under-filled due to very few labels/groups, add more groups.
    if current_size < target_valid_size:
        remaining = group_frame[~group_frame["group"].astype(str).isin(valid_groups)]
        remaining = remaining.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 10_000_000)),
        )
        for _, row in remaining.iterrows():
            if current_size >= target_valid_size:
                break
            valid_groups.append(str(row["group"]))
            current_size += int(row["num_samples"])

    valid_mask = pool_df[group_col].fillna("").astype(str).isin(set(valid_groups))
    valid_df = pool_df.loc[valid_mask].reset_index(drop=True)
    train_df = pool_df.loc[~valid_mask].reset_index(drop=True)

    if train_df.empty or valid_df.empty:
        raise RuntimeError(
            "Group-preserving inner split produced an empty train or valid split. "
            "Disable cv.use_group or reduce cv.valid_ratio."
        )

    return train_df, valid_df


def _outer_splits(
    frame: pd.DataFrame,
    label_col: str,
    group_col: str | None,
    use_group: bool,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = frame[label_col].astype(str).to_numpy()
    indices = np.arange(len(frame))

    if use_group and group_col and group_col in frame.columns:
        groups = frame[group_col].fillna("").astype(str).to_numpy()
        if StratifiedGroupKFold is None:
            raise RuntimeError(
                "StratifiedGroupKFold is not available in this sklearn version. "
                "Set cv.use_group=false or upgrade scikit-learn."
            )

        splitter = StratifiedGroupKFold(
            n_splits=int(n_splits),
            shuffle=True,
            random_state=int(seed),
        )
        return list(splitter.split(indices, labels, groups))

    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(seed),
    )
    return list(splitter.split(indices, labels))


def _write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. "
            "Pass --overwrite to replace K-fold splits."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    # Do not keep helper source column in generated training CSVs.
    out = frame.copy()
    if "_source_csv" in out.columns:
        out = out.drop(columns=["_source_csv"])

    out.to_csv(path, index=False)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(_resolve(args.config))

    cv_cfg = cfg["cv"]
    seed = int(cv_cfg.get("seed", 42))
    n_splits = int(cv_cfg.get("n_splits", 5))
    valid_ratio = float(cv_cfg.get("valid_ratio", 0.20))

    id_col = str(cv_cfg.get("id_col", "sample_id"))
    label_col = str(cv_cfg.get("label_col", "label"))
    group_col = str(cv_cfg.get("group_col", "group_id"))
    use_group = bool(cv_cfg.get("use_group", True))

    duration_col = cv_cfg.get("duration_col", None)
    text_col = cv_cfg.get("text_col", None)
    leakage_cols = list(cv_cfg.get("leakage_check_cols", [id_col, group_col]))

    output_dir = _resolve(cv_cfg.get("output_dir", "data/processed/vnemos/kfold_mer5"))
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _read_input_csvs(list(cv_cfg["input_csvs"]))
    frame = _deduplicate_rows(frame, id_col=id_col)

    if label_col not in frame.columns:
        raise ValueError(f"Missing label_col={label_col!r} in input CSVs.")

    outer_splits = _outer_splits(
        frame=frame,
        label_col=label_col,
        group_col=group_col,
        use_group=use_group,
        n_splits=n_splits,
        seed=seed,
    )

    all_stats: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []

    for fold_idx, (pool_idx, test_idx) in enumerate(outer_splits):
        fold_name = f"fold_{fold_idx:02d}"
        fold_dir = output_dir / fold_name

        pool_df = frame.iloc[pool_idx].reset_index(drop=True)
        test_df = frame.iloc[test_idx].reset_index(drop=True)

        train_df, valid_df = _inner_train_valid_split(
            pool_df=pool_df,
            label_col=label_col,
            group_col=group_col,
            use_group=use_group,
            valid_ratio=valid_ratio,
            seed=seed + fold_idx,
        )

        issues = _check_no_overlap(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            cols=leakage_cols,
        )
        for issue in issues:
            issue["fold"] = fold_name
        all_issues.extend(issues)

        _write_csv(train_df, fold_dir / "train.csv", overwrite=args.overwrite)
        _write_csv(valid_df, fold_dir / "valid.csv", overwrite=args.overwrite)
        _write_csv(test_df, fold_dir / "test.csv", overwrite=args.overwrite)

        fold_stats = {
            "fold": fold_name,
            "train": _split_stats(
                train_df,
                "train",
                label_col,
                duration_col,
                text_col,
            ),
            "valid": _split_stats(
                valid_df,
                "valid",
                label_col,
                duration_col,
                text_col,
            ),
            "test": _split_stats(
                test_df,
                "test",
                label_col,
                duration_col,
                text_col,
            ),
            "leakage_issues": issues,
        }

        (fold_dir / "fold_stats.json").write_text(
            json.dumps(fold_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_stats.append(fold_stats)

    summary = {
        "config": cv_cfg,
        "num_samples": int(len(frame)),
        "label_distribution": _label_distribution(frame, label_col),
        "folds": all_stats,
        "num_leakage_issues": int(len(all_issues)),
        "leakage_issues": all_issues,
    }

    (output_dir / "kfold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for fold_stats in all_stats:
        for split in ["train", "valid", "test"]:
            split_stats = dict(fold_stats[split])
            row = {
                "fold": fold_stats["fold"],
                "split": split,
                "num_samples": split_stats.pop("num_samples"),
            }
            labels = split_stats.pop("label_distribution", {})
            for label, count in labels.items():
                row[f"label_{label}"] = count
            row.update(split_stats)
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "kfold_stats.csv", index=False)

    if all_issues:
        pd.DataFrame(all_issues).to_csv(output_dir / "kfold_leakage_issues.csv", index=False)
        print(f"[WARN] Found {len(all_issues)} leakage issues. See kfold_leakage_issues.csv")
    else:
        print("[OK] No fold leakage found for configured leakage_check_cols.")

    print(f"[OK] Wrote K-fold splits to: {output_dir}")
    print(json.dumps({"num_samples": len(frame), "n_splits": n_splits}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())