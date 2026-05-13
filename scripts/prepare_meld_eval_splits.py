# scripts/prepare_meld_eval_splits.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


LABELS = ["anger", "fear", "happiness", "sadness", "neutral"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
EXPECTED_SPLITS = ["train", "valid", "test"]
CANONICAL_SOURCE = "meld_vi_text_original_meld_audio"


def _resolve_project_path(path: str | Path, project_root: str | Path = ".") -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root).resolve() / path


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_col = _first_existing_column(
        df,
        ["text_for_model", "transcript_final", "text_vi", "normalized_text", "text"],
    )
    if text_col is None:
        raise KeyError(
            "No usable Vietnamese text column found. Expected one of: "
            "text_for_model, transcript_final, text_vi, normalized_text, text."
        )

    df["text"] = df[text_col].fillna("").astype(str)
    df["normalized_text"] = df["text"].fillna("").astype(str)
    df["transcript_final"] = df["text"].fillna("").astype(str)
    df["text_for_model"] = df["text"].fillna("").astype(str)

    if "raw_text" not in df.columns:
        raw_col = _first_existing_column(df, ["text_vi", "transcript_final", "text"])
        df["raw_text"] = df[raw_col].fillna("").astype(str) if raw_col else df["text"]

    return df


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        raise KeyError("Missing required column: label")

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(
            "Unknown labels found in MELD metadata after preprocessing: "
            f"{unknown}. Expected labels: {LABELS}. "
            "The MELD subset must already map joy->happiness and remove disgust/surprise."
        )

    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)
    return df


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        if "utterance_id" in df.columns:
            df["sample_id"] = df["utterance_id"].astype(str)
        else:
            raise KeyError("Missing sample_id and utterance_id; one is required.")

    df["sample_id"] = df["sample_id"].astype(str)

    if "utterance_id" not in df.columns:
        df["utterance_id"] = df["sample_id"]

    if "group_id" not in df.columns:
        if "dialogue_id" in df.columns:
            df["group_id"] = df["dialogue_id"].astype(str)
        else:
            # Fallback: derive from sample id by removing final utterance suffix when possible.
            df["group_id"] = df["sample_id"].str.replace(r"_u\d+$", "", regex=True)

    if "speaker" not in df.columns:
        df["speaker"] = ""

    return df


def _normalize_audio_paths(df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    if "audio_path" not in df.columns:
        raise KeyError("Missing required column: audio_path")

    def normalize_one(value: Any) -> str:
        raw = str(value).strip()
        if not raw:
            return raw
        path = Path(raw)
        if path.is_absolute():
            return str(path)
        return str((project_root / path).resolve())

    df["audio_path"] = df["audio_path"].map(normalize_one)

    if "audio_relpath" not in df.columns:
        df["audio_relpath"] = df["audio_path"].map(
            lambda p: str(Path(p).relative_to(project_root)) if str(p).startswith(str(project_root)) else str(p)
        )

    if "duration" not in df.columns:
        if "duration_sec" in df.columns:
            df["duration"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0.0)
        else:
            df["duration"] = 0.0

    return df


def _validate_audio_exists(df: pd.DataFrame) -> dict[str, Any]:
    exists = df["audio_path"].map(lambda p: Path(str(p)).exists())
    missing = df.loc[~exists, ["sample_id", "split", "label", "audio_path"]].copy()

    report = {
        "num_rows": int(len(df)),
        "num_audio_missing": int((~exists).sum()),
        "missing_audio_examples": missing.head(20).to_dict(orient="records"),
    }

    if report["num_audio_missing"] > 0:
        raise FileNotFoundError(
            f"{report['num_audio_missing']} audio files are missing. "
            "See missing_audio_examples in the audit report."
        )

    return report


def _validate_source_setting(
    df: pd.DataFrame,
    *,
    require_original_audio: bool,
    fix_source: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {}

    if "source" not in df.columns:
        df["source"] = CANONICAL_SOURCE

    source_counts_before = df["source"].value_counts(dropna=False).to_dict()
    report["source_counts_before"] = {str(k): int(v) for k, v in source_counts_before.items()}

    # Some generated metadata may keep a stale source string even when audio_relpath points
    # to original_meld_audio_wav. Do not fail only because of source text; verify the path evidence.
    if require_original_audio:
        evidence_cols = []
        if "audio_relpath" in df.columns:
            evidence_cols.append(df["audio_relpath"].fillna("").astype(str))
        if "original_audio_source" in df.columns:
            evidence_cols.append(df["original_audio_source"].fillna("").astype(str))
        if "audio_path" in df.columns:
            evidence_cols.append(df["audio_path"].fillna("").astype(str))

        if not evidence_cols:
            raise ValueError(
                "require_original_audio=true but no audio_relpath/original_audio_source/audio_path "
                "columns are available for validation."
            )

        evidence = evidence_cols[0]
        for col in evidence_cols[1:]:
            evidence = evidence + " " + col

        looks_original = evidence.str.contains(
            "original_meld_audio|meld-dataset|MELD.Raw|train_splits|dev_splits|output_repeated_splits_test",
            case=False,
            regex=True,
            na=False,
        )

        report["num_rows_without_original_audio_evidence"] = int((~looks_original).sum())
        if int((~looks_original).sum()) > 0:
            bad = df.loc[~looks_original, ["sample_id", "split", "label", "audio_path"]].head(20)
            report["non_original_audio_examples"] = bad.to_dict(orient="records")
            raise ValueError(
                "Some rows do not contain evidence that audio comes from original MELD audio. "
                "This setting must use translated Vietnamese text + original English audio. "
                "See non_original_audio_examples in the audit report."
            )

    if fix_source:
        df["source"] = CANONICAL_SOURCE

    source_counts_after = df["source"].value_counts(dropna=False).to_dict()
    report["source_counts_after"] = {str(k): int(v) for k, v in source_counts_after.items()}
    report["canonical_source"] = CANONICAL_SOURCE
    report["require_original_audio"] = bool(require_original_audio)

    return df, report


def _validate_splits(df: pd.DataFrame) -> dict[str, Any]:
    if "split" not in df.columns:
        raise KeyError("Missing required column: split")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    unknown_splits = sorted(set(df["split"]) - set(EXPECTED_SPLITS))
    if unknown_splits:
        raise ValueError(f"Unknown splits found: {unknown_splits}. Expected: {EXPECTED_SPLITS}")

    split_counts = df["split"].value_counts().reindex(EXPECTED_SPLITS, fill_value=0)
    if (split_counts == 0).any():
        missing = split_counts[split_counts == 0].index.tolist()
        raise ValueError(f"Missing split rows for: {missing}")

    duplicate_sample_ids = df["sample_id"].duplicated().sum()
    if duplicate_sample_ids:
        dupes = df.loc[df["sample_id"].duplicated(keep=False), ["sample_id", "split", "label"]].head(20)
        raise ValueError(f"Duplicate sample_id values found. Examples: {dupes.to_dict(orient='records')}")

    report: dict[str, Any] = {
        "split_counts": {str(k): int(v) for k, v in split_counts.to_dict().items()},
        "duplicate_sample_ids": int(duplicate_sample_ids),
    }

    for col in ["sample_id", "audio_path"]:
        overlap_pairs = []
        for i, split_a in enumerate(EXPECTED_SPLITS):
            set_a = set(df.loc[df["split"] == split_a, col].astype(str))
            for split_b in EXPECTED_SPLITS[i + 1 :]:
                set_b = set(df.loc[df["split"] == split_b, col].astype(str))
                inter = set_a.intersection(set_b)
                if inter:
                    overlap_pairs.append(
                        {
                            "column": col,
                            "split_a": split_a,
                            "split_b": split_b,
                            "num_overlap": len(inter),
                            "examples": sorted(list(inter))[:20],
                        }
                    )
        if overlap_pairs:
            raise ValueError(f"Cross-split leakage detected for {col}: {overlap_pairs}")
        report[f"{col}_cross_split_overlap"] = 0

    # group_id overlap is reported but not always fatal for MELD if official dialogue split is used.
    group_overlap_pairs = []
    if "group_id" in df.columns:
        for i, split_a in enumerate(EXPECTED_SPLITS):
            set_a = set(df.loc[df["split"] == split_a, "group_id"].astype(str))
            for split_b in EXPECTED_SPLITS[i + 1 :]:
                set_b = set(df.loc[df["split"] == split_b, "group_id"].astype(str))
                inter = set_a.intersection(set_b)
                if inter:
                    group_overlap_pairs.append(
                        {
                            "split_a": split_a,
                            "split_b": split_b,
                            "num_overlap": len(inter),
                            "examples": sorted(list(inter))[:20],
                        }
                    )
    report["group_id_cross_split_overlap"] = group_overlap_pairs

    return report


def _write_split_stats(df: pd.DataFrame, path: Path) -> None:
    label_stats = (
        pd.crosstab(df["split"], df["label"])
        .reindex(index=EXPECTED_SPLITS, fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    rows = []
    for split in EXPECTED_SPLITS:
        part = df[df["split"] == split]
        row = {
            "split": split,
            "num_samples": int(len(part)),
        }
        for label in LABELS:
            row[f"label_{label}"] = int(label_stats.loc[split, label])
        row["duration_mean"] = float(pd.to_numeric(part["duration"], errors="coerce").fillna(0.0).mean())
        row["duration_sum"] = float(pd.to_numeric(part["duration"], errors="coerce").fillna(0.0).sum())
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "sample_id",
        "utterance_id",
        "split",
        "label",
        "label_id",
        "text",
        "raw_text",
        "normalized_text",
        "transcript_final",
        "text_for_model",
        "audio_path",
        "audio_relpath",
        "group_id",
        "speaker",
        "source",
        "text_en",
        "text_vi",
        "duration",
        "duration_sec",
        "sample_rate",
        "original_audio_source",
    ]
    existing = [col for col in preferred if col in df.columns]
    extras = [col for col in df.columns if col not in existing]
    return df[existing + extras]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MELD-ViText-OriginalAudio split CSVs for MER2 evaluation."
    )
    parser.add_argument("--metadata-csv", default="data/MELD/metadata.csv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--prefix", default="meld")
    parser.add_argument("--report-dir", default="outputs/meld/reports/data")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-require-original-audio",
        action="store_true",
        help="Disable original MELD audio evidence validation. Not recommended for paper runs.",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep source column unchanged. By default it is normalized to meld_vi_text_original_meld_audio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    metadata_csv = _resolve_project_path(args.metadata_csv, project_root)
    output_dir = _resolve_project_path(args.output_dir, project_root)
    report_dir = _resolve_project_path(args.report_dir, project_root)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        split: output_dir / f"{args.prefix}_{split}.csv"
        for split in EXPECTED_SPLITS
    }

    if not args.overwrite:
        existing = [str(path) for path in split_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                "Split files already exist. Use --overwrite to replace them: "
                + ", ".join(existing)
            )

    df = pd.read_csv(metadata_csv)
    audit: dict[str, Any] = {
        "metadata_csv": str(metadata_csv),
        "num_rows_input": int(len(df)),
        "columns_input": list(df.columns),
        "label_order": LABELS,
        "setting": "MELD-ViText-OriginalAudio",
    }

    df = _normalize_ids(df)
    df = _normalize_labels(df)
    df = _normalize_text_columns(df)
    df = _normalize_audio_paths(df, project_root)
    df, source_report = _validate_source_setting(
        df,
        require_original_audio=not args.no_require_original_audio,
        fix_source=not args.keep_source,
    )

    audio_report = _validate_audio_exists(df)
    split_report = _validate_splits(df)

    audit.update(source_report)
    audit.update(audio_report)
    audit.update(split_report)
    audit["num_rows_output"] = int(len(df))

    out_df = _select_output_columns(df)

    for split, out_path in split_paths.items():
        part = out_df[out_df["split"] == split].copy().reset_index(drop=True)
        part.to_csv(out_path, index=False, encoding="utf-8")
        audit[f"{split}_csv"] = str(out_path)
        audit[f"{split}_rows"] = int(len(part))

    stats_path = report_dir / f"{args.prefix}_split_stats.csv"
    _write_split_stats(df, stats_path)
    audit["split_stats_csv"] = str(stats_path)

    audit_path = report_dir / f"{args.prefix}_data_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote MELD split CSVs to: {output_dir}")
    print(json.dumps(
        {
            "num_rows": audit["num_rows_output"],
            "splits": audit["split_counts"],
            "stats": str(stats_path),
            "audit": str(audit_path),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())