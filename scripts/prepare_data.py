from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from src.data.manifest_reader import prepare_vnemos_paired_from_existing_splits
from src.data.schemas import export_paired_schema
from src.data.validators import (
    validate_paired_compatibility_splits,
    validate_paired_jsonl_splits,
)
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, write_csv, write_json, write_jsonl, write_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize existing VNEMOS paired split JSONL files into a single "
            "stable benchmark contract for text, speech, and fusion."
        )
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/dataset/prepare_mer5.yaml"),
        help="Path to dataset preparation config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run preparation and validation but skip writing outputs.",
    )
    return parser


def _fail_if_exists(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Output files already exist. Use --force to overwrite. Existing files: "
            + ", ".join(str(p) for p in existing)
        )


def main() -> int:
    args = build_parser().parse_args()

    config = load_yaml(args.dataset_config)
    project_root = Path(config.get("project", {}).get("root", ".")).resolve()

    dataset_cfg = dict(config["dataset"])
    output_cfg = dict(config.get("output", {}))

    transcripts_dir = ensure_dir(project_root / dataset_cfg["transcripts_dir"])
    compat_dir = ensure_dir(project_root / dataset_cfg.get("compatibility_csv_dir", "data/splits"))

    targets = [
        transcripts_dir / "vnemos_mer_all.jsonl",
        transcripts_dir / "train.jsonl",
        transcripts_dir / "valid.jsonl",
        transcripts_dir / "test.jsonl",
        transcripts_dir / "schema.json",
        transcripts_dir / "split_summary.json",
        transcripts_dir / "resolved_prepare_config.yaml",
    ]

    if bool(dataset_cfg.get("write_compatibility_csv", True)):
        targets.extend(
            [
                compat_dir / "vnemos_train.csv",
                compat_dir / "vnemos_valid.csv",
                compat_dir / "vnemos_test.csv",
            ]
        )

    if not args.validate_only:
        _fail_if_exists(targets, args.force)

    prepared = prepare_vnemos_paired_from_existing_splits(config=config)

    validate_paired_jsonl_splits(
        prepared.train_frame,
        prepared.valid_frame,
        prepared.test_frame,
        project_root=project_root,
    )

    if bool(dataset_cfg.get("write_compatibility_csv", True)):
        validate_paired_compatibility_splits(
            prepared.compat_train_frame,
            prepared.compat_valid_frame,
            prepared.compat_test_frame,
            project_root=project_root,
        )

    print("[OK] Validation passed.")
    print("\nPrepared dataset stats:")
    pprint(prepared.stats)

    if args.validate_only:
        print("\n[INFO] validate-only enabled. No files were written.")
        return 0

    if bool(output_cfg.get("write_master_jsonl", True)):
        write_jsonl(
            prepared.all_frame.to_dict(orient="records"),
            transcripts_dir / "vnemos_mer_all.jsonl",
        )

    if bool(output_cfg.get("write_split_jsonl", True)):
        write_jsonl(
            prepared.train_frame.to_dict(orient="records"),
            transcripts_dir / "train.jsonl",
        )
        write_jsonl(
            prepared.valid_frame.to_dict(orient="records"),
            transcripts_dir / "valid.jsonl",
        )
        write_jsonl(
            prepared.test_frame.to_dict(orient="records"),
            transcripts_dir / "test.jsonl",
        )

    if bool(output_cfg.get("write_schema", True)):
        write_json(export_paired_schema(), transcripts_dir / "schema.json")

    if bool(output_cfg.get("write_summary", True)):
        write_json(prepared.stats, transcripts_dir / "split_summary.json")

    if bool(dataset_cfg.get("write_compatibility_csv", True)):
        write_csv(prepared.compat_train_frame, compat_dir / "vnemos_train.csv")
        write_csv(prepared.compat_valid_frame, compat_dir / "vnemos_valid.csv")
        write_csv(prepared.compat_test_frame, compat_dir / "vnemos_test.csv")

    write_yaml(config, transcripts_dir / "resolved_prepare_config.yaml")

    print(f"\n[OK] Wrote canonical paired manifests to: {transcripts_dir}")
    if bool(dataset_cfg.get("write_compatibility_csv", True)):
        print(f"[OK] Wrote compatibility CSVs to: {compat_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())