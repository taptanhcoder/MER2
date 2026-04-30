from __future__ import annotations

from importlib import import_module
from pathlib import Path


PACKAGE_MODULES = [
    "src",
    "src.common",
    "src.data",
    "src.text",
    "src.speech",
    "src.fusion",
    "src.evaluation",
    "src.calibration",
    "src.experiments",
    "src.utils",
]


def test_package_imports() -> None:
    for module_name in PACKAGE_MODULES:
        module = import_module(module_name)
        assert module is not None


def test_expected_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    expected_dirs = [
        root / "configs",
        root / "configs" / "dataset",
        root / "configs" / "text",
        root / "configs" / "speech",
        root / "configs" / "fusion",
        root / "configs" / "experiment",
        root / "configs" / "runtime",
        root / "data",
        root / "data" / "UIT-VSMEC",
        root / "data" / "VNEMOS",
        root / "data" / "splits",
        root / "src",
        root / "scripts",
        root / "outputs",
        root / "outputs" / "runs",
        root / "outputs" / "predictions",
        root / "outputs" / "calibration",
        root / "outputs" / "reports",
        root / "notebooks",
        root / "tests",
    ]

    for directory in expected_dirs:
        assert directory.exists(), f"Missing directory: {directory}"


def test_expected_root_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    expected_files = [
        root / "README.md",
        root / "requirements.txt",
        root / ".gitignore",
        root / "scripts" / "prepare_data.py",
        root / "scripts" / "train_text.py",
        root / "scripts" / "train_speech.py",
        root / "scripts" / "calibrate.py",
        root / "scripts" / "run_fusion.py",
        root / "scripts" / "run_benchmark.py",
        root / "scripts" / "summarize_results.py",
    ]

    for file_path in expected_files:
        assert file_path.exists(), f"Missing file: {file_path}"