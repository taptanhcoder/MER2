from __future__ import annotations

from dataclasses import dataclass
from typing import Final


def _norm_label(value: str) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True)
class LabelSpaceSpec:
    name: str
    labels: list[str]
    source_mappings: dict[str, dict[str, str | None]]

    @property
    def label2id(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id2label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    def mapping_for(self, dataset_name: str) -> dict[str, str | None]:
        key = _norm_label(dataset_name)
        if key not in self.source_mappings:
            raise KeyError(
                f"Label space '{self.name}' does not define mapping for dataset '{dataset_name}'. "
                f"Available datasets: {sorted(self.source_mappings)}"
            )
        return self.source_mappings[key]

    def to_export(self) -> dict[str, object]:
        return {
            "target_label_space": self.name,
            "labels": self.labels,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "source_mappings": self.source_mappings,
        }


UIT_VSMEC7_LABELS: Final[list[str]] = [
    "anger",
    "fear",
    "enjoyment",
    "sadness",
    "disgust",
    "surprise",
    "other",
]

MER5_LABELS: Final[list[str]] = [
    "anger",
    "fear",
    "happiness",
    "sadness",
    "neutral",
]

PAPER6_PROXY_LABELS: Final[list[str]] = [
    "anger",
    "fear",
    "enjoyment",
    "sadness",
    "surprise",
    "neutral",
]

UIT_VSMEC_TO_MER5: Final[dict[str, str | None]] = {
    "anger": "anger",
    "fear": "fear",
    "sadness": "sadness",
    "enjoyment": "happiness",
    "other": "neutral",
    "disgust": None,
    "surprise": None,
}

VNEMOS_TO_MER5: Final[dict[str, str]] = {
    "angry": "anger",
    "anger": "anger",
    "fear": "fear",
    "happiness": "happiness",
    "sadness": "sadness",
    "neutral": "neutral",
}

EXCLUDED_UIT_LABELS: Final[list[str]] = ["disgust", "surprise"]

LABEL_SPACE_REGISTRY: Final[dict[str, LabelSpaceSpec]] = {
    "uit_vsmec7_raw": LabelSpaceSpec(
        name="uit_vsmec7_raw",
        labels=UIT_VSMEC7_LABELS,
        source_mappings={
            "uit_vsmec": {
                "anger": "anger",
                "fear": "fear",
                "enjoyment": "enjoyment",
                "sadness": "sadness",
                "disgust": "disgust",
                "surprise": "surprise",
                "other": "other",
            },
            "vnemos": {
                "angry": "anger",
                "anger": "anger",
                "fear": "fear",
                "happiness": "enjoyment",
                "sadness": "sadness",
                "neutral": "other",
            },
        },
    ),
    "paper6_proxy": LabelSpaceSpec(
        name="paper6_proxy",
        labels=PAPER6_PROXY_LABELS,
        source_mappings={
            "uit_vsmec": {
                "anger": "anger",
                "fear": "fear",
                "enjoyment": "enjoyment",
                "sadness": "sadness",
                "disgust": None,
                "surprise": "surprise",
                "other": "neutral",
            },
            "vnemos": {
                "angry": "anger",
                "anger": "anger",
                "fear": "fear",
                "happiness": "enjoyment",
                "sadness": "sadness",
                "neutral": "neutral",
            },
        },
    ),
    "mer5": LabelSpaceSpec(
        name="mer5",
        labels=MER5_LABELS,
        source_mappings={
            "uit_vsmec": UIT_VSMEC_TO_MER5,
            "vnemos": VNEMOS_TO_MER5,
        },
    ),
    "mer5_overlap": LabelSpaceSpec(
        name="mer5_overlap",
        labels=MER5_LABELS,
        source_mappings={
            "uit_vsmec": UIT_VSMEC_TO_MER5,
            "vnemos": VNEMOS_TO_MER5,
        },
    ),
}


LABEL2ID: Final[dict[str, int]] = LABEL_SPACE_REGISTRY["mer5"].label2id
ID2LABEL: Final[dict[int, str]] = LABEL_SPACE_REGISTRY["mer5"].id2label


def available_label_spaces() -> list[str]:
    return sorted(LABEL_SPACE_REGISTRY.keys())


def get_label_space(name: str) -> LabelSpaceSpec:
    key = _norm_label(name)
    if key not in LABEL_SPACE_REGISTRY:
        raise KeyError(
            f"Unknown label space '{name}'. "
            f"Available: {available_label_spaces()}"
        )
    return LABEL_SPACE_REGISTRY[key]


def infer_label_space_name(config: dict[str, object] | None, default: str = "mer5") -> str:
    if not config:
        return default

    label_space_cfg = config.get("label_space")
    if isinstance(label_space_cfg, dict):
        if "name" in label_space_cfg and label_space_cfg["name"]:
            return str(label_space_cfg["name"])
    if isinstance(label_space_cfg, str) and label_space_cfg:
        return label_space_cfg

    dataset_cfg = config.get("dataset")
    if isinstance(dataset_cfg, dict):
        if "label_space" in dataset_cfg and dataset_cfg["label_space"]:
            return str(dataset_cfg["label_space"])

    return default


def canonicalize_label(
    label: str,
    source_dataset: str,
    target_label_space: str,
) -> str | None:
    spec = get_label_space(target_label_space)
    mapping = spec.mapping_for(source_dataset)
    key = _norm_label(label)
    if key not in mapping:
        raise KeyError(
            f"Unknown label '{label}' for source dataset '{source_dataset}' "
            f"under target label space '{target_label_space}'. "
            f"Known labels: {sorted(mapping)}"
        )
    return mapping[key]


def canonicalize_uit_label(label: str, target_label_space: str = "mer5") -> str | None:
    return canonicalize_label(
        label=label,
        source_dataset="uit_vsmec",
        target_label_space=target_label_space,
    )


def canonicalize_vnemos_label(label: str, target_label_space: str = "mer5") -> str | None:
    return canonicalize_label(
        label=label,
        source_dataset="vnemos",
        target_label_space=target_label_space,
    )


def label_to_id(label: str, target_label_space: str = "mer5") -> int:
    spec = get_label_space(target_label_space)
    key = str(label)
    if key not in spec.label2id:
        raise KeyError(
            f"Unknown label '{label}' in label space '{target_label_space}'. "
            f"Available: {spec.labels}"
        )
    return spec.label2id[key]


def id_to_label(label_id: int, target_label_space: str = "mer5") -> str:
    spec = get_label_space(target_label_space)
    idx = int(label_id)
    if idx not in spec.id2label:
        raise KeyError(
            f"Unknown label id '{label_id}' in label space '{target_label_space}'. "
            f"Available ids: {sorted(spec.id2label)}"
        )
    return spec.id2label[idx]


def export_label_map(target_label_space: str = "mer5") -> dict[str, object]:
    return get_label_space(target_label_space).to_export()