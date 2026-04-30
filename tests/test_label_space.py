from src.data.label_space import (
    MER5_LABEL2ID,
    UIT_VSMEC_NATIVE_LABEL2ID,
    UIT_VSMEC_NATIVE_TO_MER5,
    canonicalize_uit_native_label,
    canonicalize_vnemos_label,
    export_label_map,
)


def test_uit_native_mapping() -> None:
    assert canonicalize_uit_native_label("enjoyment") == "enjoyment"
    assert canonicalize_uit_native_label("other") == "other"
    assert canonicalize_uit_native_label("anger") == "anger"
    assert canonicalize_uit_native_label("disgust") == "disgust"
    assert canonicalize_uit_native_label("surprise") == "surprise"


def test_text_native_to_fusion_mapping() -> None:
    assert UIT_VSMEC_NATIVE_TO_MER5["enjoyment"] == "happiness"
    assert UIT_VSMEC_NATIVE_TO_MER5["other"] == "neutral"
    assert UIT_VSMEC_NATIVE_TO_MER5["disgust"] is None
    assert UIT_VSMEC_NATIVE_TO_MER5["surprise"] is None


def test_vnemos_mapping() -> None:
    assert canonicalize_vnemos_label("angry") == "anger"
    assert canonicalize_vnemos_label("fear") == "fear"
    assert canonicalize_vnemos_label("happiness") == "happiness"


def test_export_label_map() -> None:
    data = export_label_map()
    assert data["text_label2id"] == UIT_VSMEC_NATIVE_LABEL2ID
    assert data["speech_label2id"] == MER5_LABEL2ID