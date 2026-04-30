from src.data.manifest_reader import infer_vnemos_group_id, normalize_text


def test_infer_vnemos_group_id_strips_timecode_and_segment() -> None:
    filename = "Copy of SceneA-00.01.02.123-00.01.05.456-seg2.wav"
    assert infer_vnemos_group_id(filename) == "scenea"


def test_normalize_text() -> None:
    text = "  Xin   Chào   \n"
    assert normalize_text(text) == "xin chào"