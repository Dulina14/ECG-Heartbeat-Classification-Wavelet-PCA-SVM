from src.inspect_dataset import EXPECTED_RECORD_COUNT
from src.inspect_record import (
    BEAT_LENGTH,
    LABEL_MAP,
    SAMPLES_AFTER_R,
    SAMPLES_BEFORE_R,
    choose_lead,
)


def test_beat_window_has_252_samples() -> None:
    assert SAMPLES_BEFORE_R == 90
    assert SAMPLES_AFTER_R == 162
    assert BEAT_LENGTH == 252


def test_paced_annotation_mapping() -> None:
    assert LABEL_MAP["/"] == "P"


def test_choose_lead_prefers_mlii() -> None:
    assert choose_lead(["V5", "MLII"]) == 1


def test_choose_lead_accepts_lead_ii() -> None:
    assert choose_lead(["II", "V1"]) == 0


def test_choose_lead_falls_back_to_first_channel() -> None:
    assert choose_lead(["V1", "V5"]) == 0


def test_official_database_record_count() -> None:
    assert EXPECTED_RECORD_COUNT == 48

