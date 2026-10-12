import numpy as np

from src.create_splits import (
    PAPER_RECORD_TEST_FOLDS,
    create_beat_folds,
    create_paper_record_holdouts,
    validate_fold_indices,
)


def test_beat_folds_form_one_disjoint_test_partition() -> None:
    labels = np.repeat(np.array(["A", "L", "N", "P", "R", "V"]), 20)
    folds = create_beat_folds(labels, seed=0)

    validate_fold_indices(folds, len(labels))

    test_frequency = np.zeros(len(labels), dtype=int)
    for train, test in folds:
        assert np.intersect1d(train, test).size == 0
        test_frequency[test] += 1

    assert len(folds) == 10
    assert np.all(test_frequency == 1)


def test_paper_record_holdouts_have_no_within_fold_record_leakage() -> None:
    paper_records = sorted({record for fold in PAPER_RECORD_TEST_FOLDS for record in fold})
    record_ids = np.repeat(np.array(paper_records + ["999"]), 2)
    folds = create_paper_record_holdouts(record_ids)

    validate_fold_indices(folds, len(record_ids))

    test_frequency = np.zeros(len(record_ids), dtype=int)
    for train, test in folds:
        assert set(record_ids[train]).isdisjoint(set(record_ids[test]))
        test_frequency[test] += 1

    assert len(folds) == 10
    assert test_frequency.max() > 1
    assert np.any(test_frequency == 0)
