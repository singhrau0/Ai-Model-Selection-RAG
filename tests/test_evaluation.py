import pytest

from src.evaluation import ndcg, ndcg_for_ids


def test_ndcg_is_one_for_ideal_ranking():
    assert ndcg([3, 2, 1], k=3) == pytest.approx(1.0)


def test_ndcg_penalizes_bad_order():
    assert 0 < ndcg([0, 1, 3], k=3) < 1


def test_ndcg_for_ids_uses_full_ideal_judgments():
    assert ndcg_for_ids(["b", "a"], {"a": 3, "b": 1}, k=2) < 1
