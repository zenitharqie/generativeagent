import pytest

from src.retrieval import NormalizationError, min_max_normalize


def test_min_max_normalization_maps_varied_values_to_zero_one() -> None:
    assert min_max_normalize((2.0, 4.0, 6.0)) == (0.0, 0.5, 1.0)


def test_normalization_preserves_input_order() -> None:
    assert min_max_normalize((10.0, 0.0, 5.0)) == (1.0, 0.0, 0.5)


def test_all_equal_values_normalize_to_one_half() -> None:
    assert min_max_normalize((7.0, 7.0, 7.0)) == (0.5, 0.5, 0.5)


def test_singleton_value_normalizes_to_one_half() -> None:
    assert min_max_normalize((3.0,)) == (0.5,)


def test_empty_values_return_an_empty_tuple() -> None:
    assert min_max_normalize(()) == ()


@pytest.mark.parametrize("invalid", [(True,), ("not-numeric",), (float("inf"),)])
def test_normalization_rejects_invalid_values(invalid: tuple) -> None:
    with pytest.raises(NormalizationError):
        min_max_normalize(invalid)
