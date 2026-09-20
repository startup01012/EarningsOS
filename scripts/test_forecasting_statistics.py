from scripts.screen_pretrained_models import _dependence_lag


def test_dependence_lag_for_non_overlapping_windows():
    assert _dependence_lag(1, 5) == 0
    assert _dependence_lag(5, 5) == 0


def test_dependence_lag_for_overlapping_windows():
    assert _dependence_lag(10, 5) == 1
    assert _dependence_lag(11, 5) == 2
