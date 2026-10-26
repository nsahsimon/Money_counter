import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import coin_utils as cu  # noqa: E402


def read_labels():
    with open(os.path.join(ROOT, "model", "labels.txt")) as f:
        pairs = []
        for line in f:
            line = line.strip()
            if line:
                idx, name = line.split(maxsplit=1)
                pairs.append((int(idx), name))
    return pairs


def test_classes_match_labels_file():
    # Each label name starts with its coin value, or is "noise" for value 0.
    for idx, name in read_labels():
        expected = 0 if name == "noise" else int(name.split("_")[0])
        assert cu.value_from_prediction(idx) == expected, name


def test_labels_file_has_one_entry_per_class():
    assert sorted(i for i, _ in read_labels()) == list(range(len(cu.CLASSES)))


@pytest.mark.parametrize("idx,value", [(0, 25), (1, 500), (2, 0), (3, 25), (4, 50), (5, 100)])
def test_value_from_prediction(idx, value):
    assert cu.value_from_prediction(idx) == value


def test_crop_bounds_inside_image():
    assert cu.crop_bounds(50, 60, 10, 200, 200) == (50, 70, 40, 60)


def test_crop_bounds_clamped_at_top_left():
    # Circle near the origin: the old code could go negative and slice wrongly.
    y0, y1, x0, x1 = cu.crop_bounds(5, 3, 10, 200, 200)
    assert (y0, x0) == (0, 0)
    assert y1 == 13 and x1 == 15


def test_crop_bounds_clamped_at_bottom_right():
    y0, y1, x0, x1 = cu.crop_bounds(195, 190, 10, 200, 200)
    assert y1 == 200 and x1 == 200
    assert y0 == 180 and x0 == 185


def test_crop_is_never_empty_for_valid_centre():
    for x, y, r in [(0, 0, 5), (199, 199, 5), (100, 100, 50)]:
        y0, y1, x0, x1 = cu.crop_bounds(x, y, r, 200, 200)
        assert y1 > y0 and x1 > x0


def test_total_ignores_noise():
    assert cu.total_value([25, 0, 500, 0, 100]) == 625


def test_total_empty():
    assert cu.total_value([]) == 0
    assert cu.total_value([0, 0]) == 0


def test_count_by_value_ignores_noise():
    assert cu.count_by_value([25, 0, 25, 500]) == {25: 2, 500: 1}
