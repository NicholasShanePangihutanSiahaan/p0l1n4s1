"""Unit tests for palm-oil flower label matching."""

from types import SimpleNamespace

from beehive_drone.flower_buzzer import contains_flower, normalize_label


FLOWER_LABELS = {
    'male-flowers',
    'pre-receptive-female',
    'receptive-female',
    'post-receptive-female',
}


def detection(label, confidence=90.0):
    return SimpleNamespace(label=label, confidence=confidence)


def test_all_four_flower_classes_are_detected():
    for label in FLOWER_LABELS:
        assert contains_flower([detection(label)], FLOWER_LABELS, 25.0)


def test_non_flower_and_low_confidence_are_ignored():
    assert not contains_flower(
        [detection('pohon'), detection('male-flowers', 20.0)],
        FLOWER_LABELS,
        25.0,
    )


def test_label_is_normalized():
    assert normalize_label('  Receptive-Female ') == 'receptive-female'
