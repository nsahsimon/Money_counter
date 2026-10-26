"""Pure helper logic for the money counter.

These functions contain no model or display code, so they can be tested
without TensorFlow or a GUI. Values follow the order in model/labels.txt.
"""

# Index matches the model output; order must match model/labels.txt.
CLASSES = [25, 500, 0, 25, 50, 100]
NOISE_VALUE = 0


def value_from_prediction(index):
    """Map a predicted class index to a coin value in FCFA (0 means noise)."""
    return CLASSES[index]


def crop_bounds(x, y, r, width, height):
    """Return (y0, y1, x0, x1) for a square crop around a circle, clamped to the image."""
    y0 = max(0, y - r)
    y1 = min(height, y + r)
    x0 = max(0, x - r)
    x1 = min(width, x + r)
    return y0, y1, x0, x1


def total_value(values):
    """Sum coin values, ignoring noise detections."""
    return sum(v for v in values if v != NOISE_VALUE)


def count_by_value(values):
    """Count detections per coin value, ignoring noise."""
    counts = {}
    for v in values:
        if v != NOISE_VALUE:
            counts[v] = counts.get(v, 0) + 1
    return counts
