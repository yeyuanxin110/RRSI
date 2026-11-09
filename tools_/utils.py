"""Image padding utilities."""

import numpy as np


def expand_to_square(image, fill_value=255):
    """Pad the bottom and right, keeping the input image at the top left."""
    height, width = image.shape[:2]
    side_length = max(height, width)
    if image.ndim == 2:
        square_image = np.full(
            (side_length, side_length), fill_value, dtype=image.dtype,
        )
    else:
        square_image = np.full(
            (side_length, side_length, image.shape[2]),
            fill_value, dtype=image.dtype,
        )
    square_image[:height, :width] = image
    return square_image
