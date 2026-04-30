"""Reusable classical computer-vision algorithms."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color, img_as_float


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to float grayscale."""
    image = img_as_float(image)
    if image.ndim == 3:
        return color.rgb2gray(image)
    return image.astype(float, copy=False)


def non_maximum_suppression(response: np.ndarray, min_distance: int = 8) -> np.ndarray:
    """Return a boolean mask of local maxima in a response image."""
    if min_distance < 1:
        raise ValueError("min_distance must be at least 1")

    footprint = 2 * min_distance + 1
    local_max = response == ndimage.maximum_filter(response, size=footprint, mode="constant")
    return local_max & (response > 0)


def harris_corners(
    image: np.ndarray,
    sigma: float = 1.2,
    k: float = 0.04,
    threshold_rel: float = 0.01,
    min_distance: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect Harris corners and return coordinates plus the Harris response."""
    gray = to_grayscale(image)

    ix = ndimage.sobel(gray, axis=1, mode="reflect")
    iy = ndimage.sobel(gray, axis=0, mode="reflect")

    ixx = ndimage.gaussian_filter(ix * ix, sigma=sigma)
    iyy = ndimage.gaussian_filter(iy * iy, sigma=sigma)
    ixy = ndimage.gaussian_filter(ix * iy, sigma=sigma)

    determinant = ixx * iyy - ixy**2
    trace = ixx + iyy
    response = determinant - k * trace**2

    threshold = threshold_rel * float(response.max()) if response.size else 0.0
    candidate_response = np.where(response >= threshold, response, 0.0)
    maxima = non_maximum_suppression(candidate_response, min_distance=min_distance)
    coordinates = np.column_stack(np.nonzero(maxima))
    return coordinates, response


def _interpolated_non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    rows, cols = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    direction = (np.rad2deg(angle) + 180) % 180

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            value = magnitude[row, col]
            theta = direction[row, col]

            if (0 <= theta < 22.5) or (157.5 <= theta < 180):
                before, after = magnitude[row, col - 1], magnitude[row, col + 1]
            elif 22.5 <= theta < 67.5:
                before, after = magnitude[row - 1, col + 1], magnitude[row + 1, col - 1]
            elif 67.5 <= theta < 112.5:
                before, after = magnitude[row - 1, col], magnitude[row + 1, col]
            else:
                before, after = magnitude[row - 1, col - 1], magnitude[row + 1, col + 1]

            if value >= before and value >= after:
                suppressed[row, col] = value

    return suppressed


def _hysteresis(edges: np.ndarray, low: float, high: float) -> np.ndarray:
    strong = edges >= high
    weak = (edges >= low) & ~strong
    labels, count = ndimage.label(weak | strong)
    if count == 0:
        return strong

    strong_labels = np.unique(labels[strong])
    keep = np.isin(labels, strong_labels[strong_labels != 0])
    return keep


def canny_edges(
    image: np.ndarray,
    sigma: float = 1.2,
    low_threshold: float = 0.04,
    high_threshold: float = 0.10,
) -> np.ndarray:
    """Detect edges with a compact Canny-style pipeline."""
    if not 0 <= low_threshold <= high_threshold:
        raise ValueError("thresholds must satisfy 0 <= low_threshold <= high_threshold")

    gray = to_grayscale(image)
    smoothed = ndimage.gaussian_filter(gray, sigma=sigma)

    gx = ndimage.sobel(smoothed, axis=1, mode="reflect")
    gy = ndimage.sobel(smoothed, axis=0, mode="reflect")
    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    angle = np.arctan2(gy, gx)
    suppressed = _interpolated_non_maximum_suppression(magnitude, angle)
    return _hysteresis(suppressed, low_threshold, high_threshold)
