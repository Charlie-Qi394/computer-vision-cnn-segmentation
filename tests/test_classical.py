import numpy as np

from cv_portfolio.classical import canny_edges, harris_corners


def test_harris_corners_detects_square_corners():
    image = np.zeros((40, 40), dtype=float)
    image[10:30, 10:30] = 1.0

    corners, response = harris_corners(image, sigma=1.0, threshold_rel=0.05, min_distance=3)

    assert response.shape == image.shape
    assert len(corners) >= 4


def test_canny_edges_returns_boolean_edge_map():
    image = np.zeros((40, 40), dtype=float)
    image[:, 20:] = 1.0

    edges = canny_edges(image, sigma=1.0, low_threshold=0.05, high_threshold=0.15)

    assert edges.dtype == bool
    assert edges.shape == image.shape
    assert edges.sum() > 0
