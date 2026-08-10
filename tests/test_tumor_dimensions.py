import math
import numpy as np
import pytest

from api.utils import (
    compute_tumor_dimensions_from_data,
    compute_tumor_dimensions_and_geometry_from_data,
)


def test_cuboid_isotropic():
    """TEST 1: Axis-aligned synthetic cuboid with 1.0mm isotropic spacing.
    Voxel dimensions: 50 x 30 x 20 voxels.
    Expected physical dimensions: 50.0 mm x 30.0 mm x 20.0 mm.
    """
    mask = np.zeros((100, 100, 100), dtype=np.uint8)
    mask[20:70, 30:60, 40:60] = 1  # 50 x 30 x 20 voxels

    affine = np.eye(4)
    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is not None
    assert geom is not None
    assert dims["length"] == 50.0
    assert dims["width"] == 30.0
    assert dims["height"] == 20.0

    # Geometry verification
    len_geom = geom["length"]
    start = np.array(len_geom["start_mm"])
    end = np.array(len_geom["end_mm"])
    mid = np.array(len_geom["midpoint_mm"])

    calc_len = np.linalg.norm(end - start)
    assert abs(calc_len - dims["length"]) < 0.05
    assert np.allclose(mid, (start + end) / 2.0, atol=0.05)


def test_anisotropic_spacing():
    """TEST 2: Anisotropic voxel spacing (sx=0.8mm, sy=0.6mm, sz=1.0mm).
    Voxels: 20 x 30 x 10.
    Physical extent along X: 20 * 0.8 = 16.0 mm.
    Physical extent along Y: 30 * 0.6 = 18.0 mm.
    Physical extent along Z: 10 * 1.0 = 10.0 mm.
    Expected ordered dimensions: Length = 18.0, Width = 16.0, Height = 10.0.
    """
    mask = np.zeros((50, 50, 50), dtype=np.uint8)
    mask[10:30, 10:40, 10:20] = 1

    affine = np.array([
        [0.8, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is not None
    assert geom is not None
    assert dims["length"] == 18.0
    assert dims["width"] == 16.0
    assert dims["height"] == 10.0

    len_geom = geom["length"]
    start = np.array(len_geom["start_mm"])
    end = np.array(len_geom["end_mm"])
    mid = np.array(len_geom["midpoint_mm"])

    calc_len = np.linalg.norm(end - start)
    assert abs(calc_len - dims["length"]) < 0.05
    assert np.allclose(mid, (start + end) / 2.0, atol=0.05)


def test_oblique_affine_rotation_invariance():
    """TEST 3 & TEST 5: Oblique/rotated affine matrix and rotation invariance.
    Create a 40 x 20 x 10 voxel cuboid with 1.0mm isotropic spacing (physical size: 40 x 20 x 10 mm).
    Apply a 30-degree rotation around Z axis to the affine matrix.
    Verify physical dimensions remain 40.0 mm x 20.0 mm x 10.0 mm.
    """
    mask = np.zeros((80, 80, 80), dtype=np.uint8)
    mask[20:60, 30:50, 30:40] = 1  # 40 x 20 x 10 voxels

    angle = math.radians(30.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    # Rotated affine around Z
    affine_rot = np.array([
        [cos_a, -sin_a, 0.0, 10.0],
        [sin_a,  cos_a, 0.0, 20.0],
        [0.0,    0.0,   1.0, 30.0],
        [0.0,    0.0,   0.0,  1.0]
    ])

    dims_identity, geom_identity = compute_tumor_dimensions_and_geometry_from_data(mask, np.eye(4))
    dims_rotated, geom_rotated = compute_tumor_dimensions_and_geometry_from_data(mask, affine_rot)

    assert dims_identity is not None and geom_identity is not None
    assert dims_rotated is not None and geom_rotated is not None

    assert dims_identity["length"] == 40.0
    assert dims_identity["width"] == 20.0
    assert dims_identity["height"] == 10.0

    # Rotated affine should produce identical physical dimensions
    assert abs(dims_rotated["length"] - 40.0) < 0.1
    assert abs(dims_rotated["width"] - 20.0) < 0.1
    assert abs(dims_rotated["height"] - 10.0) < 0.1

    # Verify endpoints norm match length
    start_rot = np.array(geom_rotated["length"]["start_mm"])
    end_rot = np.array(geom_rotated["length"]["end_mm"])
    mid_rot = np.array(geom_rotated["length"]["midpoint_mm"])

    calc_len = np.linalg.norm(end_rot - start_rot)
    assert abs(calc_len - dims_rotated["length"]) < 0.1
    assert np.allclose(mid_rot, (start_rot + end_rot) / 2.0, atol=0.05)


def test_empty_mask():
    """TEST 4: Empty mask with no tumor voxels.
    Verify compute_tumor_dimensions_and_geometry_from_data returns (None, None).
    """
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    affine = np.eye(4)

    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)
    assert dims is None
    assert geom is None


def test_single_voxel_degenerate():
    """TEST 6a: Single voxel segmentation (degenerate case).
    With 1.0mm isotropic spacing, 1 voxel has physical dimensions 1.0 x 1.0 x 1.0 mm.
    """
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[10, 10, 10] = 1

    affine = np.eye(4)
    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is not None and geom is not None
    assert dims["length"] == 1.0
    assert dims["width"] == 1.0
    assert dims["height"] == 1.0

    start = np.array(geom["length"]["start_mm"])
    end = np.array(geom["length"]["end_mm"])
    assert abs(np.linalg.norm(end - start) - 1.0) < 0.05


def test_flat_planar_segmentation():
    """TEST 6b: Flat 2D planar segmentation (20 x 30 x 1 voxels).
    Verify no NaNs or Inf, returns finite ordered values length >= width >= height >= 0.
    """
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[10:30, 5:35, 20:21] = 1  # 20 x 30 x 1 voxels

    affine = np.eye(4)
    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is not None and geom is not None
    assert not math.isnan(dims["length"])
    assert not math.isnan(dims["width"])
    assert not math.isnan(dims["height"])

    assert dims["length"] == 30.0
    assert dims["width"] == 20.0
    assert dims["height"] == 1.0
    assert dims["length"] >= dims["width"] >= dims["height"] >= 0

    start = np.array(geom["length"]["start_mm"])
    end = np.array(geom["length"]["end_mm"])
    assert abs(np.linalg.norm(end - start) - dims["length"]) < 0.05


def test_irregular_shape():
    """TEST 6c: Irregular L-shaped tumor segmentation.
    Verify ordered dimensions length >= width >= height > 0 and endpoints norm matches length.
    """
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    # Form an L shape
    mask[10:30, 10:15, 10:20] = 1
    mask[10:15, 10:30, 10:20] = 1

    affine = np.eye(4)
    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is not None and geom is not None
    assert dims["length"] >= dims["width"] >= dims["height"] > 0

    start = np.array(geom["length"]["start_mm"])
    end = np.array(geom["length"]["end_mm"])
    assert abs(np.linalg.norm(end - start) - dims["length"]) < 0.05


def test_geometry_dimension_correspondence():
    """TEST G: Verify returned Length geometry corresponds to largest PHYSICAL extent record."""
    mask = np.zeros((80, 80, 80), dtype=np.uint8)
    # Create a shape where physical extents differ from voxel extents
    mask[20:60, 25:55, 30:50] = 1  # 40 x 30 x 20 voxels

    # Anisotropic spacing to create different physical extents
    affine = np.array([
        [0.5, 0.0, 0.0, 0.0],  # X: 40 * 0.5 = 20.0 mm
        [0.0, 0.8, 0.0, 0.0],  # Y: 30 * 0.8 = 24.0 mm (largest)
        [0.0, 0.0, 1.0, 0.0],  # Z: 20 * 1.0 = 20.0 mm
        [0.0, 0.0, 0.0, 1.0]
    ])

    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is not None and geom is not None
    # The largest physical extent should be 24.0 mm (Y axis)
    assert dims["length"] == 24.0

    # Verify geometry corresponds to this largest extent
    len_geom = geom["length"]
    start = np.array(len_geom["start_mm"])
    end = np.array(len_geom["end_mm"])
    calc_len = np.linalg.norm(end - start)
    assert abs(calc_len - dims["length"]) < 0.05


def test_midpoint_calculation():
    """TEST H: Verify midpoint is geometric center of endpoints."""
    mask = np.zeros((60, 60, 60), dtype=np.uint8)
    mask[20:40, 20:50, 25:35] = 1  # 20 x 30 x 10 voxels

    affine = np.eye(4)
    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is not None and geom is not None

    len_geom = geom["length"]
    start = np.array(len_geom["start_mm"])
    end = np.array(len_geom["end_mm"])
    mid = np.array(len_geom["midpoint_mm"])

    expected_mid = (start + end) / 2.0
    assert np.allclose(mid, expected_mid, atol=0.05)


def test_empty_segmentation_geometry():
    """TEST I: Empty segmentation should return None for geometry."""
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    affine = np.eye(4)

    dims, geom = compute_tumor_dimensions_and_geometry_from_data(mask, affine)

    assert dims is None
    assert geom is None
