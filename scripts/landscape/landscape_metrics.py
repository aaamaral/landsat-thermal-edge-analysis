from __future__ import annotations

import numpy as np
from scipy import ndimage

# pylandstats uses 8-connectivity (queen) for patches; match it.
_EIGHT_CONN = ndimage.generate_binary_structure(2, 2)
_FOUR_CONN = ndimage.generate_binary_structure(2, 1)


def _patch_area_and_shared(labels):
    """Per-label pixel area and internal (rook) shared-edge count, vectorized.

    Returns (area, shared) arrays indexed by label value (index 0 = background).
    """
    size = int(labels.max()) + 1
    area = np.bincount(labels.ravel(), minlength=size)
    shared = np.zeros(size, dtype=np.int64)
    left, right = labels[:, :-1], labels[:, 1:]
    same_h = (left == right) & (left > 0)
    shared += np.bincount(left[same_h], minlength=size)
    top, bottom = labels[:-1, :], labels[1:, :]
    same_v = (top == bottom) & (top > 0)
    shared += np.bincount(top[same_v], minlength=size)
    return area, shared


def aggregation_index(mask) -> float:
    """He et al. (2000) aggregation index (%), 4-neighbour like adjacencies."""
    mask = np.asarray(mask, dtype=bool)
    like = int(np.sum(mask[:, :-1] & mask[:, 1:]) + np.sum(mask[:-1, :] & mask[1:, :]))
    n = int(mask.sum())
    if n == 0:
        return 0.0
    side = int(np.floor(np.sqrt(n)))
    if n == side * side:
        max_like = 2 * side * (side - 1)
    elif n <= side * (side + 1):
        max_like = 2 * side * (side - 1) + 2 * (n - side * side) - 1
    else:
        max_like = 2 * side * (side - 1) + 2 * (n - side * side) - 2
    return (like / max_like) * 100 if max_like > 0 else 0.0


def patch_cohesion(mask) -> float:
    """Gustafson & Parker (1994) patch cohesion index (%), 4-connected patches."""
    mask = np.asarray(mask, dtype=bool)
    total_cells = mask.size
    labels, count = ndimage.label(mask, structure=_FOUR_CONN)
    if count == 0:
        return 0.0
    area, shared = _patch_area_and_shared(labels)
    sizes = area[1:].astype(np.float64)
    perimeter = 4 * sizes - 2 * shared[1:]
    sum_p = float(perimeter.sum())
    sum_p_sqrt_a = float((perimeter * np.sqrt(sizes)).sum())
    if sum_p_sqrt_a == 0:
        return 0.0
    return (1 - sum_p / sum_p_sqrt_a) * (1 - 1 / np.sqrt(total_cells)) ** -1 * 100


def _shape_index_mean(labels, n_patches):
    """Mean FRAGSTATS shape index over patches, vectorized."""
    area, shared = _patch_area_and_shared(labels)
    sizes = area[1:].astype(np.int64)
    perimeter = 4 * sizes - 2 * shared[1:]
    side = np.floor(np.sqrt(sizes)).astype(np.int64)
    min_perimeter = np.where(
        sizes == side * side, 4 * side,
        np.where(sizes <= side * side + side, 4 * side + 2, 4 * side + 4),
    )
    return float((perimeter / min_perimeter).mean())


def _adjacency_matrix(array, classes, nodata=0):
    """Double-counted 4-neighbour adjacency matrix (incl. self-adjacencies),
    matching pylandstats' compute_total_adjacency_df. Vectorized via bincount."""
    k = len(classes)
    remap = np.full(int(array.max()) + 1, -1, dtype=np.int64)
    for index, class_value in enumerate(classes):
        remap[class_value] = index
    g = np.zeros((k, k), dtype=np.float64)
    for first, second in ((array[:, :-1], array[:, 1:]), (array[:-1, :], array[1:, :])):
        first = first.ravel()
        second = second.ravel()
        keep = (first != nodata) & (second != nodata)
        codes = remap[first[keep]] * k + remap[second[keep]]
        counts = np.bincount(codes, minlength=k * k).reshape(k, k).astype(np.float64)
        g += counts + counts.T
    return g


def window_metrics(window_array, focal_class, res=30.0, nodata=0) -> dict:
    """Focal class-level + landscape-level metrics over one window array.

    Pure numpy/scipy reimplementation that matches pylandstats 3.1.0 to machine
    precision (validated in tests), avoiding a pylandstats Landscape object per
    polygon (the dominant cost at scale). cohesion and ai are in-house.
    """
    array = np.asarray(window_array)
    valid = int((array != nodata).sum())
    area_ha = valid * res * res / 1e4
    classes = [int(c) for c in np.unique(array) if int(c) != nodata]
    focal_mask = array == focal_class

    result = {
        "pland_focal": int(focal_mask.sum()) / valid * 100 if valid else 0.0,
        "np": 0,
        "pd": 0.0,
        "lpi": 0.0,
        "ed": 0.0,
        "area_mn": 0.0,
        "shape_mn": 0.0,
    }

    labels, n_patches = ndimage.label(focal_mask, structure=_EIGHT_CONN)
    result["np"] = int(n_patches)
    if n_patches:
        sizes = np.bincount(labels.ravel())[1:]
        result["lpi"] = int(sizes.max()) / valid * 100 if valid else 0.0
        result["area_mn"] = float(sizes.mean()) * res * res / 1e4
        result["pd"] = n_patches / area_ha * 100 if area_ha else 0.0
        result["shape_mn"] = _shape_index_mean(labels, n_patches)

    edges = 0
    for first, second in ((array[:, :-1], array[:, 1:]), (array[:-1, :], array[1:, :])):
        edges += int(np.sum((first == focal_class) & (second != focal_class) & (second != nodata)))
        edges += int(np.sum((second == focal_class) & (first != focal_class) & (first != nodata)))
    result["ed"] = edges * res / area_ha if area_ha else 0.0

    if len(classes) < 2:
        result["shdi"] = float("nan")
        result["contag"] = float("nan")
    else:
        g = _adjacency_matrix(array, classes, nodata)
        # SHDI (pylandstats): entropy of the adjacency column sums, not cell areas.
        column = g.sum(axis=0)
        column_p = column[column > 0] / column.sum()
        result["shdi"] = float(-(column_p * np.log(column_p)).sum())
        # CONTAGION: 1 - joint entropy of the flattened adjacency matrix / (2 ln m).
        flat = g.flatten()
        flat_p = flat[flat > 0] / flat.sum()
        joint_entropy = -(flat_p * np.log(flat_p)).sum()
        result["contag"] = float((1 - joint_entropy / (2 * np.log(len(classes)))) * 100)

    result["cohesion"] = patch_cohesion(focal_mask)
    result["ai"] = aggregation_index(focal_mask)
    return result
