"""Edge geometry on the 30 m MapBiomas polygon grid (local, exact).

For every pixel inside a labelled polygon this module derives, from the label and class
rasters only (no temperature):

* ``ring``    distance ring to the polygon's own cartographic boundary. Ring 0 is the
              boundary pixel layer (pixels 8-adjacent to a different label); ring k holds
              pixels whose Euclidean distance (pixel units) to that layer is in [k, k+1).
              The ring centre lies ~30k + 15 m from the mapped edge line, which reproduces
              the 0-30/30-60/60-90/>90 m bands of the submitted version for k = 0, 1, 2, >=3.
              Rings are capped at ``ring_cap`` (the cap ring pools everything deeper).
* ``nb_class`` MapBiomas class of the neighbour across the nearest boundary pixel (mode of
              the different-label 8-neighbours of that boundary pixel). 0 means the nearest
              boundary is the study-area limit (label 0), i.e. an artificial clipping edge.
* ``sector``  outward orientation of the nearest boundary (1=N, 2=E, 3=S, 4=W; 0=undefined).

It also returns per-(polygon, neighbour class) contact lengths counted on 4-neighbour pixel
faces, and the outward-normal components needed for interface azimuths.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

PIXEL_M = 30.0
RING_CAP = 10  # rings 0..9 explicit (0-300 m), ring 10 pools >= 300 m

# (drow, dcol) offsets; 4-neighbours first so they win mode ties.
OFFSETS_8 = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
OFFSETS_4 = OFFSETS_8[:4]


def _shift(a, dr, dc, fill):
    """Value of the neighbour at (row+dr, col+dc); ``fill`` outside the array."""
    out = np.full_like(a, fill)
    h, w = a.shape
    rs_dst = slice(max(0, -dr), h - max(0, dr))
    cs_dst = slice(max(0, -dc), w - max(0, dc))
    rs_src = slice(max(0, dr), h - max(0, -dr))
    cs_src = slice(max(0, dc), w - max(0, -dc))
    out[rs_dst, cs_dst] = a[rs_src, cs_src]
    return out


def boundary_layer(labels):
    """True for labelled pixels with at least one 8-neighbour of a different label."""
    labels = np.asarray(labels)
    inside = labels > 0
    edge = np.zeros(labels.shape, dtype=bool)
    for dr, dc in OFFSETS_8:
        edge |= _shift(labels, dr, dc, 0) != labels
    return edge & inside


def boundary_neighbor_class(labels, classes):
    """Mode class of the different-label 8-neighbours of each boundary pixel (-1 elsewhere).

    Neighbours with label 0 (outside the study area) contribute class 0. Ties are broken by
    the first offset in ``OFFSETS_8`` (4-neighbours first).
    """
    labels = np.asarray(labels)
    classes = np.asarray(classes)
    n = len(OFFSETS_8)
    nb_cls = np.empty((n,) + labels.shape, dtype=np.int16)
    for i, (dr, dc) in enumerate(OFFSETS_8):
        nl = _shift(labels, dr, dc, 0)
        nc = _shift(classes.astype(np.int16), dr, dc, 0)
        nc = np.where(nl == 0, 0, nc)
        nb_cls[i] = np.where(nl != labels, nc, -1)
    # count, for each candidate offset i, how many offsets share its class
    counts = np.zeros((n,) + labels.shape, dtype=np.int8)
    for i in range(n):
        valid_i = nb_cls[i] >= 0
        c = np.zeros(labels.shape, dtype=np.int8)
        for j in range(n):
            c += (nb_cls[j] == nb_cls[i]) & valid_i
        counts[i] = c
    best = np.argmax(counts, axis=0)  # first max -> 4-neighbour preference
    out = np.take_along_axis(nb_cls, best[None], axis=0)[0]
    out[(labels <= 0) | (counts.max(axis=0) == 0)] = -1
    return out


def boundary_normal(labels):
    """Outward normal (east, north) of each pixel = sum of unit offsets to different labels."""
    labels = np.asarray(labels)
    east = np.zeros(labels.shape, dtype=np.float32)
    north = np.zeros(labels.shape, dtype=np.float32)
    for dr, dc in OFFSETS_8:
        diff = _shift(labels, dr, dc, 0) != labels
        norm = np.hypot(dr, dc)
        east += diff * (dc / norm)
        north += diff * (-dr / norm)
    return east, north


def azimuth_sector(east, north):
    """1=N, 2=E, 3=S, 4=W quadrant of the outward direction; 0 if the vector is null."""
    az = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    sector = (((az + 45.0) % 360.0) // 90.0).astype(np.uint8) + 1
    sector[(np.abs(east) < 1e-6) & (np.abs(north) < 1e-6)] = 0
    return sector


def edge_geometry(labels, classes, ring_cap=RING_CAP):
    """Per-pixel ring / nb_class / sector for a (padded) tile.

    Returns a dict of arrays with the tile shape: ``ring`` (uint8, 255 outside polygons),
    ``nb_class`` (int16, -1 outside), ``sector`` (uint8), ``dist_px`` (float32) and
    ``own_boundary`` (bool: the nearest boundary pixel belongs to the same polygon).
    """
    labels = np.asarray(labels)
    inside = labels > 0
    edge = boundary_layer(labels)
    if not edge.any():
        dist = np.full(labels.shape, np.inf, dtype=np.float32)
        idx = None
    else:
        dist, idx = ndimage.distance_transform_edt(~edge, return_indices=True)
    nbc = boundary_neighbor_class(labels, classes)
    be, bn = boundary_normal(labels)
    ring = np.full(labels.shape, 255, dtype=np.uint8)
    nb_class = np.full(labels.shape, -1, dtype=np.int16)
    sector = np.zeros(labels.shape, dtype=np.uint8)
    own = np.zeros(labels.shape, dtype=bool)
    if idx is not None:
        ri, ci = idx
        r = np.minimum(np.floor(dist + 1e-6), ring_cap).astype(np.uint8)
        ring[inside] = r[inside]
        nb_class[inside] = nbc[ri, ci][inside]
        own[inside] = (labels[ri, ci] == labels)[inside]
        # direction: towards the nearest boundary pixel, then that pixel's outward normal
        rows, cols = np.indices(labels.shape)
        ve = (ci - cols).astype(np.float32)
        vn = -(ri - rows).astype(np.float32)
        on_edge = dist < 0.5
        ve = np.where(on_edge, be, ve)
        vn = np.where(on_edge, bn, vn)
        sec = azimuth_sector(ve, vn)
        sector[inside] = sec[inside]
    return {"ring": ring, "nb_class": nb_class, "sector": sector,
            "dist_px": dist.astype(np.float32), "own_boundary": own}


def contact_faces(labels, classes):
    """Per-(label, neighbour class) count of shared 4-neighbour pixel faces.

    Returns a DataFrame-ready dict of arrays: label, nb_class, faces, normal_e, normal_n
    (normal components summed over faces, pointing out of the focal polygon). Faces against
    the study-area limit are reported with nb_class 0.
    """
    labels = np.asarray(labels)
    classes = np.asarray(classes).astype(np.int16)
    out_l, out_c, out_e, out_n = [], [], [], []
    for dr, dc in OFFSETS_4:
        nl = _shift(labels, dr, dc, 0)
        nc = np.where(nl == 0, 0, _shift(classes, dr, dc, 0))
        m = (labels > 0) & (nl != labels)
        out_l.append(labels[m])
        out_c.append(nc[m])
        out_e.append(np.full(m.sum(), dc, dtype=np.int8))
        out_n.append(np.full(m.sum(), -dr, dtype=np.int8))
    lab = np.concatenate(out_l).astype(np.int64)
    cls = np.concatenate(out_c).astype(np.int64)
    e = np.concatenate(out_e).astype(np.float64)
    n = np.concatenate(out_n).astype(np.float64)
    key = lab * 64 + cls
    uk, inv = np.unique(key, return_inverse=True)
    return {
        "label": uk // 64,
        "nb_class": uk % 64,
        "faces": np.bincount(inv),
        "normal_e": np.bincount(inv, weights=e),
        "normal_n": np.bincount(inv, weights=n),
    }
