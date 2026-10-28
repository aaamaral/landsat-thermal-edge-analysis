from __future__ import annotations


def polygon_id(focal_class: int, component_label: int) -> str:
    # labels can span the full int range; pad to 19 digits for fixed-width,
    # lexically sortable ids.
    return f"c{focal_class:03d}_p{component_label:019d}"
