"""Backward-compatibility re-exports from :mod:`mrx.geometry`.

.. deprecated::
    Import directly from :mod:`mrx.geometry`.
"""

from mrx.geometry import (  # noqa: F401
    _coeffs_to_raw_grid,
    _tp_evaluate,
    compute_geometry_terms_from_spline,
    min_jacobian_from_coeffs,
    spline_map_F_DF_at_quad,
    spline_map_jacobian_j_at_quad,
)
