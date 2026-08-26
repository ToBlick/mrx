"""Smoke tests of ``mrx.plotting`` on the session torus: every figure builds
with the Agg backend and carries the artists it promises."""
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

import mrx
from mrx.differential_forms import DiscreteFunction
from mrx.plotting import (
    get_2d_grids,
    plot_crossections_separate,
    plot_torus,
    plot_twin_axis,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(scope="module")
def cuts(tiny_seq):
    zetas = np.array([0.0, 0.25, 0.5])
    grids_pol = [get_2d_grids(tiny_seq.map, cut_axis=2, cut_value=float(z),
                              nx=12, ny=12, nz=1) for z in zetas]
    surface = get_2d_grids(tiny_seq.map, cut_axis=0, cut_value=1.0 - 1e-6,
                           ny=24, nz=24, invert_z=True)
    return zetas, grids_pol, surface


def test_get_2d_grids_shapes_and_map(tiny_seq, cuts):
    zetas, grids_pol, surface = cuts
    x, y, (Y1, Y2, Y3), (x1, x2, x3) = grids_pol[1]
    assert x.shape == (144, 3) and y.shape == (144, 3)
    assert Y1.shape == Y2.shape == Y3.shape == (12, 12)
    assert x3.shape == (1,) and float(x3[0]) == 0.25
    # lax.map batches the map; per-point evaluation differs by round-off.
    assert float(jnp.max(jnp.abs(y - jnp.stack([tiny_seq.map(p) for p in x])))) <= mrx.eps(10)
    # The donut torus of the fixture, R in [R0 - eps, R0 + eps] = [2/3, 4/3],
    # as its (4, 6, 4) p=2 spline interpolant renders it (inner edge 0.643).
    R = np.hypot(np.asarray(Y1), np.asarray(Y2))
    assert 0.6 <= R.min() and R.max() <= 1.4, (R.min(), R.max())
    assert surface[2][0].shape == (24, 24)


def test_plot_torus_and_crossections(tiny_seq, cuts):
    zetas, grids_pol, surface = cuts
    dofs = jnp.arange(tiny_seq.n0, dtype=float) / tiny_seq.n0
    f = DiscreteFunction(dofs, tiny_seq.basis_0, tiny_seq.e0)

    def p_h(x):
        return f(x)[0]

    fig, ax = plot_torus(p_h, grids_pol, surface, cbar_label=r"$f$")
    assert ax.name == "3d"
    assert len(ax.collections) == 1 + len(grids_pol)      # wireframe + cuts
    assert len(fig.axes) == 2                              # the 3-D axes + colour bar
    lo, hi = ax.get_xlim3d(), ax.get_zlim3d()
    assert abs((lo[1] - lo[0]) - (hi[1] - hi[0])) <= 1e-9   # set_axes_equal
    plt.close(fig)

    fig, axes = plot_crossections_separate(p_h, grids_pol, zetas, plot_centerline=True)
    assert len(axes) == len(grids_pol) and len(fig.axes) == len(grids_pol) + 1
    assert all(ax.get_xlim() == axes[0].get_xlim() for ax in axes)
    plt.close(fig)


def test_plot_twin_axis():
    F = np.geomspace(1.0, 1e-4, 30)
    E = np.linspace(1.0, 0.5, 30)
    fig, (ax1, ax2) = plot_twin_axis(F, E, left_label="F", right_label="E",
                                     num_iters_inner=5, right_plot_kwargs={"color": "red"})
    assert ax1.get_yscale() == "log" and ax2.get_yscale() == "linear"
    assert ax2.get_shared_x_axes().joined(ax1, ax2)
    line = ax1.get_lines()[0]
    assert float(line.get_xdata()[-1]) == 5 * 29
    assert ax2.get_lines()[0].get_color() == "red"
    assert ax2.yaxis.label.get_color() == "red"
    plt.close(fig)
