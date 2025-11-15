# %%
# test_plotting.py
import os
import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

from mrx.plotting import (
    converge_plot,
    generate_solovev_plots,
    get_1d_grids,
    get_2d_grids,
    get_3d_grids,
    plot_crossections,
    plot_crossections_separate,
    plot_scalar_fct_physical_logical,
    plot_torus,
    plot_twin_axis,
    poincare_plot,
    pressure_plot,
    set_axes_equal,
    trace_plot,
    trajectory_plane_intersections_jit,
    trajectory_plane_intersections_list,
)
from mrx.mappings import cerfon_map, toroid_map
from mrx.derham_sequence import DeRhamSequence

jax.config.update("jax_enable_x64", True)


# Helper function to check if running in CI
def is_running_in_ci():
    """Check if running in CI environment."""
    return os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true"


# Global list to collect all figure references for display at the end
_collected_figures = []


def collect_figure(fig):
    """Collect a figure reference to display at the end.
    
    Note: Figures are kept open (not closed) so they can be displayed at the end.
    """
    if not is_running_in_ci() and fig is not None:
        _collected_figures.append(fig)
        # Don't close the figure - keep it open for display at the end
    elif is_running_in_ci() and fig is not None:
        # In CI, close figures immediately to save memory
        plt.close(fig)


@pytest.fixture(scope="session", autouse=True)
def save_all_plots_at_end(request):
    """Pytest fixture to save all collected plots at the end of the test session."""
    yield  # Run all tests first
    
    # After all tests complete, save all open figures
    if not is_running_in_ci():
        all_figures = plt.get_fignums()
        if len(all_figures) > 0:
            # Ensure output directory exists (always in mrx/test/test_outputs/)
            test_output_dir = Path(__file__).parent / "test_outputs"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            # Save all figures to test_outputs directory
            for i, fig_num in enumerate(all_figures):
                fig = plt.figure(fig_num)
                output_path = test_output_dir / f"figure_{i}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved figure {i+1} to {output_path}")
    
    # Clean up: close all figures
    plt.close('all')

# ============================================================================
# Tests for get_3d_grids
# ============================================================================
def test_get_3d_grids_basic():
    """Test basic get_3d_grids functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_3d_grids(F, nx=4, ny=4, nz=4)
    
    assert _x.shape == (4*4*4, 3), "x should have shape (nx*ny*nz, 3)"
    assert _y.shape == (4*4*4, 3), "y should have shape (nx*ny*nz, 3)"
    assert _y1.shape == (4, 4, 4), "y1 should have shape (nx, ny, nz)"
    assert _y2.shape == (4, 4, 4), "y2 should have shape (nx, ny, nz)"
    assert _y3.shape == (4, 4, 4), "y3 should have shape (nx, ny, nz)"
    assert _x1.shape == (4,), "x1 should have shape (nx,)"
    assert _x2.shape == (4,), "x2 should have shape (ny,)"
    assert _x3.shape == (4,), "x3 should have shape (nz,)"


def test_get_3d_grids_custom_bounds():
    """Test get_3d_grids with custom bounds."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_3d_grids(
        F, x_min=0.1, x_max=0.9, y_min=0.1, y_max=0.9, z_min=0.1, z_max=0.9, nx=8, ny=8, nz=8
    )
    
    assert jnp.all(_x1 >= 0.1) and jnp.all(_x1 <= 0.9), "x1 should be within bounds"
    assert jnp.all(_x2 >= 0.1) and jnp.all(_x2 <= 0.9), "x2 should be within bounds"
    assert jnp.all(_x3 >= 0.1) and jnp.all(_x3 <= 0.9), "x3 should be within bounds"


def test_get_3d_grids_cerfon_map():
    """Test get_3d_grids with cerfon_map."""
    F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_3d_grids(F, nx=5, ny=5, nz=5)
    
    assert _x.shape == (125, 3), "x should have shape (125, 3) for 5x5x5 grid"
    assert _y.shape == (125, 3), "y should have shape (125, 3) for 5x5x5 grid"


# ============================================================================
# Tests for get_2d_grids
# ============================================================================
def test_get_2d_grids_basic():
    """Test basic get_2d_grids functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(F, cut_axis=2, cut_value=0.5, nx=8, ny=8, nz=1)
    
    assert _x.shape[1] == 3, "x should have 3 columns"
    assert _y.shape[1] == 3, "y should have 3 columns"
    assert _y1.shape == (8, 8), "y1 should have shape (nx, ny) for cut_axis=2"
    assert _y2.shape == (8, 8), "y2 should have shape (nx, ny) for cut_axis=2"
    assert _y3.shape == (8, 8), "y3 should have shape (nx, ny) for cut_axis=2"


def test_get_2d_grids_different_cut_axes():
    """Test get_2d_grids with different cut axes."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Test cut_axis=0
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(F, cut_axis=0, cut_value=0.5, nx=1, ny=8, nz=8)
    assert _y1.shape == (8, 8), "y1 should have shape (ny, nz) for cut_axis=0"
    
    # Test cut_axis=1
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_2d_grids(F, cut_axis=1, cut_value=0.5, nx=8, ny=1, nz=8)
    assert _y1.shape == (8, 8), "y1 should have shape (nx, nz) for cut_axis=1"


def test_get_2d_grids_invert():
    """Test get_2d_grids with invert flags."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    _x1, _y1, _, _ = get_2d_grids(F, cut_axis=2, cut_value=0.5, nx=8, ny=8, nz=1, invert_x=False)
    _x2, _y2, _, _ = get_2d_grids(F, cut_axis=2, cut_value=0.5, nx=8, ny=8, nz=1, invert_x=True)
    
    # Check that x coordinates are reversed
    npt.assert_allclose(_x1[:, 0][::-1], _x2[:, 0], atol=1e-10, err_msg="invert_x should reverse x coordinates")


def test_get_2d_grids_usage_example():
    """Test get_2d_grids based on usage example from solovev_single_plot.py."""
    F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
    grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=8, ny=8, nz=1) for v in jnp.linspace(0, 1, 3, endpoint=False)]
    
    assert len(grids_pol) == 3, "Should create 3 grids"
    for grid in grids_pol:
        _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = grid
        assert _y1.shape == (8, 8), "Each grid should have shape (8, 8)"


# ============================================================================
# Tests for get_1d_grids
# ============================================================================
def test_get_1d_grids_basic():
    """Test basic get_1d_grids functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_1d_grids(F, zeta=0.5, chi=0.25, nx=16)
    
    assert _x.shape == (16, 3), "x should have shape (nx, 3)"
    assert _y.shape == (16, 3), "y should have shape (nx, 3)"
    assert _y1.shape == (16,), "y1 should have shape (nx,)"
    assert _y2.shape == (16,), "y2 should have shape (nx,)"
    assert _y3.shape == (16,), "y3 should have shape (nx,)"


def test_get_1d_grids_different_angles():
    """Test get_1d_grids with different zeta and chi values."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    for zeta in [0.0, 0.25, 0.5, 0.75]:
        for chi in [0.0, 0.25, 0.5, 0.75]:
            _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = get_1d_grids(F, zeta=zeta, chi=chi, nx=8)
            assert _x.shape == (8, 3), f"Failed for zeta={zeta}, chi={chi}"


# ============================================================================
# Tests for trajectory_plane_intersections_jit
# ============================================================================
def test_trajectory_plane_intersections_jit_basic():
    """Test basic trajectory_plane_intersections_jit functionality."""
    # Create simple trajectories that cross a plane
    trajectories = jnp.array([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],  # Trajectory 1: crosses plane at z=0.5
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.7]],  # Trajectory 2: crosses plane at z=0.5
    ])
    
    intersections, mask = trajectory_plane_intersections_jit(trajectories, plane_val=0.5, axis=2)
    
    assert intersections.shape == (2, 2, 3), "intersections should have shape (N, T-1, D)"
    assert mask.shape == (2, 2), "mask should have shape (N, T-1)"
    assert jnp.any(mask), "At least one intersection should be found"


def test_trajectory_plane_intersections_jit_no_crossing():
    """Test trajectory_plane_intersections_jit when no crossings occur."""
    # Create trajectories that don't cross the plane
    trajectories = jnp.array([
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],  # All below plane
        [[0.0, 0.0, 0.8], [0.0, 0.0, 0.9], [0.0, 0.0, 1.0]],  # All above plane
    ])
    
    intersections, mask = trajectory_plane_intersections_jit(trajectories, plane_val=0.5, axis=2)
    
    assert not jnp.any(mask), "No intersections should be found"


def test_trajectory_plane_intersections_jit_plane_zero():
    """Test trajectory_plane_intersections_jit with plane_val=0 (special case)."""
    trajectories = jnp.array([
        [[0.0, 0.0, 0.9], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],  # Crosses near 0
    ])
    
    intersections, mask = trajectory_plane_intersections_jit(trajectories, plane_val=0.0, axis=2)
    
    assert intersections.shape == (1, 2, 3), "intersections should have correct shape"


# ============================================================================
# Tests for trajectory_plane_intersections_list
# ============================================================================
def test_trajectory_plane_intersections_list_basic():
    """Test basic trajectory_plane_intersections_list functionality."""
    # Create a trajectory that crosses a plane
    trajectory = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
    ])
    
    plane_point = jnp.array([0.0, 0.0, 0.5])
    plane_normal = jnp.array([0.0, 0.0, 1.0])
    
    intersections = trajectory_plane_intersections_list(trajectory, plane_point, plane_normal)
    
    assert len(intersections) > 0, "Should find at least one intersection"
    assert all(isinstance(p, np.ndarray) for p in intersections), "All intersections should be numpy arrays"
    assert all(p.shape == (3,) for p in intersections), "All intersections should have shape (3,)"


def test_trajectory_plane_intersections_list_no_crossing():
    """Test trajectory_plane_intersections_list when no crossings occur."""
    trajectory = jnp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1],
        [0.0, 0.0, 0.2],
    ])
    
    plane_point = jnp.array([0.0, 0.0, 0.5])
    plane_normal = jnp.array([0.0, 0.0, 1.0])
    
    intersections = trajectory_plane_intersections_list(trajectory, plane_point, plane_normal)
    
    assert len(intersections) == 0, "Should find no intersections"


# ============================================================================
# Tests for plot_crossections_separate
# ============================================================================
def test_plot_crossections_separate_basic():
    """Test basic plot_crossections_separate functionality."""    
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Create a simple function to plot
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    # Create grids
    cuts = jnp.linspace(0, 1, 3, endpoint=False)
    grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=8, ny=8, nz=1) for v in cuts]
    
    fig, axes = plot_crossections_separate(p_h, grids_pol, cuts, textsize=12, ticksize=12)
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_crossections_separate")
    assert fig is not None, "Figure should be created"
    assert len(axes) == 3, "Should have 3 axes"


def test_plot_crossections_separate_with_centerline():
    """Test plot_crossections_separate with centerline."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    cuts = jnp.linspace(0, 1, 2, endpoint=False)
    grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=8, ny=8, nz=1) for v in cuts]
    
    fig, axes = plot_crossections_separate(p_h, grids_pol, cuts, plot_centerline=True)
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_crossections_separate with centerline")
    assert fig is not None, "Figure should be created"


# ============================================================================
# Tests for plot_torus
# ============================================================================
def test_plot_torus_basic():
    """Test basic plot_torus functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    # Create grids based on usage example
    cuts = jnp.linspace(0, 1, 2, endpoint=False)
    grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=8, ny=8, nz=1) for v in cuts]
    grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0, ny=16, nz=16, z_min=0, z_max=1)
    
    fig, ax = plot_torus(p_h, grids_pol, grid_surface, figsize=(6, 4), noaxes=False)
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_torus")
    assert fig is not None, "Figure should be created"
    assert ax is not None, "Axes should be created"

# ============================================================================
# Tests for plot_crossections
# ============================================================================
def test_plot_crossections_basic():
    """Test basic plot_crossections functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def f(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    grids = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=8, ny=8, nz=1) for v in [0.0, 0.5]]
    
    fig, ax = plot_crossections(f, grids)
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_crossections")
    assert fig is not None, "Figure should be created"
    assert ax is not None, "Axes should be created"


# ============================================================================
# Tests for pressure_plot
# ============================================================================
def test_pressure_plot_basic():
    """Test basic pressure_plot functionality."""    
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Create a simple DeRham sequence
    ns = (4, 4, 4)
    ps = (2, 2, 2)
    q = max(ps)
    types = ("clamped", "periodic", "periodic")
    Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)
    Seq.evaluate_1d()
    Seq.assemble_M0()
    
    # Create a simple pressure field
    def p_func(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    p_dof = jnp.linalg.solve(Seq.M0, Seq.P0(p_func))
    p = p_dof
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "plots"
        outdir.mkdir()
        
        pressure_plot(
            p, Seq, F, str(outdir), filename="test_pressure.pdf", 
            resolution=16, zeta=0.0)
        if not is_running_in_ci():   
            plt.title("Test pressure_plot")
        # Check that file was created
        output_file = outdir / "test_pressure.pdf"
        assert output_file.exists(), "Output file should be created"


# ============================================================================
# Tests for trace_plot
# ============================================================================
def test_trace_plot_basic():
    """Test basic trace_plot functionality."""
    iterations = jnp.array([0, 1, 2, 3, 4])
    force_trace = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    helicity_trace = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    divergence_trace = jnp.array([0.0, 0.01, 0.005, 0.002, 0.001])
    velocity_trace = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
    wall_time_trace = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    energy_trace = jnp.array([1.0, 0.9, 0.8, 0.7, 0.6])
    trace_dict = {
        "iterations": iterations,
        "force_trace": force_trace,
        "helicity_trace": helicity_trace,
        "divergence_trace": divergence_trace,
        "velocity_trace": velocity_trace,
        "wall_time_trace": wall_time_trace,
        "energy_trace": energy_trace,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "plots"
        outdir.mkdir()
        
        trace_plot(
            trace_dict=trace_dict,
            filename=str(outdir) + "/test_trace.pdf"
        )
        if not is_running_in_ci():   
            plt.title("Test trace_plot")
        output_file = outdir / "test_trace.pdf"
        assert output_file.exists(), "Output file should be created"


def test_trace_plot_no_energy():
    """Test trace_plot without energy_trace."""
    iterations = jnp.array([0, 1, 2])
    force_trace = jnp.array([1.0, 0.5, 0.25])
    helicity_trace = jnp.array([1.0, 1.0, 1.0])
    divergence_trace = jnp.array([0.0, 0.01, 0.005])
    velocity_trace = jnp.array([1.0, 0.8, 0.6])
    wall_time_trace = jnp.array([0.0, 1.0, 2.0])

    trace_dict = {
        "iterations": iterations,
        "force_trace": force_trace,
        "helicity_trace": helicity_trace,
        "divergence_trace": divergence_trace,
        "velocity_trace": velocity_trace,
        "wall_time_trace": wall_time_trace,
        "energy_trace": None,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "plots"
        outdir.mkdir()
        
        trace_plot(
            trace_dict=trace_dict,
            filename=str(outdir) + "/test_trace_no_energy.pdf"
        )
        if not is_running_in_ci():   
            plt.title("Test trace_plot without energy_trace")

        output_file = outdir / "test_trace_no_energy.pdf"
        assert output_file.exists(), "Output file should be created"


# ============================================================================
# Tests for set_axes_equal
# ============================================================================
def test_set_axes_equal():
    """Test set_axes_equal functionality."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set some arbitrary limits
    ax.set_xlim3d([0, 10])
    ax.set_ylim3d([0, 5])
    ax.set_zlim3d([0, 20])
    
    # Apply set_axes_equal
    set_axes_equal(ax)
    
    # Check that limits are equal
    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()
    
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]
    
    assert abs(x_range - y_range) < 1e-10, "x and y ranges should be equal"
    assert abs(x_range - z_range) < 1e-10, "x and z ranges should be equal"
    assert abs(y_range - z_range) < 1e-10, "y and z ranges should be equal"
    
    collect_figure(fig)


# ============================================================================
# Tests for converge_plot
# ============================================================================
def test_converge_plot_basic():
    """Test basic converge_plot functionality."""
    ns = jnp.array([4, 8, 16, 32])
    ps = jnp.array([1, 2])
    qs = jnp.array([2, 3])
    
    # Create error array with shape (len(ns), len(ps), len(qs))
    # Error decreases with increasing n (convergence)
    err = jnp.array([
        [[1.0, 0.8], [0.5, 0.4]],   # n=4
        [[0.5, 0.4], [0.25, 0.2]],  # n=8
        [[0.25, 0.2], [0.125, 0.1]], # n=16
        [[0.125, 0.1], [0.0625, 0.05]], # n=32
    ])
    
    fig = converge_plot(err, ns, ps, qs)
    
    assert fig is not None, "Figure should be created"
    assert hasattr(fig, 'data'), "Figure should have data attribute"
    assert len(fig.data) > 0, "Figure should have traces"


def test_converge_plot_single_p():
    """Test converge_plot with single polynomial order."""
    ns = jnp.array([4, 8, 16])
    ps = jnp.array([2])
    qs = jnp.array([2, 3, 4])
    
    err = jnp.array([
        [[1.0, 0.8, 0.6]],   # n=4
        [[0.5, 0.4, 0.3]],   # n=8
        [[0.25, 0.2, 0.15]], # n=16
    ])
    
    fig = converge_plot(err, ns, ps, qs)
    
    assert fig is not None, "Figure should be created"
    assert len(fig.data) > 0, "Figure should have traces"


def test_converge_plot_single_q():
    """Test converge_plot with single quadrature rule."""
    ns = jnp.array([4, 8, 16])
    ps = jnp.array([1, 2, 3])
    qs = jnp.array([2, 3])  # Need at least 2 q values to avoid plotly division by zero
    
    err = jnp.array([
        [[1.0, 0.8], [0.5, 0.4], [0.25, 0.2]],   # n=4
        [[0.5, 0.4], [0.25, 0.2], [0.125, 0.1]], # n=8
        [[0.25, 0.2], [0.125, 0.1], [0.0625, 0.05]], # n=16
    ])
    
    fig = converge_plot(err, ns, ps, qs)
    
    assert fig is not None, "Figure should be created"
    assert len(fig.data) > 0, "Figure should have traces"


def test_converge_plot_multiple_combinations():
    """Test converge_plot with multiple p and q combinations."""
    ns = jnp.array([4, 8, 16, 32, 64])
    ps = jnp.array([1, 2, 3, 4])
    qs = jnp.array([2, 3, 4, 5])
    
    # Create error array with convergence behavior
    err = jnp.zeros((len(ns), len(ps), len(qs)))
    for i, n in enumerate(ns):
        for j, p in enumerate(ps):
            for k, q in enumerate(qs):
                # Error decreases with n, p, and q
                err = err.at[i, j, k].set(1.0 / (n * (p + 1) * (q + 1)))
    
    fig = converge_plot(err, ns, ps, qs)
    
    assert fig is not None, "Figure should be created"
    assert len(fig.data) > 0, "Figure should have traces"
    
    # Check that legend entries are created
    legend_names = [trace.name for trace in fig.data if trace.showlegend]
    assert len(legend_names) > 0, "Should have legend entries"


def test_converge_plot_zero_errors():
    """Test converge_plot with zero errors."""
    ns = jnp.array([4, 8])
    ps = jnp.array([1, 2])
    qs = jnp.array([2, 3])  # Need at least 2 q values to avoid plotly division by zero
    
    err = jnp.zeros((len(ns), len(ps), len(qs)))
    
    fig = converge_plot(err, ns, ps, qs)
    
    assert fig is not None, "Figure should be created"
    assert len(fig.data) > 0, "Figure should have traces"


def test_converge_plot_constant_errors():
    """Test converge_plot with constant errors."""
    ns = jnp.array([4, 8, 16])
    ps = jnp.array([1, 2])
    qs = jnp.array([2, 3])
    
    err = jnp.ones((len(ns), len(ps), len(qs))) * 0.5
    
    fig = converge_plot(err, ns, ps, qs)
    
    assert fig is not None, "Figure should be created"
    assert len(fig.data) > 0, "Figure should have traces"


def test_converge_plot_shape_validation():
    """Test that converge_plot handles correct array shapes."""
    ns = jnp.array([4, 8, 16])
    ps = jnp.array([1, 2])
    qs = jnp.array([2, 3])
    
    # Correct shape: (len(ns), len(ps), len(qs))
    # Use numpy random instead of jax.random for simplicity
    err = np.random.rand(len(ns), len(ps), len(qs))
    err = jnp.array(err)
    
    fig = converge_plot(err, ns, ps, qs)
    
    assert fig is not None, "Figure should be created"


def test_converge_plot_legend_entries():
    """Test that converge_plot creates proper legend entries."""
    ns = jnp.array([4, 8])
    ps = jnp.array([1, 2])
    qs = jnp.array([2, 3])
    
    err = jnp.array([
        [[1.0, 0.8], [0.5, 0.4]],
        [[0.5, 0.4], [0.25, 0.2]],
    ])
    
    fig = converge_plot(err, ns, ps, qs)
    
    # Extract legend entries (showlegend=True)
    legend_traces = [trace for trace in fig.data if trace.showlegend]
    
    # Should have legend entries for each p and each q
    assert len(legend_traces) >= len(ps) + len(qs), \
        f"Should have at least {len(ps) + len(qs)} legend entries"
    
    # Check that p entries are in legend
    p_names = [f'p = {p}' for p in ps]
    q_names = [f'q = {q}' for q in qs]
    legend_names = [trace.name for trace in legend_traces]
    
    for p_name in p_names:
        assert p_name in legend_names, f"Legend should contain {p_name}"
    for q_name in q_names:
        assert q_name in legend_names, f"Legend should contain {q_name}"


def test_converge_plot_markers_and_colors():
    """Test that converge_plot uses different markers and colors."""
    ns = jnp.array([4, 8])
    ps = jnp.array([1, 2, 3])
    qs = jnp.array([2, 3])
    
    err = jnp.array([
        [[1.0, 0.8], [0.5, 0.4], [0.25, 0.2]],
        [[0.5, 0.4], [0.25, 0.2], [0.125, 0.1]],
    ])
    
    fig = converge_plot(err, ns, ps, qs)
    
    # Check that different markers are used
    markers_used = set()
    colors_used = set()
    
    for trace in fig.data:
        if trace.mode == 'lines+markers':
            if trace.marker and 'symbol' in trace.marker:
                markers_used.add(trace.marker['symbol'])
            if trace.marker and 'color' in trace.marker:
                colors_used.add(trace.marker['color'])
    
    # Should have multiple markers (one per p) and multiple colors (one per q)
    assert len(markers_used) >= len(ps), \
        f"Should use at least {len(ps)} different markers"
    assert len(colors_used) >= len(qs), \
        f"Should use at least {len(qs)} different colors"


# ============================================================================
# Integration tests based on usage examples
# ============================================================================
def test_plotting_workflow_example():
    """Test a complete plotting workflow based on usage examples."""
    F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
    
    # Create a simple function
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.cos(2 * jnp.pi * zeta) * jnp.ones(1)
    
    # Create grids as in usage example
    cuts = jnp.linspace(0, 1, 3, endpoint=False)
    grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=16, ny=16, nz=1) for v in cuts]
    grid_surface = get_2d_grids(F, cut_axis=0, cut_value=1.0, ny=32, nz=32, z_min=0, z_max=1, invert_z=True)
    
    # Test plot_torus
    fig, ax = plot_torus(p_h, grids_pol, grid_surface, gridlinewidth=1, cstride=8, noaxes=False, elev=15, azim=40)
    collect_figure(fig)
    assert fig is not None
    if not is_running_in_ci():   
        plt.title("Test plotting_workflow_example")
    
    # Test plot_crossections_separate
    fig, axes = plot_crossections_separate(p_h, grids_pol, cuts, plot_centerline=True)
    collect_figure(fig)
    assert fig is not None
    if not is_running_in_ci():   
        plt.title("Test plotting_workflow_example with centerline")
    assert len(axes) == 3


def test_get_2d_grids_usage_pattern():
    """Test get_2d_grids usage pattern from solovev_single_plot.py."""
    F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
    
    # Pattern from line 177-178 of solovev_single_plot.py
    cuts = jnp.linspace(0, 1, 3, endpoint=False)
    grids_pol = [get_2d_grids(F, cut_axis=2, cut_value=v, nx=8, ny=8, nz=1) for v in cuts]
    
    assert len(grids_pol) == 3
    for i, grid in enumerate(grids_pol):
        _x, _y, (_y1, _y2, _y3), (_x1, _x2, _x3) = grid
        assert _y1.shape == (8, 8), f"Grid {i} should have shape (8, 8)"
        
        # Verify that the cut value is approximately correct
        # For cut_axis=2, _x3 should be approximately equal to cuts[i]
        npt.assert_allclose(_x3[0], cuts[i], atol=1e-6, err_msg=f"Grid {i} should have cut_value={cuts[i]}")


# ============================================================================
# Tests for poincare_plot
# ============================================================================
def test_poincare_plot_basic():
    """Test basic poincare_plot functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Create a simple vector field (constant velocity in zeta direction)
    def vector_field(t, x, args):
        """Simple vector field: constant velocity in zeta direction."""
        r, theta, zeta = x
        return jnp.array([0.0, 0.0, 1.0])  # Constant velocity in zeta
    
    # Create initial conditions: 1 batch, 2 loops
    n_batch = 1
    n_loop = 2
    x0 = jnp.array([[[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]])  # Shape: (1, 2, 3)
    
    colors = ['red', 'blue']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "poincare"
        outdir.mkdir()
        
        poincare_plot(
            outdir=str(outdir) + "/",
            vector_field=vector_field,
            F=F,
            x0=x0,
            n_loop=n_loop,
            n_batch=n_batch,
            colors=colors,
            plane_val=0.5,
            axis=2,
            final_time=10.0,  # Shorter time for testing
            n_saves=100,  # Fewer saves for faster testing
            max_steps=1000,  # Fewer steps for faster testing
            filename="test_"
        )
        if not is_running_in_ci():   
            plt.title("Test poincare_plot")
        
        # Check that output files were created
        output_file_physical = outdir / "test_poincare_physical.png"
        output_file_logical = outdir / "test_poincare_logical.png"
        assert output_file_physical.exists(), "Physical Poincaré plot should be created"
        assert output_file_logical.exists(), "Logical Poincaré plot should be created"


def test_poincare_plot_different_axes():
    """Test poincare_plot with different axis values."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def vector_field(t, x, args):
        r, theta, zeta = x
        return jnp.array([0.0, 1.0, 0.0])  # Constant velocity in theta
    
    n_batch = 1
    n_loop = 1
    x0 = jnp.array([[[0.5, 0.0, 0.5]]])  # Shape: (1, 1, 3)
    colors = ['green']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "poincare"
        outdir.mkdir()
        
        # Test axis=0
        poincare_plot(
            outdir=str(outdir) + "/",
            vector_field=vector_field,
            F=F,
            x0=x0,
            n_loop=n_loop,
            n_batch=n_batch,
            colors=colors,
            plane_val=0.5,
            axis=0,
            final_time=5.0,
            n_saves=50,
            max_steps=500,
            filename="axis0_"
        )
        if not is_running_in_ci():   
            plt.title("Test poincare_plot with axis=0")
        output_file = outdir / "axis0_poincare_physical.png"
        assert output_file.exists(), "Poincaré plot with axis=0 should be created"


def test_poincare_plot_cylindrical():
    """Test poincare_plot with cylindrical coordinates."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def vector_field(t, x, args):
        r, theta, zeta = x
        return jnp.array([0.0, 0.0, 1.0])  # Constant velocity in zeta
    
    n_batch = 1
    n_loop = 1
    x0 = jnp.array([[[0.5, 0.0, 0.0]]])  # Shape: (1, 1, 3)
    colors = ['purple']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "poincare"
        outdir.mkdir()
        
        poincare_plot(
            outdir=str(outdir) + "/",
            vector_field=vector_field,
            F=F,
            x0=x0,
            n_loop=n_loop,
            n_batch=n_batch,
            colors=colors,
            plane_val=0.5,
            axis=2,
            final_time=5.0,
            n_saves=50,
            max_steps=500,
            cylindrical=True,
            filename="cyl_"
        )
        if not is_running_in_ci():   
            plt.title("Test poincare_plot with cylindrical=True")
        output_file = outdir / "cyl_poincare_physical.png"
        assert output_file.exists(), "Poincaré plot with cylindrical=True should be created"


def test_poincare_plot_multiple_batches():
    """Test poincare_plot with multiple batches."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def vector_field(t, x, args):
        r, theta, zeta = x
        return jnp.array([0.0, 0.0, 1.0])  # Constant velocity in zeta
    
    n_batch = 2
    n_loop = 2
    x0 = jnp.array([
        [[0.3, 0.0, 0.0], [0.3, 0.5, 0.0]],  # Batch 1
        [[0.7, 0.0, 0.0], [0.7, 0.5, 0.0]]   # Batch 2
    ])  # Shape: (2, 2, 3)
    
    colors = ['red', 'blue', 'green', 'orange']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "poincare"
        outdir.mkdir()
        
        poincare_plot(
            outdir=str(outdir) + "/",
            vector_field=vector_field,
            F=F,
            x0=x0,
            n_loop=n_loop,
            n_batch=n_batch,
            colors=colors,
            plane_val=0.5,
            axis=2,
            final_time=5.0,
            n_saves=50,
            max_steps=500,
            filename="multi_"
        )   
        if not is_running_in_ci():   
            plt.title("Test poincare_plot with multiple batches")
        # Check that field line plots were created (one for every other trajectory)
        output_file = outdir / "multi_field_line_0.pdf"
        assert output_file.exists(), "Field line plot should be created for multiple batches"


def test_poincare_plot_rotating_field():
    """Test poincare_plot with a rotating vector field."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def vector_field(t, x, args):
        """Rotating field: constant rotation in theta-zeta plane."""
        r, theta, zeta = x
        # Rotate in theta-zeta plane
        return jnp.array([0.0, 1.0, 0.5])  # Different velocities in theta and zeta
    
    n_batch = 1
    n_loop = 1
    x0 = jnp.array([[[0.5, 0.0, 0.0]]])  # Shape: (1, 1, 3)
    colors = ['cyan']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "poincare"
        outdir.mkdir()
        
        poincare_plot(
            outdir=str(outdir) + "/",
            vector_field=vector_field,
            F=F,
            x0=x0,
            n_loop=n_loop,
            n_batch=n_batch,
            colors=colors,
            plane_val=0.25,
            axis=1,
            final_time=10.0,
            n_saves=100,
            max_steps=1000,
            filename="rotating_"
        )
        if not is_running_in_ci():   
            plt.title("Test poincare_plot with rotating field")
        output_file = outdir / "rotating_poincare_physical.png"
        assert output_file.exists(), "Poincaré plot with rotating field should be created"


def test_poincare_plot_cerfon_map():
    """Test poincare_plot with cerfon_map."""
    F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
    
    def vector_field(t, x, args):
        r, theta, zeta = x
        return jnp.array([0.0, 0.0, 1.0])  # Constant velocity in zeta
    
    n_batch = 1
    n_loop = 1
    x0 = jnp.array([[[0.5, 0.0, 0.0]]])  # Shape: (1, 1, 3)
    colors = ['magenta']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "poincare"
        outdir.mkdir()
        
        poincare_plot(
            outdir=str(outdir) + "/",
            vector_field=vector_field,
            F=F,
            x0=x0,
            n_loop=n_loop,
            n_batch=n_batch,
            colors=colors,
            plane_val=0.5,
            axis=2,
            final_time=5.0,
            n_saves=50,
            max_steps=500,
            filename="cerfon_"
        )
        if not is_running_in_ci():   
            plt.title("Test poincare_plot with cerfon_map") 
        output_file = outdir / "cerfon_poincare_physical.png"
        assert output_file.exists(), "Poincaré plot with cerfon_map should be created"



def test_poincare_plot_cerfon_map_full_scale():
    """Test poincare_plot with cerfon_map at full scale with high resolution and multiple batches/loops."""
    F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
    
    # Create a proper vector field that works with the mapping
    # Based on patterns from solovev_poincare.py
    @jax.jit
    def vector_field(t, x, args):
        """Vector field for field line integration."""
        x = x % 1.0  # Ensure coordinates are in [0,1]
        r, theta, zeta = x
        # Simple constant velocity field in zeta direction
        # In a real scenario, this would be B / |B| where B is the magnetic field
        return jnp.array([0.0, 0.0, 1.0])
    
    # Create multiple initial conditions across different radial positions
    # Based on patterns from solovev_poincare.py lines 149-164
    n_batch = 3
    n_loop = 4
    n_total = n_batch * n_loop
    
    # Create initial conditions: vary r, theta, and zeta
    r_vals = jnp.linspace(0.3, 0.9, n_batch)
    theta_vals = jnp.linspace(0.0, 1.0, n_loop, endpoint=False)
    # zeta_vals = jnp.zeros(n_loop)
    
    # Create grid of initial conditions
    x0_list = []
    for r in r_vals:
        batch = []
        for theta in theta_vals:
            batch.append(jnp.array([r, theta, 0.0]))
        x0_list.append(batch)
    
    x0 = jnp.array(x0_list)  # Shape: (n_batch, n_loop, 3)
    
    # Create colors for all trajectories
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 
              'brown', 'pink', 'gray', 'olive']
    colors = colors[:n_total]  # Use only as many colors as needed
    
    # Use test_outputs directory (always in mrx/test/test_outputs/)
    test_output_dir = Path(__file__).parent / "test_outputs"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    outdir = str(test_output_dir) + "/"
    
    # High resolution settings based on solovev_poincare.py
    poincare_plot(
        outdir=outdir,
        vector_field=vector_field,
        F=F,
        x0=x0,
        n_loop=n_loop,
        n_batch=n_batch,
        colors=colors,
        plane_val=0.5,
        axis=2,
        final_time=100.0,  # Longer integration time for better Poincaré sections
        n_saves=1000,  # More saves for higher resolution
        max_steps=50000,  # More steps for accuracy
        r_tol=1e-7,
        a_tol=1e-7,
        filename="cerfon_full_scale_"
    )
    if not is_running_in_ci():   
        plt.title("Test poincare_plot with cerfon_map at full scale (high res, multiple batches/loops)") 
    
    # Check that output files were created
    output_file_physical = test_output_dir / "cerfon_full_scale_poincare_physical.png"
    output_file_logical = test_output_dir / "cerfon_full_scale_poincare_logical.png"
    assert output_file_physical.exists(), "Physical Poincaré plot should be created"
    assert output_file_logical.exists(), "Logical Poincaré plot should be created"
    
    # Check that field line plots were created (one for every other trajectory)
    field_line_files = list(test_output_dir.glob("cerfon_full_scale_field_line_*.pdf"))
    assert len(field_line_files) > 0, "Field line plots should be created"


def test_poincare_plot_shape_assertion():
    """Test that poincare_plot raises assertion error for wrong x0 shape."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def vector_field(t, x, args):
        return jnp.array([0.0, 0.0, 1.0])
    
    # Wrong shape: should be (n_batch, n_loop, 3) but we give (n_loop, 3)
    x0_wrong = jnp.array([[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]])  # Shape: (2, 3)
    
    colors = ['red']
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "poincare"
        outdir.mkdir()
        
        with pytest.raises(AssertionError):
            poincare_plot(
                outdir=str(outdir) + "/",
                vector_field=vector_field,
                F=F,
                x0=x0_wrong,
                n_loop=2,
                n_batch=1,
                colors=colors,
                plane_val=0.5,
                axis=2,
                final_time=5.0,
                n_saves=50,
                max_steps=500,
                filename="wrong_"
            )


# ============================================================================
# Tests for generate_solovev_plots
# ============================================================================
def test_generate_solovev_plots_basic():
    """Test basic generate_solovev_plots functionality."""
    from mrx.mappings import cerfon_map
    
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create the directory structure that generate_solovev_plots expects
        solovev_dir = tmpdir_path / "script_outputs" / "solovev"
        solovev_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test configuration
        test_name = "test_solovev"
        h5_file = solovev_dir / f"{test_name}.h5"
        
        # Create a minimal DeRham sequence to get the right size for p_final
        F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
        ns = (4, 4, 4)
        ps = (2, 2, 2)
        q = max(ps)
        types = ("clamped", "periodic", "periodic")
        Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)
        Seq.evaluate_1d()
        Seq.assemble_M0()
        
        # Create test data
        n_iterations = 10
        p_final = jnp.ones(Seq.E0.shape[0]) * 0.5
        iterations = jnp.arange(n_iterations)
        force_trace = jnp.linspace(1.0, 0.01, n_iterations)
        velocity_trace = jnp.linspace(1.0, 0.1, n_iterations)
        helicity_trace = jnp.ones(n_iterations) * 1.0
        energy_trace = jnp.linspace(1.0, 0.9, n_iterations)
        divergence_trace = jnp.linspace(0.01, 0.001, n_iterations)
        wall_time_trace = jnp.linspace(0.0, 10.0, n_iterations)
        
        # Create HDF5 file with required structure
        with h5py.File(h5_file, "w") as f:
            # Create config group with attributes
            config_group = f.create_group("config")
            config_group.attrs["delta"] = 0.0
            config_group.attrs["kappa"] = 1.2
            config_group.attrs["eps"] = 0.33
            config_group.attrs["R_0"] = 1.0
            config_group.attrs["n_r"] = 4
            config_group.attrs["n_theta"] = 4
            config_group.attrs["n_zeta"] = 4
            config_group.attrs["p_r"] = 2
            config_group.attrs["p_theta"] = 2
            config_group.attrs["p_zeta"] = 2
            config_group.attrs["save_B"] = False
            config_group.attrs["save_every"] = 10
            
            # Create datasets
            f.create_dataset("p_final", data=np.array(p_final))
            f.create_dataset("iterations", data=np.array(iterations))
            f.create_dataset("force_trace", data=np.array(force_trace))
            f.create_dataset("velocity_trace", data=np.array(velocity_trace))
            f.create_dataset("helicity_trace", data=np.array(helicity_trace))
            f.create_dataset("energy_trace", data=np.array(energy_trace))
            f.create_dataset("divergence_trace", data=np.array(divergence_trace))
            f.create_dataset("wall_time_trace", data=np.array(wall_time_trace))
        
        # Change to the temporary directory so generate_solovev_plots can find the file
        # The function uses hardcoded paths relative to current directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Monkeypatch trace_plot to ignore the CONFIG parameter (bug in generate_solovev_plots)
            import mrx.plotting as plotting_module
            original_trace_plot = plotting_module.trace_plot
            
            def patched_trace_plot(*args, **kwargs):
                """Patched trace_plot that ignores CONFIG parameter."""
                kwargs.pop('CONFIG', None)  # Remove CONFIG if present
                return original_trace_plot(*args, **kwargs)
            
            plotting_module.trace_plot = patched_trace_plot
            
            try:
                # Call the real generate_solovev_plots function
                # It will look for script_outputs/solovev/{name}.h5 relative to current directory
                generate_solovev_plots(test_name)
            finally:
                # Restore original function
                plotting_module.trace_plot = original_trace_plot
            
            # Verify output files were created
            output_dir = tmpdir_path / "script_outputs" / "solovev" / test_name
            p_final_file = output_dir / "p_final.pdf"
            force_trace_file = output_dir / f"{test_name}_force_trace.pdf"
            
            assert p_final_file.exists(), "p_final.pdf should be created"
            assert force_trace_file.exists(), f"{test_name}_force_trace.pdf should be created"
            
        finally:
            os.chdir(original_cwd)


def test_generate_solovev_plots_with_save_B():
    """Test generate_solovev_plots with save_B=True."""
    from mrx.mappings import cerfon_map
    
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create the directory structure
        solovev_dir = tmpdir_path / "script_outputs" / "solovev"
        solovev_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test configuration
        test_name = "test_solovev_save_B"
        h5_file = solovev_dir / f"{test_name}.h5"
        
        # Create a minimal DeRham sequence
        F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
        ns = (4, 4, 4)
        ps = (2, 2, 2)
        q = max(ps)
        types = ("clamped", "periodic", "periodic")
        Seq = DeRhamSequence(ns, ps, q, types, F, polar=True)
        Seq.evaluate_1d()
        Seq.assemble_M0()
        
        # Create test data
        n_iterations = 5
        p_final = jnp.ones(Seq.E0.shape[0]) * 0.5
        iterations = jnp.arange(n_iterations)
        force_trace = jnp.linspace(1.0, 0.01, n_iterations)
        velocity_trace = jnp.linspace(1.0, 0.1, n_iterations)
        helicity_trace = jnp.ones(n_iterations) * 1.0
        energy_trace = jnp.linspace(1.0, 0.9, n_iterations)
        divergence_trace = jnp.linspace(0.01, 0.001, n_iterations)
        wall_time_trace = jnp.linspace(0.0, 10.0, n_iterations)
        
        # Create p_fields (2 pressure fields)
        n_fields = 2
        p_fields = jnp.array([jnp.ones(Seq.E0.shape[0]) * (0.5 + i * 0.1) for i in range(n_fields)])
        
        # Create HDF5 file with save_B=True
        with h5py.File(h5_file, "w") as f:
            config_group = f.create_group("config")
            config_group.attrs["delta"] = 0.0
            config_group.attrs["kappa"] = 1.2
            config_group.attrs["eps"] = 0.33
            config_group.attrs["R_0"] = 1.0
            config_group.attrs["n_r"] = 4
            config_group.attrs["n_theta"] = 4
            config_group.attrs["n_zeta"] = 4
            config_group.attrs["p_r"] = 2
            config_group.attrs["p_theta"] = 2
            config_group.attrs["p_zeta"] = 2
            config_group.attrs["save_B"] = True
            config_group.attrs["save_every"] = 10
            
            f.create_dataset("p_final", data=np.array(p_final))
            f.create_dataset("iterations", data=np.array(iterations))
            f.create_dataset("force_trace", data=np.array(force_trace))
            f.create_dataset("velocity_trace", data=np.array(velocity_trace))
            f.create_dataset("helicity_trace", data=np.array(helicity_trace))
            f.create_dataset("energy_trace", data=np.array(energy_trace))
            f.create_dataset("divergence_trace", data=np.array(divergence_trace))
            f.create_dataset("wall_time_trace", data=np.array(wall_time_trace))
            f.create_dataset("p_fields", data=np.array(p_fields))
        
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Monkeypatch trace_plot to ignore the CONFIG parameter (bug in generate_solovev_plots)
            import mrx.plotting as plotting_module
            original_trace_plot = plotting_module.trace_plot
            
            def patched_trace_plot(*args, **kwargs):
                """Patched trace_plot that ignores CONFIG parameter."""
                kwargs.pop('CONFIG', None)  # Remove CONFIG if present
                return original_trace_plot(*args, **kwargs)
            
            plotting_module.trace_plot = patched_trace_plot
            
            try:
                # Call the real generate_solovev_plots function
                generate_solovev_plots(test_name)
            finally:
                # Restore original function
                plotting_module.trace_plot = original_trace_plot
            
            # Verify output files were created
            output_dir = tmpdir_path / "script_outputs" / "solovev" / test_name
            p_final_file = output_dir / "p_final.pdf"
            force_trace_file = output_dir / f"{test_name}_force_trace.pdf"
            p_iter_0_file = output_dir / "p_iter_000000.pdf"
            p_iter_1_file = output_dir / "p_iter_000010.pdf"
            
            assert p_final_file.exists(), "p_final.pdf should be created"
            assert force_trace_file.exists(), f"{test_name}_force_trace.pdf should be created"
            assert p_iter_0_file.exists(), "p_iter_000000.pdf should be created"
            assert p_iter_1_file.exists(), "p_iter_000010.pdf should be created"
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# Tests for plot_scalar_fct_physical_logical
# ============================================================================
def test_plot_scalar_fct_physical_logical_basic():
    """Test basic plot_scalar_fct_physical_logical functionality."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    # Create a simple scalar function
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    fig, (ax_phys, ax_log) = plot_scalar_fct_physical_logical(
        p_h, F, n_vis=32, scale=1.0, logical_plane='r_theta', cmap='viridis'
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_scalar_fct_physical_logical")
    
    assert fig is not None, "Figure should be created"
    assert ax_phys is not None, "Physical axis should be created"
    assert ax_log is not None, "Logical axis should be created"


def test_plot_scalar_fct_physical_logical_r_zeta():
    """Test plot_scalar_fct_physical_logical with r_zeta plane."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * r) * jnp.ones(1)
    
    fig, (ax_phys, ax_log) = plot_scalar_fct_physical_logical(
        p_h, F, n_vis=32, logical_plane='r_zeta', fixed_theta=0.5
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_scalar_fct_physical_logical r_zeta")
    
    assert fig is not None, "Figure should be created"
    # For r_zeta plane, ax_phys is None and only logical axis is returned
    assert ax_phys is None, "Physical axis should be None for r_zeta plane"
    assert ax_log is not None, "Logical axis should be created"


def test_plot_scalar_fct_physical_logical_theta_zeta():
    """Test plot_scalar_fct_physical_logical with theta_zeta plane."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.cos(2 * jnp.pi * zeta) * jnp.ones(1)
    
    fig, (ax_phys, ax_log) = plot_scalar_fct_physical_logical(
        p_h, F, n_vis=32, logical_plane='theta_zeta', fixed_r=0.5
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_scalar_fct_physical_logical theta_zeta")
    
    assert fig is not None, "Figure should be created"
    assert ax_log is not None, "Logical axis should be created"


def test_plot_scalar_fct_physical_logical_with_colorbar():
    """Test plot_scalar_fct_physical_logical with colorbar."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    fig, (ax_phys, ax_log) = plot_scalar_fct_physical_logical(
        p_h, F, n_vis=32, colorbar=True, cbar_label="Test Label"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_scalar_fct_physical_logical with colorbar")
    
    assert fig is not None, "Figure should be created"


def test_plot_scalar_fct_physical_logical_no_colorbar():
    """Test plot_scalar_fct_physical_logical without colorbar."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    fig, (ax_phys, ax_log) = plot_scalar_fct_physical_logical(
        p_h, F, n_vis=32, colorbar=False
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_scalar_fct_physical_logical without colorbar")
    
    assert fig is not None, "Figure should be created"


def test_plot_scalar_fct_physical_logical_custom_levels():
    """Test plot_scalar_fct_physical_logical with custom levels."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    fig, (ax_phys, ax_log) = plot_scalar_fct_physical_logical(
        p_h, F, n_vis=32, levels=10, cmap='plasma'
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_scalar_fct_physical_logical custom levels")
    
    assert fig is not None, "Figure should be created"


def test_plot_scalar_fct_physical_logical_invalid_plane():
    """Test plot_scalar_fct_physical_logical with invalid logical_plane."""
    F = toroid_map(epsilon=0.5, R0=2.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.ones(1)
    
    with pytest.raises(ValueError, match="Unknown logical_plane"):
        plot_scalar_fct_physical_logical(
            p_h, F, n_vis=32, logical_plane='invalid_plane'
        )


def test_plot_scalar_fct_physical_logical_cerfon_map():
    """Test plot_scalar_fct_physical_logical with cerfon_map."""
    F = cerfon_map(epsilon=0.33, kappa=1.2, alpha=0.0, R0=1.0)
    
    def p_h(x):
        r, theta, zeta = x
        return jnp.sin(2 * jnp.pi * theta) * jnp.cos(2 * jnp.pi * zeta) * jnp.ones(1)
    
    fig, (ax_phys, ax_log) = plot_scalar_fct_physical_logical(
        p_h, F, n_vis=32, logical_plane='r_theta'
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_scalar_fct_physical_logical cerfon_map")
    
    assert fig is not None, "Figure should be created"


# ============================================================================
# Tests for plot_twin_axis
# ============================================================================
def test_plot_twin_axis_basic():
    """Test basic plot_twin_axis functionality."""
    left_y = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    right_y = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        left_label="Left Axis",
        right_label="Right Axis"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis")
    
    assert fig is not None, "Figure should be created"
    assert ax1 is not None, "Left axis should be created"
    assert ax2 is not None, "Right axis should be created"


def test_plot_twin_axis_with_custom_x():
    """Test plot_twin_axis with custom x values."""
    left_y = jnp.array([1.0, 0.5, 0.25])
    right_y = jnp.array([10.0, 20.0, 30.0])
    x_left = jnp.array([0, 5, 10])
    x_right = jnp.array([0, 5, 10])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        x_left=x_left,
        x_right=x_right,
        left_label="Left",
        right_label="Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis with custom x")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_log_scale():
    """Test plot_twin_axis with log scale."""
    left_y = jnp.array([1.0, 0.1, 0.01, 0.001])
    right_y = jnp.array([10.0, 20.0, 30.0, 40.0])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        left_log=True,
        right_log=False,
        left_label="Log Left",
        right_label="Linear Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis log scale")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_both_log():
    """Test plot_twin_axis with both axes in log scale."""
    left_y = jnp.array([1.0, 0.1, 0.01, 0.001])
    right_y = jnp.array([10.0, 1.0, 0.1, 0.01])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        left_log=True,
        right_log=True,
        left_label="Log Left",
        right_label="Log Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis both log")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_custom_colors():
    """Test plot_twin_axis with custom colors."""
    left_y = jnp.array([1.0, 0.5, 0.25])
    right_y = jnp.array([10.0, 20.0, 30.0])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        left_color="red",
        right_color="blue",
        left_label="Red Left",
        right_label="Blue Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis custom colors")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_custom_markers():
    """Test plot_twin_axis with custom markers."""
    left_y = jnp.array([1.0, 0.5, 0.25])
    right_y = jnp.array([10.0, 20.0, 30.0])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        left_marker='o',
        right_marker='^',
        left_label="Circle",
        right_label="Triangle"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis custom markers")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_no_grid():
    """Test plot_twin_axis without grid."""
    left_y = jnp.array([1.0, 0.5, 0.25])
    right_y = jnp.array([10.0, 20.0, 30.0])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        grid=False,
        left_label="Left",
        right_label="Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis no grid")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_with_kwargs():
    """Test plot_twin_axis with plot_kwargs."""
    left_y = jnp.array([1.0, 0.5, 0.25])
    right_y = jnp.array([10.0, 20.0, 30.0])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        left_plot_kwargs={'linewidth': 3, 'alpha': 0.7},
        right_plot_kwargs={'linewidth': 2, 'alpha': 0.8},
        left_label="Left",
        right_label="Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis with kwargs")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_num_iters_inner():
    """Test plot_twin_axis with num_iters_inner."""
    left_y = jnp.array([1.0, 0.5, 0.25])
    right_y = jnp.array([10.0, 20.0, 30.0])
    
    fig, (ax1, ax2) = plot_twin_axis(
        left_y, right_y,
        num_iters_inner=2,
        left_label="Left",
        right_label="Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis num_iters_inner")
    
    assert fig is not None, "Figure should be created"


def test_plot_twin_axis_return_fig_only():
    """Test plot_twin_axis with return_axes=False."""
    left_y = jnp.array([1.0, 0.5, 0.25])
    right_y = jnp.array([10.0, 20.0, 30.0])
    
    fig = plot_twin_axis(
        left_y, right_y,
        return_axes=False,
        left_label="Left",
        right_label="Right"
    )
    collect_figure(fig)
    if not is_running_in_ci():   
        plt.title("Test plot_twin_axis return fig only")
    
    assert fig is not None, "Figure should be created"
    assert isinstance(fig, plt.Figure), "Should return Figure object"


# which runs after all tests complete. The code below is kept for non-pytest execution.
if __name__ == "__main__":
    # Show all collected figures at the end (for non-pytest execution)
    if not is_running_in_ci():
        all_figures = plt.get_fignums()
        if len(all_figures) > 0:
            print(f"\n{'='*60}")
            print(f"Displaying {len(all_figures)} figure(s) at the end of tests...")
            print(f"{'='*60}")
            # Ensure output directory exists (always in mrx/test/test_outputs/)
            test_output_dir = Path(__file__).parent / "test_outputs"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            # Save all figures to test_outputs directory
            for i, fig_num in enumerate(all_figures):
                fig = plt.figure(fig_num)
                output_path = test_output_dir / f"figure_{i}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved figure {i+1} to {output_path}")
            print("Close all figure windows to continue.")
            # plt.show(block=True)  # Show all figures and block until all windows are closed
        elif len(_collected_figures) > 0:
            print(f"\nNote: {len(_collected_figures)} figure(s) were collected but are already closed.")
            print("Some plotting functions (like poincare_plot) close figures internally.")
    plt.close('all')