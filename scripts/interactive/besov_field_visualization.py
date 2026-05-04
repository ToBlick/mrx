# %% [markdown]
# # Random Besov Field Visualization
#
# Lightweight interactive script that:
#
# 1. builds a sparse random Besov-like field in physical Cartesian coordinates,
# 2. pulls it back to the logical domain with the analytic rotating-ellipse map,
# 3. pushes it forward again for physical visualization,
# 4. and plots several 2-D slices in both logical and physical coordinates.

# %%
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from mrx.differential_forms import Pullback, Pushforward
from mrx.mappings import rotating_ellipse_map
from mrx.utils import build_random_besov_function

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class Config:
    form_degree: int = 0
    sobolev_order: float = 1.0
    upper_limit: int = 50
    num_modes: int = 256
    scale: float = 1.0
    smoothness_margin: float = 0.0
    normalization_samples: int = 256
    seed: int = 0
    zeta_cuts: tuple[float, ...] = (0.0, 0.25, 0.5)
    logical_nr: int = 96
    logical_ntheta: int = 128


CONFIG = Config()
ANALYTIC_MAP = None
LOGICAL_FIELD = None
PHYSICAL_FIELD = None


# %% Helpers
def _build_analytic_map(config: Config = CONFIG):
    return rotating_ellipse_map()


def _field_quantity(values: jnp.ndarray, form_degree: int) -> jnp.ndarray:
    if form_degree in (0, 3):
        return values[..., 0]
    return jnp.linalg.norm(values, axis=-1)


def _slice_axis_labels(cut_axis: int) -> tuple[str, str]:
    if cut_axis == 0:
        return r"$\theta$", r"$\zeta$"
    if cut_axis == 1:
        return r"$r$", r"$\zeta$"
    return r"$r$", r"$\theta$"


def _evaluate_field_on_points(field, points: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(field)(points)


def _build_logical_slice_grid(zeta_cut: float, config: Config = CONFIG):
    r_axis = jnp.linspace(0.0, 1.0, config.logical_nr)
    # Exclude the periodic endpoint to avoid a duplicated seam at theta=0=1.
    theta_axis = jnp.linspace(0.0, 1.0, config.logical_ntheta, endpoint=False)
    r_grid, theta_grid = jnp.meshgrid(r_axis, theta_axis, indexing="ij")
    zeta_grid = jnp.full_like(r_grid, zeta_cut)
    logical_points = jnp.stack([r_grid, theta_grid, zeta_grid], axis=-1).reshape(-1, 3)
    return logical_points, r_grid, theta_grid


def _plot_logical_slices(config: Config = CONFIG):
    fig, axes = plt.subplots(
        1,
        len(config.zeta_cuts),
        figsize=(5 * len(config.zeta_cuts), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, zeta_cut in zip(axes, config.zeta_cuts):
        logical_points, r_grid, theta_grid = _build_logical_slice_grid(
            zeta_cut, config)
        logical_values = _evaluate_field_on_points(LOGICAL_FIELD, logical_points)
        quantity = _field_quantity(logical_values, config.form_degree).reshape(
            config.logical_nr, config.logical_ntheta
        )
        image = ax.pcolormesh(
            np.asarray(r_grid),
            np.asarray(theta_grid),
            np.asarray(quantity),
            shading="auto",
            cmap="viridis",
        )
        xlabel, ylabel = _slice_axis_labels(2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(rf"Logical slice at $\zeta={zeta_cut:.2f}$")
        fig.colorbar(image, ax=ax, shrink=0.85)
    return fig, axes


def _plot_physical_slices(config: Config = CONFIG):
    fig, axes = plt.subplots(
        1,
        len(config.zeta_cuts),
        figsize=(5 * len(config.zeta_cuts), 4.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, zeta_cut in zip(axes, config.zeta_cuts):
        logical_points, _, _ = _build_logical_slice_grid(zeta_cut, config)
        mapped_points = _evaluate_field_on_points(ANALYTIC_MAP, logical_points)
        physical_values = _evaluate_field_on_points(PHYSICAL_FIELD, logical_points)
        quantity = _field_quantity(physical_values, config.form_degree).reshape(
            config.logical_nr, config.logical_ntheta
        )
        mapped_points = np.asarray(mapped_points).reshape(config.logical_nr, config.logical_ntheta, 3)
        x_phys = mapped_points[..., 0]
        y_phys = mapped_points[..., 1]
        z_phys = mapped_points[..., 2]
        radius = np.sqrt(x_phys ** 2 + y_phys ** 2)
        image = ax.pcolormesh(
            radius,
            z_phys,
            np.asarray(quantity),
            shading="auto",
            cmap="viridis",
        )
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Z$")
        ax.set_title(rf"Physical slice at $\zeta={zeta_cut:.2f}$")
        fig.colorbar(image, ax=ax, shrink=0.85)
    return fig, axes


# %% Build the analytic map and the logical/physical fields
ANALYTIC_MAP = _build_analytic_map()
PHYSICAL_SOURCE_FIELD = build_random_besov_function(
    CONFIG.form_degree,
    key=jax.random.PRNGKey(CONFIG.seed),
    s=CONFIG.sobolev_order,
    upper_limit=CONFIG.upper_limit,
    num_modes=CONFIG.num_modes,
    scale=CONFIG.scale,
    smoothness_margin=CONFIG.smoothness_margin,
    normalization_samples=CONFIG.normalization_samples,
)

LOGICAL_FIELD = Pullback(PHYSICAL_SOURCE_FIELD, ANALYTIC_MAP, CONFIG.form_degree)
PHYSICAL_FIELD = Pushforward(LOGICAL_FIELD, ANALYTIC_MAP, CONFIG.form_degree)

print(
    "built Besov visualization field:",
    f"k={CONFIG.form_degree}, defined_in=physical_xyz, "
    f"upper_limit={CONFIG.upper_limit}, num_modes={CONFIG.num_modes}, "
    f"zeta_cuts={CONFIG.zeta_cuts}"
)


# %% Plot slices in the logical domain
_plot_logical_slices()


# %% Plot the corresponding slices in the physical domain
_plot_physical_slices()
# %%
