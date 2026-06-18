# %% [markdown]
# # Hcurl Nullspace Visualization
#
# Interactive script that:
#
# 1. builds a de Rham sequence on the analytical rotating ellipse geometry,
# 2. assembles the operators needed for the free `k=1` Hcurl nullspace,
# 3. computes the nullspace vector,
# 4. plots a poloidal cross-section of the pushed-forward physical field,
# 5. plots the outer torus surface colored by `|null_vector|`.

# %%
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mrx
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import extend_map_nfp, rotating_ellipse_map
from mrx.operators import (assemble_derivative_operators,
                           assemble_laplacian_operators,
                           assemble_incidence_operators,
                           assemble_mass_operators,
                           assemble_tensor_mass_preconditioner)
from mrx.plotting import get_2d_grids, set_axes_equal

jax.config.update("jax_enable_x64", True)


# %% Configuration
@dataclass(frozen=True)
class Config:
    ns: tuple[int, int, int] = (6, 8, 4)
    p: int = 3
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    tol: float = 1e-9
    maxiter: int = 200
    nullspace_eps: float = 1e-6
    rotating_eps: float = 0.2
    rotating_kappa: float = 1.4
    rotating_r0: float = 1.0
    rotating_nfp: int = 3
    jacobian_pos_tol: float = 0.0
    zeta_cuts: tuple[float, ...] = (0.0, 0.2, 0.4)
    cut_nx: int = 48
    cut_ny: int = 64
    surface_ntheta: int = 128
    surface_nzeta: int = 128
    surface_quiver_stride_theta: int = 8
    surface_quiver_stride_zeta: int = 8
    surface_quiver_length: float = 0.12
    surface_quiver_offset: float = 0.02
    quiver_stride_r: int = 3
    quiver_stride_t: int = 4


CONFIG = Config()
SEQ = None
OPERATORS = None
ANALYTIC_MAP = None
FULL_ANALYTIC_MAP = None
NULL_VECTOR = None
NULL_FIELD = None
FULL_NULL_FIELD = None
DENSE_EIGENVALUES = None


# %% Helpers
def _build_rotating_ellipse_map(config: Config = CONFIG):
    return rotating_ellipse_map(
        eps=config.rotating_eps,
        kappa=config.rotating_kappa,
        R0=config.rotating_r0,
        nfp=config.rotating_nfp,
    )


def _extend_to_full_torus(map_one_period, config: Config = CONFIG):
    return extend_map_nfp(map_one_period, config.rotating_nfp)


def _require_sequence() -> DeRhamSequence:
    if SEQ is None:
        raise RuntimeError("Sequence is not built yet; run build_case() first")
    return SEQ


def _require_operators():
    if OPERATORS is None:
        raise RuntimeError("Operators are not assembled yet; run assemble_case() first")
    return OPERATORS


def _require_null_field():
    if NULL_FIELD is None:
        raise RuntimeError("Null field is not available yet; run compute_hcurl_nullspace() first")
    return NULL_FIELD


def _require_full_null_field():
    if FULL_NULL_FIELD is None:
        raise RuntimeError("Full-torus null field is not available yet; run compute_hcurl_nullspace() first")
    return FULL_NULL_FIELD


def _require_null_vector():
    if NULL_VECTOR is None:
        raise RuntimeError("Null vector is not available yet; run compute_hcurl_nullspace() first")
    return NULL_VECTOR


def _evaluate_on_points(field, points):
    return jax.lax.map(field, points, batch_size=mrx.MAP_BATCH_SIZE_OUTER)


def _evaluate_field_on_grid(field, grid):
    values = _evaluate_on_points(field, grid[0]).reshape(*grid[2][0].shape, 3)
    return values


def _localize_full_torus_point(x, nfp: int):
    x = jnp.asarray(x, dtype=jnp.float64)
    xi = x[2] * nfp
    zeta_local = xi - jnp.floor(xi)
    return x.at[2].set(zeta_local)


def _localize_full_torus_zeta(zeta: float, nfp: int):
    xi = zeta * nfp
    return float(xi - np.floor(xi))


def _build_poloidal_slice_grid(zeta_cut: float, config: Config = CONFIG):
    r_axis = jnp.linspace(1e-6, 1.0 - 1e-6, config.cut_nx)
    theta_axis = jnp.linspace(0.0, 1.0, config.cut_ny, endpoint=False)
    r_grid, theta_grid = jnp.meshgrid(r_axis, theta_axis, indexing="ij")
    zeta_grid = jnp.full_like(r_grid, zeta_cut)
    logical_points = jnp.stack([r_grid, theta_grid, zeta_grid], axis=-1).reshape(-1, 3)
    return logical_points, r_grid, theta_grid, zeta_grid


def _build_full_torus_pushforward(logical_field, map_one_period, config: Config = CONFIG):
    full_map = _extend_to_full_torus(map_one_period, config)

    def localized_logical_field(x):
        return logical_field(_localize_full_torus_point(x, config.rotating_nfp))

    return full_map, Pushforward(localized_logical_field, full_map, 1)


def _surface_colors(values):
    values_np = np.asarray(values)
    vmin = float(values_np.min())
    vmax = float(values_np.max())
    if vmax <= vmin:
        vmax = vmin + 1e-12
    normalized = (values_np - vmin) / (vmax - vmin)
    return plt.cm.plasma(normalized), vmin, vmax


def _surface_view_direction(elev_deg: float, azim_deg: float):
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    return np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ])


def _surface_normals(X, Y, Z, major_radius: float):
    tangent_theta = np.stack([
        np.gradient(X, axis=0),
        np.gradient(Y, axis=0),
        np.gradient(Z, axis=0),
    ], axis=-1)
    tangent_zeta = np.stack([
        np.gradient(X, axis=1),
        np.gradient(Y, axis=1),
        np.gradient(Z, axis=1),
    ], axis=-1)
    normals = np.cross(tangent_theta, tangent_zeta)
    normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_norm = np.where(normal_norm > 0.0, normal_norm, 1.0)
    normals = normals / normal_norm

    phi = np.arctan2(-Y, X)
    centerline = np.stack([
        major_radius * np.cos(phi),
        -major_radius * np.sin(phi),
        np.zeros_like(Z),
    ], axis=-1)
    outward = np.stack([X, Y, Z], axis=-1) - centerline
    flip_mask = np.sum(normals * outward, axis=-1, keepdims=True) < 0.0
    normals = np.where(flip_mask, -normals, normals)
    return normals


def _symmetrize_dense(matrix):
    return 0.5 * (matrix + matrix.T)


def _build_dense_from_matvec(matvec, n_cols):
    def build_column(col_idx):
        basis = jnp.zeros(n_cols).at[col_idx].set(1.0)
        return jnp.asarray(matvec(basis))

    columns = jax.lax.map(
        build_column,
        jnp.arange(n_cols),
        batch_size=mrx.MAP_BATCH_SIZE_OUTER,
    )
    return jnp.swapaxes(columns, 0, 1)


def _l2_normalize(seq: DeRhamSequence, vector, k: int, *, dirichlet: bool):
    norm = float(seq.l2_norm(vector, k, dirichlet=dirichlet))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return jnp.asarray(vector) / norm


def _l2_distance_up_to_sign(seq: DeRhamSequence, left, right, k: int, *, dirichlet: bool):
    left = _l2_normalize(seq, left, k, dirichlet=dirichlet)
    right = _l2_normalize(seq, right, k, dirichlet=dirichlet)
    plus = left + right
    minus = left - right
    return min(
        float(seq.l2_norm(plus, k, dirichlet=dirichlet)),
        float(seq.l2_norm(minus, k, dirichlet=dirichlet)),
    )


def _check_positive_jacobian(seq: DeRhamSequence, *, tol: float):
    jacobian = jnp.asarray(seq.jacobian_j)
    min_j = float(jnp.min(jacobian))
    max_j = float(jnp.max(jacobian))
    n_nonpositive = int(jnp.sum(jacobian <= tol))
    print(
        "jacobian_check:",
        f"min={min_j:.6e}, max={max_j:.6e}, nonpositive_count={n_nonpositive}/{jacobian.size}, tol={tol:.3e}",
    )
    if n_nonpositive > 0:
        bad_idx = int(jnp.argmin(jacobian))
        bad_x = seq.quad.x[bad_idx]
        bad_val = float(jacobian[bad_idx])
        raise ValueError(
            "Analytical map geometry has non-positive Jacobian determinant at "
            f"quadrature index {bad_idx}, x={bad_x.tolist()}, detJ={bad_val:.6e}"
        )


# %% Stage 1: build the sequence on the analytical map
def build_case(config: Config = CONFIG):
    global SEQ, ANALYTIC_MAP, FULL_ANALYTIC_MAP, OPERATORS, NULL_VECTOR, NULL_FIELD, FULL_NULL_FIELD

    SEQ = DeRhamSequence(
        config.ns,
        (config.p, config.p, config.p),
        2 * config.p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=config.tol,
        maxiter=config.maxiter,
        betti_numbers=config.betti,
    )
    SEQ.evaluate_1d()
    SEQ.assemble_reference_mass_matrix()

    ANALYTIC_MAP = _build_rotating_ellipse_map(config)
    FULL_ANALYTIC_MAP = _extend_to_full_torus(ANALYTIC_MAP, config)
    SEQ.set_map(ANALYTIC_MAP)
    _check_positive_jacobian(SEQ, tol=config.jacobian_pos_tol)

    OPERATORS = None
    NULL_VECTOR = None
    NULL_FIELD = None
    FULL_NULL_FIELD = None

    print(
        "built analytical-map sequence:",
        f"ns={config.ns}, p={config.p}, nfp={config.rotating_nfp}"
    )
    return SEQ, ANALYTIC_MAP


# %% Stage 2: assemble what is needed for the free Hcurl nullspace
def assemble_case():
    global OPERATORS
    seq = _require_sequence()
    operators = assemble_mass_operators(seq, seq.geometry, ks=(0, 1, 2))
    operators = assemble_tensor_mass_preconditioner(
        seq,
        operators=operators,
        ks=(0, 1),
    )
    operators = assemble_incidence_operators(seq, operators, ks=(0, 1))
    operators = assemble_derivative_operators(
        seq,
        seq.geometry,
        operators,
        ks=(0, 1),
    )
    OPERATORS = assemble_laplacian_operators(
        seq,
        seq.geometry,
        operators,
        ks=(0, 1),
    )
    seq.set_operators(OPERATORS)
    print(
        "assembled the free k=1 nullspace operator slice: "
        "mass(0,1,2), tensor mass precond(0,1), incidence(0,1), derivative(0,1), hodge(0,1)"
    )
    return OPERATORS


# %% Stage 3: compute the free k=1 nullspace
def compute_hcurl_nullspace(config: Config = CONFIG):
    global DENSE_EIGENVALUES, NULL_VECTOR, NULL_FIELD, FULL_NULL_FIELD
    seq = _require_sequence()
    operators = _require_operators()
    n0 = seq.n0
    n1 = seq.n1

    m0 = np.asarray(_symmetrize_dense(_build_dense_from_matvec(
        lambda v: seq.apply_mass_matrix(v, 0, dirichlet=False, operators=operators),
        n0,
    )))
    k1 = np.asarray(_symmetrize_dense(_build_dense_from_matvec(
        lambda v: seq.apply_stiffness(v, 1, dirichlet=False, operators=operators),
        n1,
    )))
    d0 = np.asarray(_build_dense_from_matvec(
        lambda v: seq.apply_derivative_matrix(
            v,
            0,
            dirichlet_in=False,
            dirichlet_out=False,
            operators=operators,
        ),
        n0,
    ))
    schur_l1 = _symmetrize_dense(k1 + d0 @ np.linalg.solve(m0, d0.T))

    dense_eigvals, dense_eigvecs = np.linalg.eigh(schur_l1)
    DENSE_EIGENVALUES = np.asarray(dense_eigvals[: min(3, dense_eigvals.size)])
    dense_null_vector = _l2_normalize(seq, dense_eigvecs[:, 0], 1, dirichlet=False)
    DENSE_EIGENVALUES = np.asarray(DENSE_EIGENVALUES[: min(3, DENSE_EIGENVALUES.size)])
    print(
        "dense eigenvalues of the k=1 Schur complement (smallest first):",
        DENSE_EIGENVALUES.tolist(),
    )

    nullspace_vectors, info = seq._find_nullspace_vectors(
        1,
        1,
        config.nullspace_eps,
        dirichlet=False,
    )
    if nullspace_vectors.shape[0] == 0:
        raise RuntimeError("The library nullspace routine returned no free k=1 vectors")

    NULL_VECTOR = nullspace_vectors[0]
    vector_mismatch = _l2_distance_up_to_sign(seq, dense_null_vector, NULL_VECTOR, 1, dirichlet=False)
    if vector_mismatch > seq.tol**0.5:
        print(
            "Dense Schur-complement null vector and library nullspace vector disagree in M1 norm: "
            f"mismatch={vector_mismatch:.6e}, tol={seq.tol**0.5:.6e}"
        )

    NULL_FIELD = Pushforward(
        DiscreteFunction(NULL_VECTOR, seq.basis_1, seq.e1),
        ANALYTIC_MAP,
        1,
    )
    _, FULL_NULL_FIELD = _build_full_torus_pushforward(
        DiscreteFunction(NULL_VECTOR, seq.basis_1, seq.e1),
        ANALYTIC_MAP,
        config,
    )

    print("computed free k=1 nullspace with the library single-vector routine")
    print(f"null vector M1 norm = {float(seq.l2_norm(NULL_VECTOR, 1, dirichlet=False)):.6e}")
    print(f"dense-vs-library M1 mismatch = {vector_mismatch:.6e}")
    print(f"nullspace info: {info}")
    return NULL_VECTOR, NULL_FIELD, DENSE_EIGENVALUES


# %% Stage 4: plot the null vector on a logical poloidal cut
def plot_logical_cross_section(config: Config = CONFIG):
    seq = _require_sequence()
    null_vector = _require_null_vector()
    logical_field = DiscreteFunction(null_vector, seq.basis_1, seq.e1)
    fig, axes = plt.subplots(
        1,
        len(config.zeta_cuts),
        figsize=(5 * len(config.zeta_cuts), 4.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, zeta_cut in zip(axes, config.zeta_cuts):
        zeta_local = _localize_full_torus_zeta(zeta_cut, config.rotating_nfp)
        logical_points, r_grid, theta_grid, _ = _build_poloidal_slice_grid(zeta_local, config)
        values = np.asarray(_evaluate_on_points(logical_field, logical_points)).reshape(
            config.cut_nx,
            config.cut_ny,
            3,
        )
        magnitude = np.linalg.norm(values, axis=-1)
        vr = values[..., 0]
        vtheta = values[..., 1]

        image = ax.pcolormesh(
            np.asarray(r_grid),
            np.asarray(theta_grid),
            magnitude,
            shading="auto",
            cmap="plasma",
        )
        ax.quiver(
            np.asarray(r_grid)[::config.quiver_stride_r, ::config.quiver_stride_t],
            np.asarray(theta_grid)[::config.quiver_stride_r, ::config.quiver_stride_t],
            vr[::config.quiver_stride_r, ::config.quiver_stride_t],
            vtheta[::config.quiver_stride_r, ::config.quiver_stride_t],
            color="white",
            pivot="mid",
            scale=25,
            linewidth=0.4,
        )
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title(
            rf"Logical slice at $\zeta_{{\mathrm{{full}}}}={zeta_cut:.2f}$"
            + "\n"
            + rf"$\zeta_{{\mathrm{{loc}}}}={zeta_local:.2f}$"
        )
        fig.colorbar(image, ax=ax, shrink=0.88, label=r"$|u|$")
    return fig, axes


# %% Stage 5: plot the physical null vector on a poloidal cut
def plot_poloidal_cross_section(config: Config = CONFIG):
    full_map = FULL_ANALYTIC_MAP
    field = _require_full_null_field()
    fig, axes = plt.subplots(
        1,
        len(config.zeta_cuts),
        figsize=(5 * len(config.zeta_cuts), 4.8),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, zeta_cut in zip(axes, config.zeta_cuts):
        logical_points, _, _, _ = _build_poloidal_slice_grid(zeta_cut, config)
        mapped_points = np.asarray(_evaluate_on_points(full_map, logical_points)).reshape(
            config.cut_nx,
            config.cut_ny,
            3,
        )
        values = np.asarray(_evaluate_on_points(field, logical_points)).reshape(
            config.cut_nx,
            config.cut_ny,
            3,
        )
        X = mapped_points[..., 0]
        Y = mapped_points[..., 1]
        Z = mapped_points[..., 2]
        R = np.sqrt(X**2 + Y**2)
        magnitude = np.linalg.norm(values, axis=-1)

        radial_denom = np.where(R > 0.0, R, 1.0)
        vR = (X * values[..., 0] + Y * values[..., 1]) / radial_denom
        vZ = values[..., 2]

        image = ax.pcolormesh(
            R,
            Z,
            magnitude,
            shading="auto",
            cmap="plasma",
        )
        ax.quiver(
            R[::config.quiver_stride_r, ::config.quiver_stride_t],
            Z[::config.quiver_stride_r, ::config.quiver_stride_t],
            vR[::config.quiver_stride_r, ::config.quiver_stride_t],
            vZ[::config.quiver_stride_r, ::config.quiver_stride_t],
            color="white",
            pivot="mid",
            scale=25,
            linewidth=0.4,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Z$")
        ax.set_title(rf"Physical slice at full $\zeta={zeta_cut:.2f}$")
        fig.colorbar(image, ax=ax, shrink=0.88, label=r"$|u|$")
    return fig, axes


# %% Stage 6: plot the outer surface colored by |null_vector|
def plot_surface_magnitude(config: Config = CONFIG):
    field = _require_full_null_field()
    surface_grid = get_2d_grids(
        FULL_ANALYTIC_MAP,
        cut_axis=0,
        cut_value=1.0,
        nx=1,
        ny=config.surface_ntheta,
        nz=config.surface_nzeta,
        invert_z=True,
    )
    values = _evaluate_field_on_grid(field, surface_grid)
    magnitude = np.linalg.norm(np.asarray(values), axis=-1)
    colors, vmin, vmax = _surface_colors(magnitude)

    X = np.asarray(surface_grid[2][0])
    Y = np.asarray(surface_grid[2][1])
    Z = np.asarray(surface_grid[2][2])
    U = np.asarray(values[..., 0])
    V = np.asarray(values[..., 1])
    W = np.asarray(values[..., 2])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    normals = _surface_normals(X, Y, Z, config.rotating_r0)
    view_direction = _surface_view_direction(ax.elev, ax.azim)

    quiver_step = (
        slice(None, None, config.surface_quiver_stride_theta),
        slice(None, None, config.surface_quiver_stride_zeta),
    )
    Xq = X[quiver_step]
    Yq = Y[quiver_step]
    Zq = Z[quiver_step]
    Uq = U[quiver_step]
    Vq = V[quiver_step]
    Wq = W[quiver_step]
    Nq = normals[quiver_step]
    visible_mask = np.sum(Nq * view_direction[None, None, :], axis=-1) > 0.0
    Xq = Xq[visible_mask]
    Yq = Yq[visible_mask]
    Zq = Zq[visible_mask]
    Uq = Uq[visible_mask]
    Vq = Vq[visible_mask]
    Wq = Wq[visible_mask]
    Nq = Nq[visible_mask]
    vector_norm = np.sqrt(Uq**2 + Vq**2 + Wq**2)
    vector_norm = np.where(vector_norm > 0.0, vector_norm, 1.0)

    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        shade=False,
        linewidth=0,
        antialiased=False,
    )

    ax.quiver(
        Xq + config.surface_quiver_offset * Nq[:, 0],
        Yq + config.surface_quiver_offset * Nq[:, 1],
        Zq + config.surface_quiver_offset * Nq[:, 2],
        Uq / vector_norm,
        Vq / vector_norm,
        Wq / vector_norm,
        length=config.surface_quiver_length,
        normalize=False,
        color="black",
        linewidth=0.8,
        alpha=0.9,
    )

    set_axes_equal(ax)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title(r"Full-torus outer surface colored by $|u|$")

    sm = plt.cm.ScalarMappable(cmap="plasma")
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.08, label=r"$|u|$")
    plt.tight_layout()
    return fig, ax


# %% Build sequence with the analytical map
build_case() # 1 min on H100

# %% Assemble operators
assemble_case() # 1.5 min on H100

# %% Compute the free Hcurl nullspace
compute_hcurl_nullspace() # 1 min on H100

# %% Plot the logical null vector on a poloidal cross-section
plot_logical_cross_section()

# %% Plot the physical null vector on a poloidal cross-section
plot_poloidal_cross_section()

# %% Plot the outer surface colored by |null_vector|
plot_surface_magnitude()
# %%
