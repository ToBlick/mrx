"""Equilibria computed by other codes (GVEC, VMEC, DESC) as input to MRX.

:func:`read_equilibrium` reads a GVEC ``.dat`` (:mod:`mrx.equilibria.gvec`), a VMEC ``.nc``
(:mod:`mrx.equilibria.vmec`) or a DESC ``.h5`` (:mod:`mrx.equilibria.desc`) file and returns the same
dict for every code, called the *state*. A VMEC input namelist ``input.<name>`` (:mod:`mrx.equilibria.indata`) gives
a state from its boundary alone: a map filled between a guessed axis and the boundary, no field, and the namelist's
profiles. :func:`build_map` turns a state into the spline map ``Phi`` of an
MRX de Rham sequence. :class:`StateField` evaluates one field of the state, for example lambda, as a JAX
function of the logical point.

The state holds the number of field periods ``nfp``, the reader ``kind`` (``"gvec"``, ``"vmec"`` or
``"desc"``), the file ``path``, the radial profiles and three fields, ``X1 = R``, ``X2 = Z`` (in the planes
``phi = const``) and ``LA = lambda`` (lambda in radians). Each field is a Fourier series
``sum_mn c_mn(r) cos(m theta - n zeta) + s_mn(r) sin(m theta - n zeta)`` with the angles in radians,
``zeta`` the toroidal angle over the full torus and ``n`` a multiple of ``nfp``. The dict of a field holds
the mode numbers ``m`` and ``n`` and the radial functions ``c_mn``, ``s_mn`` as B-splines: the coefficient
arrays ``cos`` and ``sin`` of shape ``(n_modes, n_base)``, the degree ``deg`` and the clamped knot vector
``T``. The radial label ``r`` is the square root of the normalised toroidal flux. ``profiles`` holds
``phi`` (the toroidal flux divided by ``2 pi``), ``iota`` (per full turn) and ``pressure`` as scipy
``BSpline``\\ s in ``r``. A GVEC state in GVEC's G-frame (``hmap = 21``) has ``X1``, ``X2`` the coordinates
in the plane of an axis-following frame instead of ``R``, ``Z``, and the frame itself in ``frame``
(:func:`mrx.equilibria.gvec.read_frame`). A stellarator-symmetric state (:func:`is_stellarator_symmetric`) has
``X1`` as a cosine series and ``X2``, lambda as sine series.

Every map is a :class:`FrameMap`, whose frame is a periodic spline in ``zeta`` on the knots of the mesh
(:func:`axis_basis`). :func:`build_map` gives it in the frame the file supplies: GVEC's G-frame, projected in L2 onto
those splines (:func:`map_frame`), for a state that has one, the cylindrical frame (:func:`cylindrical_frame`) for every other file (VMEC, DESC, a cylindrical
GVEC state), whose planes are the planes ``phi = const``, with the third coordinate ``X3``, out of the frame's plane,
zero. :func:`mrx.geometry.build_sequence` by default re-sections the file's map into the planes normal to its axis
(:func:`mrx.relaxation.mesh_motion.axis_normal_map`), the frame of that axis, again with ``X3 = 0``. ``X3`` is there
for a moving mesh and for a shape optimisation.

The logical coordinates ``(r, theta, zeta)`` of a state are right-handed: the map
``(R cos phi, R sin phi, Z)`` has a positive Jacobian. A file whose angles are left-handed, as in a VMEC
wout, is read with the poloidal angle reversed, ``theta -> -theta``, as DESC does. Lambda and iota then
change sign, and ``theta_reversed`` in the state says so.
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline

from mrx.differential_forms import det33
from mrx.equilibria.desc import read_desc
from mrx.equilibria.gvec import read_state
from mrx.equilibria.indata import indata_state, is_indata
from mrx.equilibria.vmec import read_wout
from mrx.extraction_operators import gram_inverse
from mrx.paths import data_file
from mrx.precision import DTYPE
from mrx.quadrature import composite_quad
from mrx.symmetry import stellarator_symmetric_scalar, stellarator_symmetry_defect
from mrx.spline_bases import SplineBasis, basis_derivative_table, knot_vector

TWO_PI = 2.0 * np.pi

#: The readers, by file extension.
READERS = {".dat": ("gvec", read_state), ".nc": ("vmec", read_wout), ".h5": ("desc", read_desc)}


def read_equilibrium(path):
    """Read the equilibrium file ``path`` and return its state. The file extension selects the reader
    (:data:`READERS`), a VMEC input file is named ``input.<name>``. A file that is not at ``path`` is looked up by name in ``$MRX_DATA/data``
    (:func:`mrx.paths.data_file`), and the state's ``path`` is the file that was read."""
    path = data_file(path)
    ext = os.path.splitext(path)[1]
    if is_indata(path):
        kind, read = "indata", indata_state
    elif ext in READERS:
        kind, read = READERS[ext]
    else:
        raise ValueError(f"{path}: not an equilibrium file, MRX reads GVEC state files (.dat), VMEC wout files (.nc), "
                         "DESC output files (.h5) and VMEC input files (input.<name>)")
    st = dict(read(path), kind=kind, path=path)
    reverse = _left_handed(st)
    return dict(_reverse_theta(st) if reverse else st, theta_reversed=reverse)


def _left_handed(st):
    """Whether the angles of ``st`` are left-handed, that is whether the state's map
    (:func:`state_position`) has a negative Jacobian. It is checked at mid-radius, where it must have one
    sign."""
    position = state_position(st)
    t, z = np.meshgrid(np.arange(16) / 16, np.arange(8) / 8, indexing="ij")
    x = jnp.stack([jnp.full(t.size, 0.5), jnp.asarray(t.ravel()), jnp.asarray(z.ravel())], axis=1)
    d = np.asarray(jax.vmap(lambda xi: det33(jax.jacfwd(position)(xi)))(x))
    if not ((d > 0).all() or (d < 0).all()):
        raise ValueError(f"{st['path']}: the Jacobian of the map changes sign at r = 0.5")
    return bool(d[0] < 0)


def _negated(f):
    return BSpline(f.t, -f.c, f.k)


def _reverse_theta(st):
    """``st`` in the poloidal angle ``-theta``. ``R`` and ``Z`` are the same functions of the new angle, lambda
    is ``-lambda``, since ``theta + lambda`` is the straight-field-line angle, and iota and chi change sign."""
    st = dict(st)
    # cos(m theta - n zeta) = cos(m (-theta) + n zeta) and sin(m theta - n zeta) = -sin(m (-theta) + n zeta)
    for key, cos_sign, sin_sign in (("X1", 1.0, -1.0), ("X2", 1.0, -1.0), ("LA", -1.0, 1.0)):
        block = dict(st[key])
        block["n"], block["cos"], block["sin"] = -block["n"], cos_sign * block["cos"], sin_sign * block["sin"]
        st[key] = block
    st["profiles"] = {name: _negated(f) if name in ("iota", "chi") else f for name, f in st["profiles"].items()}
    return st


def is_stellarator_symmetric(st):
    """Whether the state ``st`` is stellarator-symmetric, that is ``R`` is even and ``Z``, lambda are odd
    under ``(theta, zeta) -> (-theta, -zeta)``. This holds when ``R`` has no sine and ``Z``, lambda have
    no cosine coefficients. A G-frame (``X1``, ``X2`` the frame coordinates) must also map the reflection to
    ``(x, y, z) -> (x, -y, -z)``: the axis and ``N`` have an even ``x`` and odd ``y``, ``z`` component, ``B``
    the opposite. Coefficients of the wrong parity count as zero below 1e-10 of those of the right one, so a
    file written without the symmetry but symmetric to round-off (a frame read from samples, for example) is
    symmetric."""
    def negligible(wrong, right):
        return np.abs(wrong).max(initial=0.0) <= 1e-10 * np.abs(right).max(initial=0.0)

    symmetric = all(negligible(st[b][w], st[b][r]) for b, w, r in (("X1", "sin", "cos"), ("X2", "cos", "sin"),
                                                                 ("LA", "cos", "sin")))
    if "frame" not in st:
        return symmetric
    cos, sin = st["frame"]["cos"], st["frame"]["sin"]
    odd = np.array([[False, True, True], [False, True, True], [True, False, False]])[:, :, None]
    return symmetric and negligible(np.where(odd, cos, sin), cos)


class StateField:
    """One field of the state (``block``, for example ``st["LA"]``) as a JAX function of the logical point
    ``(r, theta, zeta)``. Here the angles are in turns (``[0, 1)``) and ``zeta`` covers one of the ``nfp``
    field periods, as in the logical domain of the map."""

    def __init__(self, block, nfp):
        self.basis = SplineBasis(block["cos"].shape[1], block["deg"], "clamped", T=jnp.asarray(block["T"]))
        self.C_cos = jnp.asarray(block["cos"])                           # (n_modes, n_base)
        self.C_sin = jnp.asarray(block["sin"])
        self.m = jnp.asarray(block["m"], dtype=float)
        self.n_per = jnp.asarray(block["n"], dtype=float) / nfp          # per field period

    def __call__(self, x):
        # r is not clipped to [0, 1]. A clip halves the autodiff radial derivative at r = 1 (JAX splits
        # the gradient of a tie). The local evaluator continues the end pieces instead.
        vals, idx = self.basis.evaluate_local(x[0])
        arg = 2.0 * jnp.pi * (self.m * x[1] - self.n_per * x[2])
        return jnp.cos(arg) @ (self.C_cos[:, idx] @ vals) + jnp.sin(arg) @ (self.C_sin[:, idx] @ vals)


def _rotate_z(angle, v):
    """The vector ``v`` rotated about the ``z`` axis by ``angle``."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


#: The smallest degree of the splines in ``zeta`` of a map's frame and of the magnetic axis, and their cells per cell of
#: the mesh's ``zeta`` splines (:func:`axis_basis`). Two cells per mesh cell give as many unknowns as the Fourier axis
#: of as many modes as ``zeta`` cells and relax W7-X and HSX as it does; on one cell per mesh cell W7-X reached the
#: force bar 1 to 21 steps later with an erratic innermost transform moment.
AXIS_MIN_DEGREE, AXIS_REFINE = 3, 2


def axis_basis(bz):
    """The periodic splines in ``zeta`` of a map's frame and of the magnetic axis: degree ``max(AXIS_MIN_DEGREE, p)``
    on the knots of the mesh's ``zeta`` splines ``bz``, each cell split into :data:`AXIS_REFINE` equal cells. The
    knots refine the mesh's, so every cell of the curve lies in one cell of the field, the curve converges with the
    mesh and its integrals fall on the field's cells. From degree 3 a curve has a continuous curvature, so the frame of
    the planes normal to it turns continuously and ``dPhi/dzeta`` is continuous across the knots."""
    p = max(AXIS_MIN_DEGREE, bz.p)
    T = np.asarray(bz.T, dtype=np.float64)
    bp = np.unique(T[(T >= 0.0) & (T <= 1.0)])
    fine = np.concatenate([np.linspace(a, b, AXIS_REFINE + 1)[:-1] for a, b in zip(bp[:-1], bp[1:])] + [bp[-1:]])
    return SplineBasis(fine.size - 1, p, "periodic", T=jnp.asarray(knot_vector(fine, p, True)))


def axis_quadrature(basis, points):
    """Gauss-Legendre nodes and weights ``(z, w)``, NumPy float64, with ``points`` points on every cell of the periodic
    ``basis``, over one period."""
    T = np.asarray(basis.T, dtype=np.float64)
    return composite_quad(np.unique(T[(T >= 0.0) & (T <= 1.0)]), points)


def project_curve(basis, z, w, values):
    """The coefficients ``(..., n)`` in the periodic ``basis`` of the L2 projection of the curve whose ``values``
    ``(n_q, ...)`` are given at the quadrature nodes ``z`` with the weights ``w``."""
    N = np.asarray(basis.collocation_matrix(jnp.asarray(z)), dtype=np.float64)       # (n_q, n)
    values = np.asarray(values, dtype=np.float64)
    c = np.linalg.solve((N * w[:, None]).T @ N, (N * w[:, None]).T @ values.reshape(len(z), -1))
    return np.moveaxis(c.reshape((basis.n,) + values.shape[1:]), 0, -1)


def curve_tables(basis, z):
    """The values and the ``zeta`` derivatives ``(len(z), n)`` of the splines of ``basis`` at the points ``z``."""
    z = jnp.asarray(z)
    return (np.asarray(basis.collocation_matrix(z), dtype=np.float64),
            np.asarray(basis_derivative_table(basis, z), dtype=np.float64).T)


def curve_derivatives(basis, coef, zeta, order):
    """The spline curve of the coefficients ``coef`` ``(..., n)`` in ``basis`` at ``zeta`` and its first ``order``
    ``zeta`` derivatives, a tuple of ``order + 1`` arrays of shape ``coef.shape[:-1]``."""
    values, idx = basis.evaluate_local(zeta)
    f = lambda z: basis.evaluate_local(z)[0]                    # noqa: E731
    tables = [values]
    for _ in range(order):
        f = jax.jacfwd(f)
        tables.append(f(zeta))
    C = coef[..., idx]
    return tuple(C @ t for t in tables)


def cylindrical_frame(n):
    """The coefficients ``(3, 3, n)`` (rows ``a``, ``N``, ``B``) of the cylindrical frame in ``n`` periodic splines:
    the axis ``a = 0``, ``N`` the first and ``B`` the third axis of the frame that turns with the field periods,
    constant in ``zeta``. Its planes are the planes ``phi = const``, ``N = e_R``, ``B = e_z``, so ``X1 = R`` and
    ``X2 = Z``. The splines sum to one, so the coefficients are exact."""
    F = np.zeros((3, 3, n))
    F[1, 0] = F[2, 2] = 1.0
    return F


def _state_frame_at(st, nfp):
    """The frame ``(a, N, B)`` ``(3, 3)`` of the state ``st`` at the logical ``zeta`` of ``nfp`` field periods, as a
    JAX function: the file's series of a G-frame (:func:`mrx.equilibria.gvec.read_frame`), or the cylindrical
    frame for a state without one."""
    if "frame" not in st:
        F = jnp.asarray(cylindrical_frame(1)[..., 0])
        return lambda zeta: F
    cos, sin = jnp.asarray(st["frame"]["cos"], dtype=float), jnp.asarray(st["frame"]["sin"], dtype=float)
    k = TWO_PI * jnp.asarray(st["frame"]["n"], dtype=float) / nfp
    return lambda zeta: cos @ jnp.cos(-k * zeta) + sin @ jnp.sin(-k * zeta)


def state_position(st):
    """The state's own map from the logical point ``(r, theta, zeta)`` (angles in turns, ``zeta`` one field
    period) to Cartesian ``(x, y, z)``, ``R_z(phi) (a + X1 N + X2 B)`` with ``phi = 2 pi zeta / nfp`` and the
    state's frame, the file's own series of it, as a JAX function. The spline map of :func:`build_map`
    approximates it."""
    nfp = st["nfp"]
    X1, X2 = StateField(st["X1"], nfp), StateField(st["X2"], nfp)
    frame = _state_frame_at(st, nfp)

    def position(x):
        a, N, B = frame(x[2])
        return _rotate_z(TWO_PI / nfp * x[2], a + X1(x) * N + X2(x) * B)

    return position


def map_frame(st, bz, nfp):
    """``(frame, error)``: the frame of the state ``st`` as the coefficients ``(3, 3, n)`` of a map's frame in the
    splines of :func:`axis_basis` of the mesh's ``zeta`` splines ``bz``, for ``nfp`` field periods. The cylindrical
    frame is exact. A G-frame is the L2 projection of the file's series, ``error`` its rms error over a field period,
    of the axis in the map's units and the largest of ``N`` and ``B``."""
    basis = axis_basis(bz)
    if "frame" not in st:
        return cylindrical_frame(basis.n), (0.0, 0.0)
    T = np.asarray(basis.T, dtype=np.float64)
    width = np.diff(np.unique(T[(T >= 0.0) & (T <= 1.0)])).max()
    # enough points for the highest mode's phase across the widest cell
    k = np.abs(np.asarray(st["frame"]["n"], dtype=np.float64)).max() / nfp
    z, w = axis_quadrature(basis, basis.p + 7 + int(np.ceil(TWO_PI * k * width)))
    F = np.asarray(jax.vmap(_state_frame_at(st, nfp))(jnp.asarray(z, dtype=float)), dtype=np.float64)   # (n_q, 3, 3)
    frame = project_curve(basis, z, w, F)
    miss = np.einsum("qn,vcn->qvc", curve_tables(basis, z)[0], frame) - F
    rms = np.sqrt(np.einsum("q,qvc->v", w, miss ** 2))
    return frame, (float(rms[0]), float(rms[1:].max()))


def _angular_coefficients(basis, freqs):
    """The L2 projections of ``exp(2 pi i f theta)`` onto the periodic ``basis``, one row per frequency."""
    freqs = np.asarray(freqs, dtype=np.float64)
    T = np.asarray(basis.T, dtype=np.float64)
    bp = np.unique(T[(T >= 0.0) & (T <= 1.0)])
    # enough points for the highest mode's phase across the widest cell
    pts, w = composite_quad(bp, basis.p + 7 + int(np.ceil(TWO_PI * np.abs(freqs).max() * np.diff(bp).max())))
    B = np.asarray(basis.collocation_matrix(jnp.asarray(pts)), dtype=np.float64)
    moments = B.T @ (w[:, None] * np.exp(1j * TWO_PI * np.outer(pts, freqs)))   # (N, n_modes)
    return np.linalg.solve(B.T @ (w[:, None] * B), moments).T


def _radial_coefficients(block, C, basis_r):
    """The exact L2 projections of the radial splines ``C`` of a block onto ``basis_r``, one column per mode."""
    # Gauss quadrature on the union of both knot vectors integrates the products exactly
    T_s, deg_s = np.asarray(block["T"]), block["deg"]
    T_r, p_r = np.asarray(basis_r.T, dtype=np.float64), basis_r.p
    pts, w = composite_quad(np.unique(np.concatenate([T_s, T_r])), (deg_s + p_r) // 2 + 1)
    Br = BSpline.design_matrix(pts, T_r, p_r).toarray()
    Bs = BSpline.design_matrix(pts, T_s, deg_s).toarray()
    return np.linalg.solve(Br.T @ (w[:, None] * Br), Br.T @ (w[:, None] * (Bs @ C.T)))


class FrameMap(eqx.Module):
    """The map ``Phi = R_z(phi) (a + X1 N + X2 B + X3 T)``, ``phi = 2 pi zeta / nfp``, of the scalar splines ``X1``,
    ``X2`` and ``X3`` with the tensor-product coefficients ``raw_X1``, ``raw_X2``, ``raw_X3`` ``(n_r, n_theta,
    n_zeta)`` in ``basis``.

    The frame is the axis ``a`` and two orthonormal vectors ``N``, ``B`` that span the plane of the cross-section
    at each ``zeta``, in the frame that turns with the field periods, and ``T = N x B`` the normal of that plane. Its
    three vectors are periodic splines in ``zeta`` on the knots of the mesh, of degree ``max(3, p)`` with two cells per mesh cell
    (:func:`axis_basis`), with the coefficients ``frame`` ``(3, 3, n_zeta)`` (rows ``a``, ``N``, ``B``, Cartesian
    component, spline). GVEC's G-frame is read into it by an L2 projection (:func:`map_frame`). The cylindrical
    frame (:func:`cylindrical_frame`) has ``a = 0``, ``N = e_R``, ``B = e_z``, and then ``X1 = R``, ``X2 = Z``.

    Every map read from a file has ``X3 = 0``: its surface ``zeta = const`` is exactly the plane of the frame and
    ``(X1, X2)`` are coordinates in it. ``X3`` lets a point leave its plane, so that a moving mesh or a shape
    optimisation can move every point in any direction.

    It is a pytree: under ``jax.jit`` the coefficients of ``X1``, ``X2``, ``X3`` and of the frame are traced inputs,
    so a new map of the same mesh reuses every compiled function of the
    sequence it is installed on, and every one of them can be differentiated."""

    raw_X1: jnp.ndarray
    raw_X2: jnp.ndarray
    raw_X3: jnp.ndarray
    frame: jnp.ndarray
    basis: object = eqx.field(static=True)
    nfp: int = eqx.field(static=True)
    #: The names of the coordinates of :meth:`section`.
    section_labels: tuple = eqx.field(static=True, default=("$X^1$", "$X^2$"))

    def __call__(self, x):
        a, N, B = self.frame_derivatives(x[2], 0)[0]
        X1, X2, X3 = (self.basis.contract(c, x) for c in (self.raw_X1, self.raw_X2, self.raw_X3))
        return _rotate_z(TWO_PI / self.nfp * x[2], a + X1 * N + X2 * B + X3 * jnp.cross(N, B))

    def section(self, x):
        """The coordinates ``(X1, X2)`` of the point ``x`` along ``N`` and ``B``, its coordinates in the plane of its
        cross-section ``zeta = const`` where ``X3 = 0``."""
        return jnp.array([self.basis.contract(self.raw_X1, x), self.basis.contract(self.raw_X2, x)])

    def plane(self, zeta):
        """``(o, e1, e2)``: the plane of the frame at the logical ``zeta`` in Cartesian coordinates, by its origin, the
        axis ``a``, and the directions ``N`` and ``B`` of the coordinates ``(X1, X2)`` of :meth:`section`."""
        ang = TWO_PI / self.nfp * zeta
        return tuple(_rotate_z(ang, v) for v in self.frame_derivatives(zeta, 0)[0])

    def frame_derivatives(self, zeta, order=2):
        """The frame ``(a, N, B)`` at the logical ``zeta`` and its first ``order`` ``zeta`` derivatives, ``order + 1``
        arrays ``(3, 3)`` (rows ``a``, ``N``, ``B``), in the frame that turns with the field periods."""
        return curve_derivatives(axis_basis(self.basis.bases[2]), self.frame, zeta, order)

    def with_coordinates(self, raw_X1, raw_X2, raw_X3):
        """The map with the same frame and the coefficients ``raw_X1``, ``raw_X2``, ``raw_X3``."""
        return eqx.tree_at(lambda m: (m.raw_X1, m.raw_X2, m.raw_X3), self, (raw_X1, raw_X2, raw_X3))

    def with_frame(self, frame):
        """The map with the same coefficients of ``X1``, ``X2``, ``X3`` in the frame of the coefficients ``frame``.
        Coordinates belong to their frame, so a map whose coordinates come from another map takes that map's frame
        with them: ``Phi.with_coordinates(*X).with_frame(...)``."""
        return eqx.tree_at(lambda m: m.frame, self, frame)

    def scaled(self, factor):
        """The map ``factor Phi``: ``X1``, ``X2``, ``X3`` and the axis ``a`` times ``factor``, ``N`` and ``B`` kept."""
        return eqx.tree_at(lambda m: (m.raw_X1, m.raw_X2, m.raw_X3, m.frame), self,
                           (factor * self.raw_X1, factor * self.raw_X2, factor * self.raw_X3,
                            self.frame.at[0].multiply(factor)))


def polar_projection(seq):
    """The projection of the tensor coefficients ``(n_r, n_theta, n_zeta)`` of a scalar spline of ``seq.basis_0`` onto
    the polar 0-forms of ``seq`` that may be nonzero on the wall, the space of the map's coordinates, as a function
    of a float64 array. It is ``E^T (E E^T)^-1 E`` with ``E`` the extraction of those 0-forms: the coefficients of
    the radial rings at the axis become those of a function that is smooth (C1) there, all others stay."""
    free = seq.free                   # the map's coefficients are not zero on the wall
    # The projection v = (E E^T)^-1 E c (DeRhamSequence.conforming) and E^T v run on the host in float64 and are
    # rounded once, so every build has the same map bit for bit, in both precisions. On a GPU the sparse applies are
    # a scatter, which adds in an order the device chooses, and near the axis the metric turns a few units in the
    # last place of R into 1e-3 of the inverse metric in float32.
    e = free.E(0)
    rows, cols, vals = e.entries()
    E = sp.csr_matrix((vals, (rows, cols)), shape=e.forward_shape)
    gram_inv = gram_inverse(e, free.core_rows(0), free._gram[(0, False)])
    shape = seq.basis_0.shape[0]

    def project(C):
        return (E.T @ (gram_inv @ (E @ np.asarray(C, dtype=np.float64).reshape(-1)))).reshape(shape)
    return project


def build_map(st, seq, nfp=None, stellarator_symmetric=False):
    """Return ``(Phi, info)``, the spline map of the state ``st`` in the 0-form space of ``seq``, a
    :class:`FrameMap` with the state's frame (:func:`map_frame`).

    The map is ``Phi = R_z(phi) (a + X1 N + X2 B)`` with ``phi = 2 pi zeta / nfp`` (``X3 = 0``), so logical ``zeta``
    covers one field period. ``nfp`` is the file's unless given. ``X1`` and ``X2`` are the L2 projections of the file's
    Fourier series onto the 0-form spline space, which is smooth (C1) at the axis. A G-frame is projected in L2 onto
    the frame's splines (:func:`map_frame`). With ``stellarator_symmetric`` ``X1`` and ``X2`` are further projected onto ``X1`` even and ``X2`` odd
    under ``(theta, zeta) -> (-theta, -zeta)``. This needs uniform angular knots and a stellarator-symmetric state,
    whose frame has that parity already.

    ``info`` holds ``nfp``, the range ``det_range`` of ``det DPhi`` at the sample points, the
    ``symmetry_defect`` (:func:`mrx.symmetry.stellarator_symmetry_defect`) and the ``frame_error`` of
    :func:`map_frame`. Raises if ``det DPhi`` is not
    positive, which :func:`read_equilibrium` guarantees for a map that does not fold.
    """
    if stellarator_symmetric and not is_stellarator_symmetric(st):
        raise ValueError("the state is not stellarator-symmetric: the projection would drop its asymmetric part")
    nfp = st["nfp"] if nfp is None else int(nfp)
    br, bt, bz = seq.basis_0.Lambda
    polar = polar_projection(seq)

    def raw(block):
        # the map's nfp, not the file's: with an override logical zeta spans nfp_file / nfp periods
        n_per = block["n"] / nfp
        if np.abs(n_per - np.round(n_per)).max() > 0:
            raise ValueError("toroidal mode numbers are not multiples of nfp")
        # the tensor coefficients: sum over modes of radial x angular coefficients
        modes = (_angular_coefficients(bt, block["m"].astype(np.float64))[:, :, None]
                 * np.conj(_angular_coefficients(bz, n_per))[:, None, :])   # exp(2 pi i (m theta - n zeta))
        C = (np.einsum("ik,kjl->ijl", _radial_coefficients(block, block["cos"], br), modes.real)
             + np.einsum("ik,kjl->ijl", _radial_coefficients(block, block["sin"], br), modes.imag))
        return jnp.asarray(polar(C), dtype=DTYPE)

    raw_X1, raw_X2 = raw(st["X1"]), raw(st["X2"])
    if stellarator_symmetric:
        raw_X1 = stellarator_symmetric_scalar(raw_X1, seq.basis_0, even=True)
        raw_X2 = stellarator_symmetric_scalar(raw_X2, seq.basis_0, even=False)
    labels = ("$X^1$", "$X^2$") if "frame" in st else ("R", "Z")
    frame, frame_error = map_frame(st, bz, nfp)
    Phi = FrameMap(raw_X1, raw_X2, jnp.zeros_like(raw_X1), jnp.asarray(frame, dtype=float), seq.basis_0.bases[0],
                   nfp, labels)
    # sample points away from the axis, where det DPhi = 0
    rng = np.random.default_rng(0)
    pts = jnp.asarray(np.column_stack([rng.uniform(0.15, 1.0, 64), rng.uniform(0.0, 1.0, 64),
                                       rng.uniform(0.0, 1.0, 64)]))
    d = np.asarray(jax.vmap(lambda x: det33(jax.jacfwd(Phi)(x)))(pts))
    if not (np.isfinite(d).all() and d.min() > 0):
        raise RuntimeError(f"{st['path']}: det DPhi of the spline map in [{d.min():.3e}, {d.max():.3e}] is not "
                           "positive (a map that folds, or a resolution too coarse for the file)")
    return Phi, {"nfp": nfp, "det_range": (float(d.min()), float(d.max())),
                 "symmetry_defect": float(stellarator_symmetry_defect(Phi, pts)), "frame_error": frame_error}
