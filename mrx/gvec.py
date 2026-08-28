"""GVEC equilibria: state files read in closed form, and the flat-schema exports.

The production route is the **state file** (``GVEC_State_*.dat``), GVEC's
own representation of an equilibrium:
    the radial B-spline basis (degree ``deg`` on the element grid ``sp``),
    the Fourier mode table ``(m, n)`` with ``n`` already multiplied by ``nfp``
    and, per mode, the radial coefficients of ``X1 = R`` (cosine series),
    ``X2 = Z`` and ``LA = lambda`` (sine series),
    followed by the profiles ``Phi``, ``chi``, ``iota``, ``p`` at the
    radial interpolation points of the ``X1`` basis.
Its angles are GVEC's ``theta`` and ``zeta`` in radians with the series
``sum f_mn(s) trig(m theta - n zeta)``; ``s`` is the radial label the
flat-schema exports call ``rho`` (``Phi = Phi_edge s^2``). :class:`StateField`
evaluates one of the three fields at a logical point in JAX -- the radial
basis through :class:`mrx.spline_bases.SplineBasis` on GVEC's own knots, the
angles as ``2 pi (m theta - n zeta / nfp)`` -- and :func:`build_gvec_map`
builds the map's polar spline coefficients of ``R`` and ``Z`` from the
series coefficients mode by mode (:func:`series_spline_dofs`: the radial
splines projected onto the map's radial basis, the angular modes in closed
form), while :func:`load_clebsch` histopolates ``lambda`` at the quadrature
points from the closed form. Nothing is evaluated on a grid
(``docs/research/analytic_map_2026-08-28.md``). Validated against the pyGVEC
export of W7-X FMM002 to round-off (``test/test_gvec.py``).

The **flat-schema export** (``quasr_*.h5``, ``w7x_*_mrx.h5``) is the
fallback for equilibria that have no state file: flat ``R``/``Z`` of length
``n_rho*n_theta*n_zeta``, ``eval_points`` of shape ``(N, 3)`` in
``(rho, theta, zeta)`` normalised to ``[0, 1]``, ``nfp``/``n_rho``/
``n_theta``/``n_zeta`` in the attributes, and optionally
``clebsch/{dPhi_dr, dchi_dr, LA}`` and ``pressure``. The grid is bridged to
the Greville points by a linear RegularGridInterpolator, whose ~3.4% bias
does not converge; every W7-X number obtained through it carries a force
floor the closed form does not (``docs/research/coarse_gvec_export_2026-08-26.md``).

Two traps that the flat schema carries:

* **Handedness.** ``mrx.mappings.stellarator_map`` uses
  ``Y = -R sin(2 pi zeta/nfp)``, which mirrors raw GVEC data
  (``det DF < 0``). :func:`build_gvec_map` measures the sign instead of
  assuming it.
* **Open versus closed periodic axes.** The quasr files sample the angles on
  ``[0, 1)`` and need a wrap point; other exports sample ``[0, 1]`` closed
  and must not be padded. Both are detected from the spacing.
* **A wrong ``nfp`` attribute.** nfp enters the map as
  ``F = (R cos(2 pi zeta/nfp), +-R sin(2 pi zeta/nfp), Z)``, so a wrong value
  wraps one field period through the wrong angle with a healthy Jacobian to
  hide it; every reader takes an ``nfp`` override.

Every function takes the file path; the extension decides the route --
``.dat`` the GVEC state, ``.nc`` a VMEC wout refit into the same blocks by
``mrx.vmec``, anything else the flat schema.
``test/synthetic_gvec.py`` writes the flat schema for an
analytic circular torus; the test suite reads that file through the same
functions as a real export.
"""
from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import BSpline

from mrx.differential_forms import DifferentialForm, DiscreteFunction
from mrx.projectors import _conforming_restriction, _solve_tensor_collocation_axis
from mrx.spline_bases import SplineBasis

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# 1. The state file
# ---------------------------------------------------------------------------

def _numbers(line):
    return [float(v) for v in line.replace(",", " ").split()]


def read_state(path):
    """Parse a state file into a dict: ``nfp``, ``sp``, ``deg``, the three
    field blocks ``X1``, ``X2``, ``LA`` (``m``, ``n``, ``coef`` of shape
    ``(n_modes, n_base)``, ``sin_cos`` 1 = sine, 2 = cosine), ``profiles``
    (``s``, ``phi``, ``chi``, ``iota``, ``pressure`` at the interpolation
    points) and ``a_minor``, ``r_major``, ``volume``."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    heads = [i for i, ln in enumerate(lines) if ln.startswith("##")]
    blocks = []                                     # (header text, data lines)
    for j, i in enumerate(heads):
        end = heads[j + 1] if j + 1 < len(heads) else len(lines)
        blocks.append((lines[i][2:].strip(" #"), [ln for ln in lines[i + 1:end] if ln.strip()]))

    def block(prefix):
        for head, data in blocks:
            if head.startswith(prefix):
                return data
        raise ValueError(f"{path}: no '## {prefix}' block")

    st = {}
    n_elems = int(_numbers(block("grid: nElems")[0])[0])
    st["sp"] = np.array(_numbers(block("grid: sp")[0]))[: n_elems + 1]
    nfp, _, _, _, hmap = _numbers(block("global")[0])
    st["nfp"], st["hmap"] = int(nfp), int(hmap)
    for name in ("X1", "X2", "LA"):
        n_base, deg, _, n_modes, sin_cos, _ = (int(v) for v in _numbers(block(f"{name}_base")[0]))
        rows = np.array([_numbers(ln) for ln in block(f"{name}:")])
        if rows.shape != (n_modes, 2 + n_base):
            raise ValueError(f"{path}: {name} block is {rows.shape}, expected {(n_modes, 2 + n_base)}")
        st[name] = dict(m=rows[:, 0].astype(int), n=rows[:, 1].astype(int),
                        coef=rows[:, 2:], sin_cos=sin_cos, deg=deg)
    st["deg"] = st["X1"]["deg"]
    prof = np.array([_numbers(ln) for ln in block("at X1_base IP point positions")])
    st["profiles"] = dict(zip(("s", "phi", "chi", "iota", "pressure"), prof.T))
    st["a_minor"], st["r_major"], st["volume"] = _numbers(block("a_minor,r_major,volume")[0])
    return st


def knots(sp, deg):
    """Clamped knot vector of the degree-``deg`` B-splines on the element grid."""
    return np.concatenate([np.full(deg, sp[0]), sp, np.full(deg, sp[-1])])


def block_knots(block, sp):
    """The radial knot vector of a field block: its own ``T`` (the wout
    route) or the clamped vector on the element grid ``sp``."""
    return np.asarray(block["T"] if "T" in block else knots(sp, block["deg"]))


def radial_design(sp, deg, s):
    """``(len(s), n_base)`` values of the radial basis at ``s``."""
    return BSpline.design_matrix(np.asarray(s, dtype=np.float64), knots(sp, deg), deg).toarray()


def evaluate(block, sp, s, theta, zeta):
    """A field block on the tensor grid ``s x theta x zeta`` (angles in
    radians, ``zeta`` the physical toroidal angle)."""
    A = radial_design(sp, block["deg"], s) @ block["coef"].T             # (n_s, n_modes)
    arg = (np.outer(block["m"], theta)[:, :, None]
           - np.outer(block["n"], zeta)[:, None, :])                      # (n_modes, n_t, n_z)
    F = np.cos(arg) if block["sin_cos"] == 2 else np.sin(arg)
    return np.einsum("sk,ktz->stz", A, F)


def profile_spline(st, name):
    """The radial spline through a profile's interpolation-point values."""
    prof, sp, deg = st["profiles"], st["sp"], st["deg"]
    c = np.linalg.solve(radial_design(sp, deg, prof["s"]), prof[name])
    return BSpline(knots(sp, deg), c, deg)


class StateField:
    """A state's ``X1``, ``X2`` or ``LA`` as a JAX function of the logical
    point ``(rho, theta, zeta)`` (angles on ``[0, 1)``, ``zeta`` per field
    period). ``vector=True`` returns a ``(1,)`` array, the convention of the
    scalar callables :func:`_map_with_sign` takes (the series map itself,
    the reference the spline map is measured against); otherwise a scalar. A block that carries its
    own knot vector ``T`` (the wout route, ``mrx.vmec``) overrides the
    element grid ``sp``, which may then be ``None``."""

    def __init__(self, block, sp, nfp, vector=False):
        self.basis = SplineBasis(block["coef"].shape[1], block["deg"], "clamped",
                                 T=jnp.asarray(block_knots(block, sp)))
        self.C = jnp.asarray(block["coef"])                              # (n_modes, n_base)
        self.m = jnp.asarray(block["m"], dtype=jnp.float64)
        self.n_per = jnp.asarray(block["n"], dtype=jnp.float64) / nfp    # per field period
        self.cos = block["sin_cos"] == 2
        self.vector = vector

    def __call__(self, x):
        vals, idx = self.basis.evaluate_local(jnp.clip(x[0], 0.0, 1.0))
        radial = self.C[:, idx] @ vals                                    # (n_modes,)
        arg = 2.0 * jnp.pi * (self.m * x[1] - self.n_per * x[2])
        f = (jnp.cos(arg) if self.cos else jnp.sin(arg)) @ radial
        return jnp.array([f]) if self.vector else f


# ---------------------------------------------------------------------------
# 2. Map and Clebsch initial condition
# ---------------------------------------------------------------------------

def _map_with_sign(R_h, Z_h, nfp, sign):
    a = TWO_PI / nfp

    def F(x):
        ang = a * x[2]
        r = R_h(x)[0]
        return jnp.array([r * jnp.cos(ang), sign * r * jnp.sin(ang), Z_h(x)[0]])
    return F


def _det_DF(map_func, n=64, seed=0):
    """Sample det(DF) away from the axis and from the r=1 knot, where a
    spline map has det DF = 0 exactly."""
    rng = np.random.default_rng(seed)
    xs = jnp.asarray(np.column_stack([
        rng.uniform(0.15, 0.95, n), rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n)]))
    dets = jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(map_func)(x)))(xs)
    return np.asarray(dets)


def _periodic_symbol(row, freqs):
    """``sum_l row[l] exp(-2 pi i m l / N)`` for every ``m`` in ``freqs``: the
    eigenvalue of the circulant matrix with first column ``row`` on the
    Fourier mode ``m``. Real for the symmetric rows of a uniform periodic
    B-spline basis (the imaginary part is checked to be round-off)."""
    N = len(row)
    sym = np.exp(-2j * np.pi * np.outer(freqs, np.arange(N)) / N) @ row
    if np.abs(sym.imag).max() > 1e-12 * np.abs(sym).max():
        raise ValueError("periodic collocation/mass row is not symmetric")
    return sym.real


def _angular_symbol(basis, freqs, l2):
    """Per-mode coefficient factor ``gamma(m)`` of the uniform periodic
    basis: the degree-``p`` spline with coefficients ``gamma(m) exp(2 pi i m
    x_j)`` (``x_j`` the Greville points, the centres of the basis functions)
    is the interpolant (``l2=False``) or the L2 projection (``l2=True``) of
    ``exp(2 pi i m theta)``.

    Interpolation: the collocation matrix is circulant, so the interpolant
    of a Fourier mode is the mode's samples over the symbol
    ``sigma(m) = sum_l B_0(x_l) exp(-2 pi i m l / N)``. ``sigma`` is
    ``N``-periodic in ``m``: a mode beyond the Nyquist frequency ``N/2`` is
    interpolated as its alias, with the alias's gain.

    L2 projection: the moments are the B-spline's Fourier transform,
    ``int B_j(theta) exp(2 pi i m theta) dtheta = h sinc(m h)^(p+1)
    exp(2 pi i m x_j)`` with ``h = 1/N`` and ``sinc(x) = sin(pi x)/(pi x)``,
    and the mass matrix is circulant with symbol ``mu(m)``; a mode beyond
    Nyquist is damped by ``sinc^(p+1)`` instead of aliased.
    """
    N, p = basis.n, basis.p
    freqs = np.asarray(freqs, dtype=np.float64)
    if not l2:
        A = np.asarray(basis.collocation_matrix(), dtype=np.float64)
        return 1.0 / _periodic_symbol(A[:, 0], freqs)
    xi, wi = np.polynomial.legendre.leggauss(p + 1)
    pts = ((np.arange(N)[:, None] + 0.5 * (xi[None, :] + 1.0)) / N).ravel()
    w = np.tile(0.5 * wi / N, N)
    B = np.asarray(basis.collocation_matrix(jnp.asarray(pts)), dtype=np.float64)
    M = B.T @ (w[:, None] * B)
    moment = np.sinc(freqs / N) ** (p + 1) / N
    return moment / _periodic_symbol(M[:, 0], freqs)


def _radial_coefficients(block, sp, basis_r, coll_r, l2):
    """``(n_r, n_modes)`` coefficients on the map's clamped radial basis of
    every mode's radial function ``c_mn(rho)`` (a spline on the state's
    knots): its Greville interpolant (``l2=False``) or its L2 projection
    (``l2=True``, Gauss quadrature on the union of the two knot sets, exact
    for the spline product). Either is exact when the map's radial space
    contains the state's."""
    T_s, deg_s, C = block_knots(block, sp), block["deg"], block["coef"]
    T_r, p_r = np.asarray(basis_r.T, dtype=np.float64), basis_r.p
    if not l2:
        x_r = np.asarray(basis_r.greville_points(), dtype=np.float64)
        samples = BSpline.design_matrix(x_r, T_s, deg_s).toarray() @ C.T
        return np.linalg.solve(np.asarray(coll_r, dtype=np.float64), samples)
    bp = np.unique(np.concatenate([T_s, T_r]))
    xi, wi = np.polynomial.legendre.leggauss((deg_s + p_r) // 2 + 1)
    lo, hi = bp[:-1], bp[1:]
    pts = (0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * xi[None, :]).ravel()
    w = (0.5 * (hi - lo)[:, None] * wi[None, :]).ravel()
    Br = BSpline.design_matrix(pts, T_r, p_r).toarray()
    Bs = BSpline.design_matrix(pts, T_s, deg_s).toarray()
    M = Br.T @ (w[:, None] * Br)
    return np.linalg.solve(M, Br.T @ (w[:, None] * (Bs @ C.T)))


def series_spline_dofs(block, sp, nfp, seq, l2=False):
    """The polar 0-form DoFs on ``seq.basis_0`` of a state field, built from
    its coefficients alone: no evaluation grid, no collocation solve.

    The field is ``sum_mn c_mn(rho) trig(2 pi (m theta - n zeta / nfp))``
    and both the Greville interpolation and the L2 projection onto the
    tensor-product spline space are linear and tensor-product, so the
    coefficients are the sum over modes of (radial coefficients of
    ``c_mn``) x (angular coefficients of the trig mode), the latter in
    closed form (:func:`_angular_symbol`):

        C[i, j, k] = sum_mn c_mn[i] gamma_t(m) gamma_z(n) trig(2 pi (m x_j - n y_k))

    with ``x_j``, ``y_k`` the angular Greville points. ``l2=False`` is the
    Greville interpolant, identical to sampling the series at the Greville
    points and solving (``seq.interpolate(f, 0)``) to round-off; ``l2=True``
    the L2 projection. The tensor coefficients are then restricted onto the
    polar space with the ring-0/ring-1 surgery of every 0-form
    interpolation (:func:`mrx.projectors._conforming_restriction`).
    """
    br, bt, bz = seq.basis_0.Λ
    m, n_per = block["m"].astype(np.float64), block["n"] / nfp
    if np.abs(n_per - np.round(n_per)).max() > 0:
        raise ValueError("toroidal mode numbers are not multiples of nfp")
    c_r = _radial_coefficients(block, sp, br, seq.greville[0].coll, l2)   # (n_r, n_modes)
    gamma = _angular_symbol(bt, m, l2) * _angular_symbol(bz, n_per, l2)   # (n_modes,)
    x_t = np.asarray(bt.greville_points(), dtype=np.float64)
    y_z = np.asarray(bz.greville_points(), dtype=np.float64)
    arg = TWO_PI * (m[:, None, None] * x_t[None, :, None]
                    - n_per[:, None, None] * y_z[None, None, :])          # (n_modes, n_t, n_z)
    trig = np.cos(arg) if block["sin_cos"] == 2 else np.sin(arg)
    C_full = np.einsum("ik,k,kjl->ijl", c_r, gamma, trig)
    return _conforming_restriction(seq.E(0), jnp.asarray(C_full.reshape(-1)))


def build_gvec_map(h5_path, seq, sign=None, stride=1, nfp=None, l2=False):
    """Build the stellarator map of a GVEC state, a VMEC wout or a
    flat-schema export as a C1 polar spline map on ``seq.basis_0``.

    A ``.dat`` state or a ``.nc`` wout (refit into the same blocks by
    :mod:`mrx.vmec`) supplies ``R`` and ``Z`` as radial-spline x Fourier
    series, and the map's spline coefficients are built from the series
    coefficients directly (:func:`series_spline_dofs`) -- nothing is
    evaluated on a grid. A ``.h5`` export supplies ``R`` and ``Z`` on its
    grid, bridged to the Greville points by linear interpolation
    (``_rgi_fn``) and collocated (``seq.interpolate``).
    Returns ``(F, info)``. ``sign`` is the toroidal handedness
    ``Y = sign * R sin(2 pi zeta/nfp)``; left ``None`` it is measured, and a
    file that is degenerate under both signs raises.
    """
    if h5_path.endswith((".dat", ".nc")):
        if h5_path.endswith(".nc"):
            from mrx.vmec import read_wout  # noqa: PLC0415  (imports this module)
            st = read_wout(h5_path)
        else:
            st = read_state(h5_path)
        nfp = st["nfp"] if nfp is None else int(nfp)
        R_fn = StateField(st["X1"], st.get("sp"), st["nfp"], vector=True)
        Z_fn = StateField(st["X2"], st.get("sp"), st["nfp"], vector=True)
        R_dof = series_spline_dofs(st["X1"], st.get("sp"), st["nfp"], seq, l2)
        Z_dof = series_spline_dofs(st["X2"], st.get("sp"), st["nfp"], seq, l2)
        axes, layout, grid = None, None, "closed form"
    else:
        axes, R_grid, Z_grid, nfp, layout = load_gvec_grids(
            h5_path, stride=stride, nfp=nfp)
        R_fn, Z_fn, grid = _rgi_fn(axes, R_grid), _rgi_fn(axes, Z_grid), R_grid.shape
        R_dof, Z_dof = seq.interpolate(R_fn, 0), seq.interpolate(Z_fn, 0)
    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.E(0))
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.E(0))

    tried = {}
    for s in ((sign,) if sign is not None else (1.0, -1.0)):
        F = _map_with_sign(R_h, Z_h, nfp, s)
        d = _det_DF(F)
        tried[s] = (float(d.min()), float(d.max()))
        if np.isfinite(d).all() and d.min() > 0:
            return F, {"R_h": R_h, "Z_h": Z_h, "R_fn": R_fn, "Z_fn": Z_fn,
                       "axes": axes, "nfp": nfp,
                       "sign": s, "layout": layout, "det_range": tried[s],
                       "grid": grid, "stride": stride}
    raise RuntimeError(f"{h5_path}: no handedness gives det DF > 0; "
                       f"sampled ranges {tried}")


def load_state_clebsch(path, n_rho=401):
    """The ``load_clebsch`` dict of a state file: profiles on ``n_rho``
    uniform radii from the profile splines (``chi' = iota Phi'``) and
    ``lam_h`` the closed-form :class:`StateField` of ``LA``."""
    st = read_state(path)
    rho = np.linspace(0.0, 1.0, n_rho)
    dPhi = profile_spline(st, "phi").derivative()(rho)
    return dict(nfp=st["nfp"], rho=rho, dPhi=dPhi,
                dchi=profile_spline(st, "iota")(rho) * dPhi,
                p=profile_spline(st, "pressure")(rho), iota_spread=0.0,
                lam_h=StateField(st["LA"], st["sp"], st["nfp"]), closed_axes=[])


def load_clebsch(path, types=("clamped", "periodic", "periodic")):
    """Read the radial profiles, a lambda callable and p(rho) from a file.

    The reference 2-form components of ``mrx.initial_conditions`` are exactly
    GVEC's ``sqrt(g) B^i``, verified against the file's own B:
    ``sqrt(g) B^theta = dchi_dr - dPhi_dr dLA_dz`` and
    ``sqrt(g) B^zeta = dPhi_dr (1 + dLA_dt)``, in GVEC's units (derivatives
    with respect to radian angles). The caller converts with
    ``Phi' = 2 pi dPhi_dr``, ``iota = dchi_dr / (nfp dPhi_dr)`` and
    ``lambda = LA / 2 pi``.

    lambda is fitted as the scalar and differentiated, never read as two
    derivatives: ``div B = 0`` rests on the mixed partials cancelling, which
    holds only when both come from one interpolant. The duplicate endpoint of
    a closed periodic sample is dropped before the fit; whether an axis is
    closed is decided from its coordinates (the last point is 1, not
    ``1 - step``), the same rule the map reader applies, and the decision is
    returned as ``closed_axes``. (It used to be decided from ``LA`` itself,
    which mistakes any lambda without angular variation -- in particular
    ``LA = 0`` -- for a closed sample.)

    Returns a dict with ``nfp``, ``rho``, ``dPhi``, ``dchi``, ``p`` (surface
    means, arrays on ``rho``), ``iota_spread`` (max angular departure of
    dchi/dPhi from a flux function at mid-radius) and ``lam_h``. A GVEC
    state file (``.dat``) returns the same dict with ``lam_h`` in closed
    form (:func:`load_state_clebsch`); a VMEC wout (``.nc``) likewise
    through :func:`mrx.vmec.load_wout_clebsch`.
    """
    if path.endswith(".dat"):
        return load_state_clebsch(path)
    if path.endswith(".nc"):
        from mrx.vmec import load_wout_clebsch  # noqa: PLC0415  (imports this module)
        return load_wout_clebsch(path)
    with h5py.File(path, "r") as h:
        shape = (int(h.attrs["n_rho"]), int(h.attrs["n_theta"]),
                 int(h.attrs["n_zeta"]))
        c = h["clebsch"]
        dPhi = np.asarray(c["dPhi_dr"]).reshape(shape)
        dchi = np.asarray(c["dchi_dr"]).reshape(shape)
        LA = np.asarray(c["LA"]).reshape(shape)
        pres = np.asarray(h["pressure"]).reshape(shape)
        ep = np.asarray(h["eval_points"])
        nfp = int(h.attrs["nfp"])

    axes = [np.unique(ep[:, i]) for i in range(3)]
    if not all(len(a) == n for a, n in zip(axes, shape)):
        raise RuntimeError(f"eval_points axes {[len(a) for a in axes]} do not "
                           f"match declared shape {shape}")

    nr = shape[0]
    prof_dPhi = dPhi.mean(axis=(1, 2))
    prof_dchi = dchi.mean(axis=(1, 2))
    prof_p = pres.mean(axis=(1, 2))
    spread = float(np.nanmax(
        np.abs(dchi / dPhi - (prof_dchi / prof_dPhi)[:, None, None])
        [nr // 4:3 * nr // 4]))

    fit_axes, LA_fit, closed = list(axes), LA, []
    for a, kind in enumerate(types):
        if kind == 'periodic' and _axis_is_closed(axes[a]):
            fit_axes[a] = axes[a][:-1]
            LA_fit = np.take(LA_fit, np.arange(len(fit_axes[a])), axis=a)
            closed.append(a)
    lam_h = fit_scalar_spline(fit_axes, LA_fit, types)

    return dict(nfp=nfp, rho=axes[0], dPhi=prof_dPhi, dchi=prof_dchi,
                p=prof_p, iota_spread=spread, lam_h=lam_h, closed_axes=closed)


# ---------------------------------------------------------------------------
# 3. The flat-schema export (grid fallback)
# ---------------------------------------------------------------------------

def _take(grid, axis, sl):
    idx = [slice(None)] * grid.ndim
    idx[axis] = sl
    return grid[tuple(idx)]


def _axis_is_closed(v):
    """Whether a periodic sample in [0, 1] is closed (its last point is 1,
    duplicating the first) rather than half-open (its last point is
    ``1 - step``). Decided from the coordinates, never from the data."""
    return abs(v[-1] - 1.0) < 0.5 * (v[1] - v[0])


def _periodic_axis(vals, grid, axis, stride=1):
    """Normalise one periodic axis to a half-open [0,1) sample, then wrap-pad.

    quasr samples half-open (0, 1/n, ..., (n-1)/n); hegna samples closed
    (0, ..., 1) with the endpoint duplicating the first point. Normalising to
    half-open first and padding unconditionally makes both paths identical and
    makes ``stride`` safe on the closed layout (79 is prime, so every stride
    would otherwise leave a short final cell).
    """
    v = np.asarray(vals, dtype=np.float64)
    if v.max() > 1.5:                      # radians -> [0,1]
        v = v / (v.max() + (v[1] - v[0]))
    layout = "half-open"
    if _axis_is_closed(v):                 # drop the duplicate endpoint
        v, grid, layout = v[:-1], _take(grid, axis, slice(0, -1)), "closed"
    v, grid = v[::stride], _take(grid, axis, slice(None, None, stride))
    pad = np.concatenate([grid, _take(grid, axis, slice(0, 1))], axis=axis)
    return np.concatenate([v, [1.0]]), pad, layout


def _radial_axis(vals, grid, axis, stride=1):
    """Subsample the clamped radial axis, always keeping the last point.

    Dropping rho=1 would turn every near-boundary evaluation into an
    extrapolation, which is where the natural-BC term lives.
    """
    v = np.asarray(vals, dtype=np.float64)
    keep = np.unique(np.r_[np.arange(0, len(v), stride), len(v) - 1])
    return v[keep], _take(grid, axis, keep)


def load_gvec_grids(h5_path, stride=1, nfp=None):
    """Return ``(axes, R_grid, Z_grid, nfp, layout)`` from a flat-schema file.

    ``nfp`` overrides the file's attribute (see the module docstring).
    ``stride`` subsamples the data grid per axis. Default 1 is the right
    default: the grid is read once and sampled only at the map's Greville
    points (geometry build 88 s at 50^3, 136 s at 80^3, peak RSS ~4 GB), and
    stride 2 roughly quadruples the O(h^2) fit error. The knob exists for
    smoke tests.
    """
    with h5py.File(h5_path, "r") as f:
        ep = np.asarray(f["eval_points"], dtype=np.float64)
        # Newer exports carry only `precomputed_*`; older ones carry both.
        nr, nt, nz = (int(f.attrs[k] if k in f.attrs else f.attrs[alt])
                      for k, alt in (("n_rho", "precomputed_nr"),
                                     ("n_theta", "precomputed_ntheta"),
                                     ("n_zeta", "precomputed_nzeta")))
        file_nfp = int(f.attrs["nfp"])
        R = np.asarray(f["R"], dtype=np.float64).reshape(nr, nt, nz)
        Z = np.asarray(f["Z"], dtype=np.float64).reshape(nr, nt, nz)
    nfp = file_nfp if nfp is None else int(nfp)

    r_raw = ep[:, 0].reshape(nr, nt, nz)[:, 0, 0]
    t_raw = ep[:, 1].reshape(nr, nt, nz)[0, :, 0]
    z_raw = ep[:, 2].reshape(nr, nt, nz)[0, 0, :]

    r_ax, R = _radial_axis(r_raw, R, 0, stride)
    _, Z = _radial_axis(r_raw, Z, 0, stride)
    t_ax, R, lay_t = _periodic_axis(t_raw, R, 1, stride)
    _, Z, _ = _periodic_axis(t_raw, Z, 1, stride)
    z_ax, R, lay_z = _periodic_axis(z_raw, R, 2, stride)
    _, Z, _ = _periodic_axis(z_raw, Z, 2, stride)
    return (r_ax, t_ax, z_ax), R, Z, nfp, (lay_t, lay_z)


def _rgi_fn(axes, grid):
    """Linear grid bridge; returns ``f(xi:(3,)) -> (1,)`` for Greville collocation."""
    pts = tuple(jnp.asarray(a) for a in axes)
    interp = RegularGridInterpolator(
        pts, jnp.asarray(grid), method="linear",
        bounds_error=False, fill_value=None)   # extrapolate (rho < rho[0])

    def f(xi):
        return interp(xi.reshape(1, 3))[0:1]
    return f


def knots_at_data(x, p, kind):
    """Knot vector on which the degree-``p`` interpolant through the sample
    ``x`` is well posed for ANY monotone sample (Schoenberg-Whitney).

    A uniform knot vector is interpolatory only for uniform data; a sample
    refined toward the edge on uniform knots is singular or nearly so.
    Clamped: de Boor's knot averaging, each interior knot the mean of ``p``
    consecutive data points, with the domain ends ``0`` and ``1`` as the
    repeated knots (the sample need not reach them). Periodic: the half-open
    sample on ``[0, 1)`` IS the knot set, in the layout ``SplineBasis``
    uses, so it must start at 0.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if np.any(np.diff(x) <= 0):
        raise ValueError("sample axis is not strictly increasing")
    if kind == "periodic":
        if x[0] != 0.0 or x[-1] >= 1.0:
            raise ValueError(f"periodic sample must be half-open on [0, 1) "
                             f"starting at 0, got [{x[0]}, {x[-1]}]")
        T = np.concatenate([x, [1.0]])
        return jnp.asarray(np.concatenate([T[-(p + 1):-1] - 1.0, T, T[1:p + 1] + 1.0]))
    if kind != "clamped":
        raise ValueError(f"no data-knot rule for spline type {kind!r}")
    if x[0] < 0.0 or x[-1] > 1.0:
        raise ValueError(f"clamped sample must lie in [0, 1], got [{x[0]}, {x[-1]}]")
    interior = np.array([x[j:j + p].mean() for j in range(1, n - p)])
    return jnp.asarray(np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)]))


def fit_scalar_spline(axes, values, types, degree=3):
    """Interpolatory tensor-product spline through grid data, as a callable.

    ``n_basis = n_data`` per axis on the knots :func:`knots_at_data` places
    from the sample itself, one square collocation solve each (the fit
    ``mrx.projectors.load_grid_field`` step 1 does), kept as a function so its
    derivatives can be taken exactly. The axes only need to form a monotone
    tensor grid; refining the radial sample toward the edge is fine.
    Evaluation is three 1-D contractions: the hegna fit has ~5e5 basis
    functions, and a ``DiscreteFunction`` would evaluate all of them per
    point.
    """
    n = tuple(len(a) for a in axes)
    Ts = [knots_at_data(x, degree, kind) for x, kind in zip(axes, types)]
    fit = DifferentialForm(0, n, (degree,) * 3, types, Ts=Ts)
    C = jnp.asarray(values).reshape(n)
    for a, (basis, x) in enumerate(zip(fit.Λ, axes)):
        C = _solve_tensor_collocation_axis(
            basis.collocation_matrix(jnp.asarray(x)), C, axis=a)

    br, bt, bz = fit.Λ

    def evaluate(x):
        vr = jax.vmap(lambda i: br(x[0], i))(br.ns)
        vt = jax.vmap(lambda i: bt(x[1], i))(bt.ns)
        vz = jax.vmap(lambda i: bz(x[2], i))(bz.ns)
        return jnp.einsum('ijk,i,j,k->', C, vr, vt, vz)

    return evaluate
