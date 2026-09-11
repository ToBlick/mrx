"""GVEC equilibria: state files read in closed form.

The input is the **state file** (``GVEC_State_*.dat``), GVEC's own
representation of an equilibrium:
    the radial B-spline basis (degree ``deg`` on the element grid ``sp``),
    the Fourier mode table ``(m, n)`` with ``n`` already multiplied by ``nfp``
    and, per mode, the radial coefficients of ``X1 = R`` (cosine series),
    ``X2 = Z`` and ``LA = lambda`` (sine series),
    followed by the profiles ``Phi``, ``chi``, ``iota``, ``p`` at the
    radial interpolation points of the ``X1`` basis.
Its angles are GVEC's ``theta`` and ``zeta`` in radians with the series
``sum f_mn(s) trig(m theta - n zeta)``; ``s`` is the radial label we call
``rho`` (``Phi = Phi_edge s^2``). :class:`StateField`
evaluates one of the three fields at a logical point in JAX -- the radial
basis through :class:`mrx.spline_bases.SplineBasis` on GVEC's own knots, the
angles as ``2 pi (m theta - n zeta / nfp)`` -- and :func:`build_gvec_map`
builds the map's polar spline coefficients of ``R`` and ``Z`` as the L2
projection of the series, mode by mode (:func:`series_spline_dofs`: the
radial splines projected onto the map's radial basis, the angular modes by
their moments against the periodic bases and one 1-D mass solve per mode,
uniform or not), while :func:`load_clebsch` histopolates ``lambda`` at the quadrature
points from the closed form. Nothing is evaluated on a grid
(``docs/research/analytic_map_2026-08-28.md``). Validated against the pyGVEC
export of W7-X FMM002 to round-off (2026-08-27).

Two conventions the state carries:

* **Handedness.** :func:`_map_with_sign` uses
  ``Y = -R sin(2 pi zeta/nfp)``, which mirrors raw GVEC data
  (``det DF < 0``). :func:`build_gvec_map` measures the sign instead of
  assuming it.
* **nfp.** It enters the map as
  ``F = (R cos(2 pi zeta/nfp), +-R sin(2 pi zeta/nfp), Z)``, so a wrong value
  wraps one field period through the wrong angle with a healthy Jacobian to
  hide it; every reader takes an ``nfp`` override.

:func:`read_equilibrium` takes the file path and the extension decides the
route -- ``.dat`` the GVEC state, ``.nc`` a VMEC wout refit into the same
blocks by ``mrx.vmec``; anything else raises -- and every other function
takes the parsed state. ``test/synthetic_gvec.py`` writes a state file for
an analytic circular torus; the test suite reads it through the same
functions as a real one.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

from mrx.precision import DTYPE
from mrx.differential_forms import DiscreteFunction, det33
from mrx.projectors import _conforming_restriction
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
    ``(n_modes, n_base)``, ``sin_cos`` 1 = sine, 2 = cosine, ``deg`` and
    the clamped radial knot vector ``T`` on the element grid), ``profiles``
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
    st["nfp"] = int(_numbers(block("global")[0])[0])
    for name in ("X1", "X2", "LA"):
        n_base, deg, _, n_modes, sin_cos, _ = (int(v) for v in _numbers(block(f"{name}_base")[0]))
        rows = np.array([_numbers(ln) for ln in block(f"{name}:")])
        if rows.shape != (n_modes, 2 + n_base):
            raise ValueError(f"{path}: {name} block is {rows.shape}, expected {(n_modes, 2 + n_base)}")
        st[name] = dict(m=rows[:, 0].astype(int), n=rows[:, 1].astype(int),
                        coef=rows[:, 2:], sin_cos=sin_cos, deg=deg, T=knots(st["sp"], deg))
    st["deg"] = st["X1"]["deg"]
    prof = np.array([_numbers(ln) for ln in block("at X1_base IP point positions")])
    st["profiles"] = dict(zip(("s", "phi", "chi", "iota", "pressure"), prof.T))
    st["a_minor"], st["r_major"], st["volume"] = _numbers(block("a_minor,r_major,volume")[0])
    return st


def knots(sp, deg):
    """Clamped knot vector of the degree-``deg`` B-splines on the element grid."""
    return np.concatenate([np.full(deg, sp[0]), sp, np.full(deg, sp[-1])])


def evaluate(block, s, theta, zeta):
    """A field block on the tensor grid ``s x theta x zeta`` (angles in
    radians, ``zeta`` the physical toroidal angle)."""
    A = BSpline.design_matrix(np.asarray(s, dtype=np.float64), block["T"],
                              block["deg"]).toarray() @ block["coef"].T   # (n_s, n_modes)
    arg = (np.outer(block["m"], theta)[:, :, None]
           - np.outer(block["n"], zeta)[:, None, :])                      # (n_modes, n_t, n_z)
    F = np.cos(arg) if block["sin_cos"] == 2 else np.sin(arg)
    return np.einsum("sk,ktz->stz", A, F)


def profile_spline(st, name):
    """The radial spline through a profile's interpolation-point values."""
    prof, deg = st["profiles"], st["deg"]
    T = knots(st["sp"], deg)
    c = np.linalg.solve(BSpline.design_matrix(prof["s"], T, deg).toarray(), prof[name])
    return BSpline(T, c, deg)


class StateField:
    """A state's ``X1``, ``X2`` or ``LA`` as a JAX function of the logical
    point ``(rho, theta, zeta)`` (angles on ``[0, 1)``, ``zeta`` per field
    period), on the block's own radial knots ``T``. ``rho`` is not clipped
    to ``[0, 1]``: the local evaluator continues the end polynomial pieces
    outside, and a clip halves the autodiff radial derivative at ``rho =
    1`` exactly (JAX splits the gradient of a tie), which halved the series
    map's ``det DF`` at the wall."""

    def __init__(self, block, nfp):
        self.basis = SplineBasis(block["coef"].shape[1], block["deg"], "clamped",
                                 T=jnp.asarray(block["T"]))
        self.C = jnp.asarray(block["coef"])                              # (n_modes, n_base)
        self.m = jnp.asarray(block["m"], dtype=DTYPE)
        self.n_per = jnp.asarray(block["n"], dtype=DTYPE) / nfp    # per field period
        self.cos = block["sin_cos"] == 2

    def __call__(self, x):
        vals, idx = self.basis.evaluate_local(x[0])
        radial = self.C[:, idx] @ vals                                    # (n_modes,)
        arg = 2.0 * jnp.pi * (self.m * x[1] - self.n_per * x[2])
        return (jnp.cos(arg) if self.cos else jnp.sin(arg)) @ radial


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
    dets = jax.vmap(lambda x: det33(jax.jacfwd(map_func)(x)))(xs)
    return np.asarray(dets)


def _angular_coefficients(basis, freqs):
    """``(n_modes, N)`` complex coefficients on the periodic ``basis`` of the
    L2 projection of ``exp(2 pi i f theta)`` for every frequency ``f`` in
    ``freqs``: the moments ``int B_j(theta) exp(2 pi i f theta) dtheta`` by
    Gauss quadrature on the basis's cells, with enough points for the
    mode's phase across the widest cell, and one ``N x N`` mass solve
    shared by all modes, at setup. On a uniform basis this is the closed
    form ``gamma(f) exp(2 pi i f x_j)`` (``x_j`` the Greville points,
    ``gamma`` the sinc-damped moment over the circulant mass symbol); a
    non-uniform basis has no closed form and needs nothing else."""
    p = basis.p
    freqs = np.asarray(freqs, dtype=np.float64)
    T = np.asarray(basis.T, dtype=np.float64)
    bp = np.unique(T[(T >= 0.0) & (T <= 1.0)])
    lo, hi = bp[:-1], bp[1:]
    q = p + 7 + int(np.ceil(TWO_PI * np.abs(freqs).max() * (hi - lo).max()))
    xi, wi = np.polynomial.legendre.leggauss(q)
    pts = (0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * xi[None, :]).ravel()
    w = (0.5 * (hi - lo)[:, None] * wi[None, :]).ravel()
    B = np.asarray(basis.collocation_matrix(jnp.asarray(pts)), dtype=np.float64)
    M = B.T @ (w[:, None] * B)
    moments = B.T @ (w[:, None] * np.exp(1j * TWO_PI * np.outer(pts, freqs)))   # (N, n_modes)
    return np.linalg.solve(M, moments).T


def _radial_coefficients(block, basis_r):
    """``(n_r, n_modes)`` coefficients on the map's clamped radial basis of
    the L2 projection of every mode's radial function ``c_mn(rho)``, a
    spline on the state's knots: the moments by Gauss quadrature on the
    union of the two knot sets (exact for the spline product) and one
    ``n_r x n_r`` mass solve shared by all modes. Exact when the map's
    radial space contains the state's (GVEC's degree-5 basis on 10 uniform
    elements at ``p = 5``, ``n_r = 15``)."""
    T_s, deg_s, C = np.asarray(block["T"]), block["deg"], block["coef"]
    T_r, p_r = np.asarray(basis_r.T, dtype=np.float64), basis_r.p
    bp = np.unique(np.concatenate([T_s, T_r]))
    xi, wi = np.polynomial.legendre.leggauss((deg_s + p_r) // 2 + 1)
    lo, hi = bp[:-1], bp[1:]
    pts = (0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * xi[None, :]).ravel()
    w = (0.5 * (hi - lo)[:, None] * wi[None, :]).ravel()
    Br = BSpline.design_matrix(pts, T_r, p_r).toarray()
    Bs = BSpline.design_matrix(pts, T_s, deg_s).toarray()
    M = Br.T @ (w[:, None] * Br)
    return np.linalg.solve(M, Br.T @ (w[:, None] * (Bs @ C.T)))


def series_tensor_coefficients(block, nfp, seq):
    """``(n_r, n_t, n_z)`` coefficients on the tensor-product 0-form space
    of ``seq`` of the L2 projection of a state field, built from its
    coefficients alone: no evaluation grid, no collocation solve.

    The field is ``sum_mn c_mn(rho) trig(2 pi (m theta - n zeta / nfp))``
    and the L2 projection onto a tensor-product spline space is linear and
    tensor-product, so the coefficients are the sum over modes of (radial
    coefficients of ``c_mn``, :func:`_radial_coefficients`) x (angular
    coefficients of the trig mode, :func:`_angular_coefficients`): with
    ``A_t[m, j]`` and ``A_z[n, k]`` the coefficients of ``exp(2 pi i m
    theta)`` and ``exp(2 pi i n zeta)`` on the two periodic bases,

        C[i, j, k] = sum_mn c_mn[i] Re/Im(A_t[m, j] conj(A_z[n, k]))

    (cosine series: the real part; sine series: the imaginary part), the
    angular bases uniform or not.
    """
    br, bt, bz = seq.basis_0.Λ
    m, n_per = block["m"].astype(np.float64), block["n"] / nfp
    if np.abs(n_per - np.round(n_per)).max() > 0:
        raise ValueError("toroidal mode numbers are not multiples of nfp")
    c_r = _radial_coefficients(block, br)                                # (n_r, n_modes)
    A_t = _angular_coefficients(bt, m)                                   # (n_modes, n_t)
    A_z = _angular_coefficients(bz, n_per)                               # (n_modes, n_z)
    modes = A_t[:, :, None] * np.conj(A_z)[:, None, :]                   # exp(2 pi i (m theta - n zeta))
    trig = modes.real if block["sin_cos"] == 2 else modes.imag           # (n_modes, n_t, n_z)
    return np.einsum("ik,kjl->ijl", c_r, trig)


def series_spline_dofs(block, nfp, seq):
    """The polar 0-form DoFs on ``seq.basis_0`` of a state field: the tensor
    coefficients of :func:`series_tensor_coefficients` restricted onto the
    polar space with the ring-0/ring-1 surgery of every 0-form
    interpolation (:func:`mrx.projectors._conforming_restriction`). The
    Greville interpolant of the series -- identical to sampling it at the
    Greville points and solving -- was measured against this projection
    and dropped (``docs/research/analytic_map_2026-08-28.md``)."""
    C_full = series_tensor_coefficients(block, nfp, seq)
    return _conforming_restriction(seq.E(0), jnp.asarray(C_full.reshape(-1)))


def read_equilibrium(path):
    """The state dict of a GVEC state (``.dat``) or a VMEC wout (``.nc``,
    refit into the same blocks by :func:`mrx.vmec.read_wout`), with
    ``kind`` (``"gvec"`` or ``"vmec"``) and ``path``; any other extension
    raises. Read once per run: :func:`mrx.geometry.build_sequence` keeps
    it on the sequence (``seq.equilibrium``) for the initial field."""
    if path.endswith(".dat"):
        return dict(read_state(path), kind="gvec", path=path)
    if path.endswith(".nc"):
        from mrx.vmec import read_wout  # noqa: PLC0415  (imports this module)
        return dict(read_wout(path), kind="vmec", path=path)
    raise ValueError(f"{path}: not an equilibrium file; MRX reads GVEC state "
                     "files (.dat) and VMEC wout files (.nc)")


def build_gvec_map(st, seq, nfp=None):
    """Build the stellarator map of a GVEC state or a VMEC wout (``st``,
    the parsed file of :func:`read_equilibrium`) as a C1 polar spline map
    on ``seq.basis_0``.

    The state supplies ``R`` and ``Z`` as radial-spline x Fourier series,
    and the map's spline coefficients are the L2 projection built from the
    series coefficients (:func:`series_spline_dofs`) -- nothing is
    evaluated on a grid. Returns ``(F, info)`` with ``info`` the ``nfp``,
    the measured toroidal handedness ``sign`` (``Y = sign * R sin(2 pi
    zeta/nfp)``; a file that is degenerate under both signs raises) and the
    sampled ``det_range``.
    """
    nfp = st["nfp"] if nfp is None else int(nfp)
    R_h = DiscreteFunction(series_spline_dofs(st["X1"], st["nfp"], seq), seq.basis_0, seq.E(0))
    Z_h = DiscreteFunction(series_spline_dofs(st["X2"], st["nfp"], seq), seq.basis_0, seq.E(0))

    tried = {}
    for s in (1.0, -1.0):
        F = _map_with_sign(R_h, Z_h, nfp, s)
        d = _det_DF(F)
        tried[s] = (float(d.min()), float(d.max()))
        if np.isfinite(d).all() and d.min() > 0:
            return F, {"nfp": nfp, "sign": s, "det_range": tried[s]}
    raise RuntimeError(f"{st['path']}: no handedness gives det DF > 0; "
                       f"sampled ranges {tried}")


def load_clebsch(st):
    """The radial profiles, a lambda callable and p(rho) of a parsed
    equilibrium ``st`` (:func:`read_equilibrium`).

    The reference 2-form components of ``mrx.initial_conditions`` are exactly
    GVEC's ``sqrt(g) B^i``, verified against the pyGVEC export's own B:
    ``sqrt(g) B^theta = dchi_dr - dPhi_dr dLA_dz`` and
    ``sqrt(g) B^zeta = dPhi_dr (1 + dLA_dt)``, in GVEC's units (derivatives
    with respect to radian angles). The caller converts with
    ``Phi' = 2 pi dPhi_dr``, ``iota = dchi_dr / (nfp dPhi_dr)`` and
    ``lambda = LA / 2 pi``.

    lambda is handed over as the scalar and differentiated, never as two
    derivatives: ``div B = 0`` rests on the mixed partials cancelling, which
    holds only when both come from one function.

    Returns a dict with ``nfp``, ``rho``, ``dPhi``, ``dchi``, ``p`` (arrays
    on 401 uniform radii from the profile splines, ``chi' = iota Phi'``) and
    ``lam_h`` (the closed-form :class:`StateField` of ``LA``), from the
    state of :func:`read_equilibrium` (a GVEC state or a VMEC wout, whose
    profile splines live in ``rho = sqrt(s)``, :func:`mrx.vmec.profile_spline`).
    """
    if st["kind"] == "vmec":
        from mrx.vmec import profile_spline as spline  # noqa: PLC0415  (imports this module)
    else:
        spline = profile_spline
    rho = np.linspace(0.0, 1.0, 401)
    dPhi = spline(st, "phi").derivative()(rho)
    return dict(nfp=st["nfp"], rho=rho, dPhi=dPhi,
                dchi=spline(st, "iota")(rho) * dPhi,
                p=spline(st, "pressure")(rho),
                lam_h=StateField(st["LA"], st["nfp"]))


# ---------------------------------------------------------------------------
# 3. Knots for sampled radial data (the wout refit, mrx.vmec)
# ---------------------------------------------------------------------------

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
