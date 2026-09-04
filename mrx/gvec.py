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
radial splines projected onto the map's radial basis, the angular modes
through the periodic B-spline's Fourier transform), while :func:`load_clebsch` histopolates ``lambda`` at the quadrature
points from the closed form. Nothing is evaluated on a grid
(``docs/research/analytic_map_2026-08-28.md``). Validated against the pyGVEC
export of W7-X FMM002 to round-off (2026-08-27).

Two conventions the state carries:

* **Handedness.** ``mrx.mappings.stellarator_map`` uses
  ``Y = -R sin(2 pi zeta/nfp)``, which mirrors raw GVEC data
  (``det DF < 0``). :func:`build_gvec_map` measures the sign instead of
  assuming it.
* **nfp.** It enters the map as
  ``F = (R cos(2 pi zeta/nfp), +-R sin(2 pi zeta/nfp), Z)``, so a wrong value
  wraps one field period through the wrong angle with a healthy Jacobian to
  hide it; every reader takes an ``nfp`` override.

Every function takes the file path; the extension decides the route --
``.dat`` the GVEC state, ``.nc`` a VMEC wout refit into the same blocks by
``mrx.vmec``; anything else raises. ``test/synthetic_gvec.py`` writes a
state file for an analytic circular torus; the test suite reads it through
the same functions as a real one.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

from mrx.precision import DTYPE
from mrx.differential_forms import DiscreteFunction
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
    the reference the spline map is measured against); otherwise a scalar.
    A block that carries its own knot vector ``T`` (the wout route,
    ``mrx.vmec``) overrides the element grid ``sp``, which may then be
    ``None``. ``rho`` is not clipped to ``[0, 1]``: the local evaluator
    continues the end polynomial pieces outside, and a clip halves the
    autodiff radial derivative at ``rho = 1`` exactly (JAX splits the
    gradient of a tie), which halved the series map's ``det DF`` at the
    wall (measured 2026-08-28)."""

    def __init__(self, block, sp, nfp, vector=False):
        self.basis = SplineBasis(block["coef"].shape[1], block["deg"], "clamped",
                                 T=jnp.asarray(block_knots(block, sp)))
        self.C = jnp.asarray(block["coef"])                              # (n_modes, n_base)
        self.m = jnp.asarray(block["m"], dtype=DTYPE)
        self.n_per = jnp.asarray(block["n"], dtype=DTYPE) / nfp    # per field period
        self.cos = block["sin_cos"] == 2
        self.vector = vector

    def __call__(self, x):
        vals, idx = self.basis.evaluate_local(x[0])
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
    """Real Fourier symbol ``sum_l row[l] cos(2 pi m l / N)`` of the circulant
    matrix with first column ``row``, evaluated at each mode ``m`` in ``freqs``.

    ``row`` is the first column of ``M = B^T W B`` for a uniform periodic
    B-spline basis, which is symmetric (``M = M^T``) AND circulant (the basis
    is shift invariant), so ``row[l] = row[N - l]`` exactly and every
    eigenvalue is real. The Gauss-quadrature assembly leaves that symmetry
    exact only to round-off, and a raw complex transform of the round-off-
    asymmetric row then carries a spurious imaginary part whose size depends
    on ``m`` (small ``freqs`` on one file, large on another) -- so enforce the
    known structure by symmetrising ``row`` and take the real cosine
    transform, after checking the input departed from symmetry only at
    round-off. A genuinely non-uniform basis is not circulant and raises.

    ``collocation_matrix`` runs at the WORKING precision, so a float32
    relaxation (every relaxation; the convergence study is float64) assembles
    this row with ~1e-7 round-off and its raw asymmetry reaches ~3e-8 --
    symmetrising removes exactly that antisymmetric noise. The guard admits
    round-off to ``1e-4`` (well above float32's ~1e-7, well below the ~1e-2
    asymmetry a genuinely non-uniform basis would show)."""
    N = len(row)
    row = np.asarray(row, dtype=np.float64)
    sym_row = 0.5 * (row + np.concatenate([row[:1], row[1:][::-1]]))
    if np.abs(row - sym_row).max() > 1e-4 * np.abs(row).max():
        raise ValueError("periodic collocation/mass row is not symmetric")
    return np.cos(2 * np.pi * np.outer(freqs, np.arange(N)) / N) @ sym_row


def _angular_symbol(basis, freqs):
    """Per-mode coefficient factor ``gamma(m)`` of the uniform periodic
    basis: the degree-``p`` spline with coefficients ``gamma(m) exp(2 pi i m
    x_j)`` (``x_j`` the Greville points, the centres of the basis functions)
    is the L2 projection of ``exp(2 pi i m theta)``.

    The moments are the B-spline's Fourier transform,
    ``int B_j(theta) exp(2 pi i m theta) dtheta = h sinc(m h)^(p+1)
    exp(2 pi i m x_j)`` with ``h = 1/N`` and ``sinc(x) = sin(pi x)/(pi x)``,
    and the mass matrix is circulant with the symbol ``mu(m) = sum_l M_l0
    exp(-2 pi i m l / N)``, so ``gamma(m) = h sinc(m h)^(p+1) / mu(m)``. A
    mode beyond the Nyquist frequency ``N/2`` is damped by ``sinc^(p+1)``
    where an interpolant would alias it with gain up to ``1/sigma(N/2) = 3``
    at ``p = 3``. The mass matrix is assembled by Gauss quadrature, exact
    for the spline product.
    """
    N, p = basis.n, basis.p
    freqs = np.asarray(freqs, dtype=np.float64)
    xi, wi = np.polynomial.legendre.leggauss(p + 1)
    pts = ((np.arange(N)[:, None] + 0.5 * (xi[None, :] + 1.0)) / N).ravel()
    w = np.tile(0.5 * wi / N, N)
    B = np.asarray(basis.collocation_matrix(jnp.asarray(pts)), dtype=np.float64)
    M = B.T @ (w[:, None] * B)
    moment = np.sinc(freqs / N) ** (p + 1) / N
    return moment / _periodic_symbol(M[:, 0], freqs)


def _radial_coefficients(block, sp, basis_r):
    """``(n_r, n_modes)`` coefficients on the map's clamped radial basis of
    the L2 projection of every mode's radial function ``c_mn(rho)``, a
    spline on the state's knots: the moments by Gauss quadrature on the
    union of the two knot sets (exact for the spline product) and one
    ``n_r x n_r`` mass solve shared by all modes. Exact when the map's
    radial space contains the state's (GVEC's degree-5 basis on 10 uniform
    elements at ``p = 5``, ``n_r = 15``)."""
    T_s, deg_s, C = block_knots(block, sp), block["deg"], block["coef"]
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


def series_tensor_coefficients(block, sp, nfp, seq):
    """``(n_r, n_t, n_z)`` coefficients on the tensor-product 0-form space
    of ``seq`` of the L2 projection of a state field, built from its
    coefficients alone: no evaluation grid, no collocation solve.

    The field is ``sum_mn c_mn(rho) trig(2 pi (m theta - n zeta / nfp))``
    and the L2 projection onto a tensor-product spline space is linear and
    tensor-product, so the coefficients are the sum over modes of (radial
    coefficients of ``c_mn``, :func:`_radial_coefficients`) x (angular
    coefficients of the trig mode, in closed form, :func:`_angular_symbol`):

        C[i, j, k] = sum_mn c_mn[i] gamma_t(m) gamma_z(n) trig(2 pi (m x_j - n y_k))

    with ``x_j``, ``y_k`` the angular Greville points.
    """
    br, bt, bz = seq.basis_0.Λ
    m, n_per = block["m"].astype(np.float64), block["n"] / nfp
    if np.abs(n_per - np.round(n_per)).max() > 0:
        raise ValueError("toroidal mode numbers are not multiples of nfp")
    c_r = _radial_coefficients(block, sp, br)                            # (n_r, n_modes)
    gamma = _angular_symbol(bt, m) * _angular_symbol(bz, n_per)          # (n_modes,)
    x_t = np.asarray(bt.greville_points(), dtype=np.float64)
    y_z = np.asarray(bz.greville_points(), dtype=np.float64)
    arg = TWO_PI * (m[:, None, None] * x_t[None, :, None]
                    - n_per[:, None, None] * y_z[None, None, :])          # (n_modes, n_t, n_z)
    trig = np.cos(arg) if block["sin_cos"] == 2 else np.sin(arg)
    return np.einsum("ik,k,kjl->ijl", c_r, gamma, trig)


def series_spline_dofs(block, sp, nfp, seq):
    """The polar 0-form DoFs on ``seq.basis_0`` of a state field: the tensor
    coefficients of :func:`series_tensor_coefficients` restricted onto the
    polar space with the ring-0/ring-1 surgery of every 0-form
    interpolation (:func:`mrx.projectors._conforming_restriction`). The
    Greville interpolant of the series -- identical to sampling it at the
    Greville points and solving -- was measured against this projection
    and dropped (``docs/research/analytic_map_2026-08-28.md``)."""
    C_full = series_tensor_coefficients(block, sp, nfp, seq)
    return _conforming_restriction(seq.E(0), jnp.asarray(C_full.reshape(-1)))


def read_equilibrium(path):
    """The state dict of a GVEC state (``.dat``) or a VMEC wout (``.nc``,
    refit into the same blocks by :func:`mrx.vmec.read_wout`); any other
    extension raises."""
    if path.endswith(".dat"):
        return read_state(path)
    if path.endswith(".nc"):
        from mrx.vmec import read_wout  # noqa: PLC0415  (imports this module)
        return read_wout(path)
    raise ValueError(f"{path}: not an equilibrium file; MRX reads GVEC state "
                     "files (.dat) and VMEC wout files (.nc)")


def build_gvec_map(path, seq, sign=None, nfp=None):
    """Build the stellarator map of a GVEC state or a VMEC wout as a C1
    polar spline map on ``seq.basis_0``.

    A ``.dat`` state or a ``.nc`` wout (refit into the same blocks by
    :mod:`mrx.vmec`) supplies ``R`` and ``Z`` as radial-spline x Fourier
    series, and the map's spline coefficients are the L2 projection built
    from the series coefficients (:func:`series_spline_dofs`) -- nothing is
    evaluated on a grid; any other file raises.
    Returns ``(F, info)``. ``sign`` is the toroidal handedness
    ``Y = sign * R sin(2 pi zeta/nfp)``; left ``None`` it is measured, and a
    file that is degenerate under both signs raises.
    """
    st = read_equilibrium(path)
    nfp = st["nfp"] if nfp is None else int(nfp)
    R_fn = StateField(st["X1"], st.get("sp"), st["nfp"], vector=True)
    Z_fn = StateField(st["X2"], st.get("sp"), st["nfp"], vector=True)
    R_dof = series_spline_dofs(st["X1"], st.get("sp"), st["nfp"], seq)
    Z_dof = series_spline_dofs(st["X2"], st.get("sp"), st["nfp"], seq)
    R_h = DiscreteFunction(R_dof, seq.basis_0, seq.E(0))
    Z_h = DiscreteFunction(Z_dof, seq.basis_0, seq.E(0))

    tried = {}
    for s in ((sign,) if sign is not None else (1.0, -1.0)):
        F = _map_with_sign(R_h, Z_h, nfp, s)
        d = _det_DF(F)
        tried[s] = (float(d.min()), float(d.max()))
        if np.isfinite(d).all() and d.min() > 0:
            return F, {"R_h": R_h, "Z_h": Z_h, "R_fn": R_fn, "Z_fn": Z_fn,
                       "nfp": nfp, "sign": s, "det_range": tried[s]}
    raise RuntimeError(f"{path}: no handedness gives det DF > 0; "
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
                p=profile_spline(st, "pressure")(rho),
                lam_h=StateField(st["LA"], st["sp"], st["nfp"]))


def load_clebsch(path):
    """Read the radial profiles, a lambda callable and p(rho) from a file.

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
    on ``rho``) and ``lam_h`` (the closed-form :class:`StateField` of
    ``LA``): a GVEC state
    (``.dat``) through :func:`load_state_clebsch`, a VMEC wout (``.nc``)
    through :func:`mrx.vmec.load_wout_clebsch`; anything else raises.
    """
    if path.endswith(".dat"):
        return load_state_clebsch(path)
    if path.endswith(".nc"):
        from mrx.vmec import load_wout_clebsch  # noqa: PLC0415  (imports this module)
        return load_wout_clebsch(path)
    raise ValueError(f"{path}: not an equilibrium file; MRX reads GVEC state "
                     "files (.dat) and VMEC wout files (.nc)")


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
