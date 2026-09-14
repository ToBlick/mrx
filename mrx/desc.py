"""DESC equilibria: the ``.h5`` output read in closed form.

A DESC output file stores a *continuation family* of ``Equilibrium`` objects
under ``/_equilibria``; the last one is the converged solution. Each carries
``R``, ``Z`` and the stream function lambda as Fourier-Zernike series

    f(rho, theta, zeta) = sum_lmn c_lmn Z_l^|m|(rho) P_m(theta) T_n(zeta)

with ``Z_l^|m|`` the Zernike radial polynomial, ``P_m`` the real poloidal
factor (``cos(|m| theta)`` for ``m >= 0``, ``sin(|m| theta)`` for ``m < 0``)
and ``T_n`` the real toroidal factor (``cos(|n| nfp zeta)`` for ``n >= 0``,
``sin(|n| nfp zeta)`` for ``n < 0``). The coefficients ``c_lmn`` are
``_R_lmn`` / ``_Z_lmn`` / ``_L_lmn`` and the ``(l, m, n)`` table is the
matching ``_R_basis/_modes`` and friends.

This module turns that into the *same* block dict :func:`mrx.gvec.read_state`
produces, so every consumer downstream -- ``build_gvec_map``,
``load_clebsch``, ``clebsch_potential_form`` -- applies verbatim, exactly as
:mod:`mrx.vmec` does for a wout. Only ``h5py``, ``numpy`` and ``scipy`` are
imported: a DESC file is read without DESC installed. Three conversions
happen here and nowhere else:

* **Angular basis.** MRX (like GVEC and VMEC) uses a SINGLE trig of the
  combined angle, ``sum_mn f_mn(rho) trig(m theta_G - n zeta_G)``, while DESC
  uses a PRODUCT of two real trig factors. The two are related by the
  product-to-sum identities, so every DESC mode splits into the pair of MRX
  modes ``(|m|, +|n| nfp)`` and ``(|m|, -|n| nfp)`` at half weight
  (:func:`_split_weights`), collapsing onto one mode when ``n = 0``. The
  identities also fix the parity: DESC's ``sym='cos'`` filter keeps the
  modes with ``sign(m) = sign(n)``, and those are exactly the ones whose
  product is a cosine of the combined angle, so ``R`` lands in a cosine
  block and ``Z``/lambda in sine blocks with no further bookkeeping. The
  block's ``n`` is the full-turn index ``|n| nfp``, VMEC's ``xn`` convention.

* **Radial parameterisation.** DESC's radial label IS MRX's ``rho``
  (``sqrt(s)`` in both), so unlike the wout refit there is no radial remap.
  Each mode's radial function ``sum_l c_lmn Z_l^|m|(rho)`` is sampled at
  Chebyshev-Lobatto nodes -- clustered at both ends, where a Zernike
  polynomial of degree ``L`` varies fastest -- and refit as a clamped
  interpolatory B-spline by :func:`mrx.vmec._fit_block`, on data-placed
  knots (:func:`mrx.gvec.knots_at_data`) carried in the block as ``T``.
  ``_fit_block`` also imposes the ``rho^m`` axis parity of each mode; a
  Zernike series satisfies it identically, so here the conditions confirm
  the fit rather than correct it (they are load-bearing for VMEC, whose
  ``rmnc`` rows carry no such guarantee).

* **Flux units.** DESC stores the toroidal flux at the boundary as ``_Psi``
  in Webers, with ``Phi(rho) = Psi rho^2``. GVEC profiles store ``Phi / 2 pi``
  (flux per radian), so the profile is divided by ``2 pi`` here. ``_Psi`` is
  NEGATIVE in some files (HSX, W7-X), which reverses the field; nothing
  special is done, because ``dPhi < 0`` flows correctly through
  ``load_clebsch`` and the map's handedness is measured, not assumed
  (:func:`mrx.gvec.build_gvec_map`).

**The one quantity that may be missing is iota.** A DESC equilibrium is
constrained by either an iota profile or a current profile; in the current
case ``_iota`` is the string ``None`` in the file and the rotational
transform is an OUTPUT of the solve that is not stored. Every shipped DESC
stellarator example of that kind (NCSX, ARIES-CS, ESTELL, HSX, WISTELL-A,
precise_QA, precise_QH) therefore cannot be read by parsing alone. When
``_iota`` is absent this module falls back to DESC itself
(:func:`_iota_from_desc`: ``eq.compute("iota")``) and raises a pointed error
if DESC is not importable. Files written with an iota constraint (SOLOVEV,
HELIOTRON, W7-X, ATF, DSHAPE, and anything from
``VMECIO.load(..., profile="iota")``) need no such fallback.

Both storage layouts are read (:func:`_equilibrium`): the ``/_equilibria``
family a continuation run and every shipped example writes, whose LAST
member is the converged solution, and the flat single equilibrium that
``eq.save()`` -- and therefore ``VMECIO.load(...).save()`` -- produces.

Guards at read time: non-stellarator-symmetric files (``_sym = False``,
which add the missing-parity partners) are not implemented; a file that is
neither layout is refused with the HDF5 keys it does have.
``test/test_desc.py`` reads a synthetic file written by
``test/synthetic_desc.py``, the inverse of this parser.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import eval_jacobi

from mrx.vmec import _fit_block

TWO_PI = 2.0 * np.pi

#: Spline degree of the refit blocks, matching :mod:`mrx.vmec`.
DEG = 3

#: Smallest radial sample count, and the multiple of DESC's radial
#: resolution ``L`` used above it. A degree-``L`` Zernike polynomial needs
#: the cubic interpolant to resolve its ``L/2`` interior extrema; four
#: nodes per extremum is the measured knee (``test_desc.py``).
MIN_NODES, NODES_PER_L = 65, 4
MAX_NODES = 257


def _nodes(n_rho: int) -> np.ndarray:
    """Chebyshev-Lobatto nodes on ``[0, 1]``: ``rho[0] = 0``, ``rho[-1] = 1``,
    strictly increasing, clustered at both ends.

    Args:
        n_rho: number of nodes, at least 3.

    Returns:
        The nodes, shape ``(n_rho,)``.
    """
    if n_rho < 3:
        raise ValueError(f"n_rho = {n_rho} < 3")
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n_rho) / (n_rho - 1)))


def _zernike_radial(rho: np.ndarray, ell: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The Zernike radial polynomial ``Z_l^|m|(rho)``, DESC's ``zernike_radial``.

    Evaluated in the Jacobi form DESC uses,
    ``(-1)^k rho^|m| P_k^(|m|, 0)(1 - 2 rho^2)`` with ``k = (l - |m|) / 2``,
    which is stable to high degree; modes of the wrong parity
    (``l - |m|`` odd) are zero, as in DESC.

    Args:
        rho: radii, shape ``(n_rho, 1)`` or broadcastable against ``ell``.
        ell: radial mode numbers, shape ``(1, n_modes)`` or broadcastable.
        m: poloidal mode numbers (sign ignored), broadcastable with ``ell``.

    Returns:
        The polynomials evaluated at ``rho``, broadcast shape.
    """
    m_abs = np.abs(np.asarray(m)).astype(float)
    k = (np.asarray(ell) - m_abs) / 2.0
    out = (rho ** m_abs) * eval_jacobi(k, m_abs, 0.0, 1.0 - 2.0 * rho ** 2)
    return np.where(k == np.floor(k), (-1.0) ** np.floor(k) * out, 0.0)


def _split_weights(m: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Product-to-sum weights of one DESC mode on the two MRX modes it feeds.

    With ``a = |m|`` and ``b = |n| nfp``, a DESC basis function is one of

    * ``cos(a t) cos(b z) = (cos(a t - b z) + cos(a t + b z)) / 2``
    * ``sin(a t) sin(b z) = (cos(a t - b z) - cos(a t + b z)) / 2``
    * ``sin(a t) cos(b z) = (sin(a t - b z) + sin(a t + b z)) / 2``
    * ``cos(a t) sin(b z) = (sin(a t + b z) - sin(a t - b z)) / 2``

    and MRX's mode ``(a, N)`` is ``trig(a t - N z)``, so ``a t - b z`` is
    ``N = +b`` and ``a t + b z`` is ``N = -b``. Every weight is ``+-1/2``:
    the plus branch flips sign only for ``cos x sin`` and the minus branch
    only for ``sin x sin``.

    Args:
        m: DESC poloidal mode numbers, signed.
        n: DESC toroidal mode numbers, signed.

    Returns:
        ``(w_plus, w_minus)``, the weights on ``N = +|n| nfp`` and
        ``N = -|n| nfp``. When ``n = 0`` the two modes coincide and the
        weights add to one.
    """
    m, n = np.asarray(m), np.asarray(n)
    w_plus = np.where((m >= 0) & (n < 0), -0.5, 0.5)
    w_minus = np.where((m < 0) & (n < 0), -0.5, 0.5)
    return w_plus, w_minus


def _is_cosine(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Whether a DESC mode's product is a COSINE of the combined angle.

    True for ``cos x cos`` and ``sin x sin``, i.e. for ``sign(m) = sign(n)``
    with zero counted as non-negative -- DESC's ``sym='cos'`` filter.
    """
    m, n = np.asarray(m), np.asarray(n)
    return (m >= 0) == (n >= 0)


def _convert_block(modes: np.ndarray, coef: np.ndarray, nfp: int,
                   rho: np.ndarray, name: str, deg: int = DEG) -> dict[str, Any]:
    """One Fourier-Zernike field as a :class:`mrx.gvec.StateField` block.

    The DESC modes are expanded onto the single-angle modes of
    :func:`_split_weights`, accumulated per ``(m, n)`` pair, sampled at
    ``rho`` and refit by :func:`mrx.vmec._fit_block`.

    Args:
        modes: ``(K, 3)`` table of ``(l, m, n)``, DESC's ``_*_basis/_modes``.
        coef: ``(K,)`` coefficients, DESC's ``_*_lmn``.
        nfp: field periods, to turn ``n`` into the full-turn index.
        rho: radial sample nodes with ``rho[0] = 0``.
        name: field name, for error messages.
        deg: spline degree of the refit.

    Returns:
        The block dict: ``m``, ``n``, ``coef`` of shape
        ``(n_modes, n_base)``, ``sin_cos``, ``deg`` and the knots ``T``.

    Raises:
        ValueError: if the field mixes cosine and sine modes of the combined
            angle, which a stellarator-symmetric file never does.
    """
    ell, m, n = (modes[:, i].astype(int) for i in range(3))
    cosine = _is_cosine(m, n)
    live = np.abs(coef) > 0.0
    if live.any() and not (cosine[live].all() or (~cosine[live]).all()):
        raise ValueError(f"{name}: mixes cos and sin modes of (m theta - n zeta); "
                         "only stellarator-symmetric DESC files are supported")
    sin_cos = 2 if (not live.any() or cosine[live][0]) else 1

    radial = _zernike_radial(rho[:, None], ell[None, :], m[None, :]) * coef[None, :]
    w_plus, w_minus = _split_weights(m, n)
    m_abs, n_full = np.abs(m), np.abs(n) * nfp

    # Accumulate the two branches of every DESC mode onto the (m, n) table.
    columns: dict[tuple[int, int], np.ndarray] = {}
    for key_n, weight in ((n_full, w_plus), (-n_full, w_minus)):
        for i in range(len(ell)):
            key = (int(m_abs[i]), int(key_n[i]))
            col = columns.get(key)
            if col is None:
                col = columns[key] = np.zeros(len(rho))
            col += weight[i] * radial[:, i]

    keys = sorted(k for k, v in columns.items() if np.abs(v).max() > 0.0)
    if keys:
        samples = np.stack([columns[k] for k in keys], axis=1)
    else:                     # identically zero (lambda of an up-down field)
        keys, samples = [(0, 0)], np.zeros((len(rho), 1))
    m_out = np.array([k[0] for k in keys], dtype=int)
    n_out = np.array([k[1] for k in keys], dtype=int)
    return _fit_block(rho, samples, sin_cos, m_out, n_out, deg)


def _profile_values(node: Any, rho: np.ndarray, name: str) -> np.ndarray | None:
    """A DESC profile object evaluated at ``rho``, or ``None`` if it is unset.

    ``PowerSeriesProfile`` is ``sum_k params[k] rho^l_k`` with the exponents
    in its basis's mode table; ``SplineProfile`` is its values at its knots,
    read back as a C2 cubic (DESC's own ``cubic2`` differs only in the end
    conditions, and both are resampled onto ``rho`` and refit here anyway).

    Args:
        node: the HDF5 group of the profile, or the dataset holding the
            string ``None``.
        rho: radii to evaluate at.
        name: profile name, for error messages.

    Returns:
        The values at ``rho``, or ``None`` when the file stores no profile.

    Raises:
        NotImplementedError: for a DESC profile class not handled here.
    """
    import h5py  # noqa: PLC0415  (h5py is heavy and only needed for DESC files)
    if isinstance(node, h5py.Dataset):             # the string "None"
        return None
    cls = node["__class__"][()].decode().rsplit(".", 1)[-1]
    params = np.asarray(node["_params"][()], dtype=np.float64)
    if cls == "PowerSeriesProfile":
        powers = np.asarray(node["_basis"]["_modes"][()])[:, 0].astype(int)
        return (rho[:, None] ** powers[None, :] @ params).astype(np.float64)
    if cls == "SplineProfile":
        from scipy.interpolate import CubicSpline  # noqa: PLC0415
        knots = np.asarray(node["_knots"][()], dtype=np.float64)
        return CubicSpline(knots, params)(rho)
    raise NotImplementedError(f"{name}: DESC profile class {cls!r} is not read by MRX; "
                              "re-save the equilibrium with a power series or spline profile")


def _iota_from_desc(path: str, rho: np.ndarray) -> np.ndarray:
    """``iota(rho)`` of a current-constrained file, from DESC itself.

    A current-constrained equilibrium does not store its rotational
    transform: it is an output of the solve. There is nothing in the file to
    parse, so DESC is asked to recompute it. The axis value is extrapolated
    linearly in ``s = rho^2`` from the two innermost interior nodes, the
    treatment :func:`mrx.vmec._lambda_nodes` uses for the same reason (iota
    is even in ``rho``, and DESC's grids exclude the singular axis).

    Args:
        path: the DESC output file.
        rho: radii to evaluate at, with ``rho[0] = 0``.

    Returns:
        ``iota`` at ``rho``, per full toroidal turn.

    Raises:
        ImportError: if DESC is not installed.
    """
    try:
        from desc.grid import LinearGrid  # noqa: PLC0415  (optional dependency)
        from desc.io import load  # noqa: PLC0415
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            f"{path} is current-constrained: it stores no iota profile, so MRX "
            "cannot read it without DESC. Either install DESC (pip install "
            "'mrx[DESC]') or re-save the equilibrium with an iota constraint, "
            "e.g. VMECIO.load(wout, profile='iota')."
        ) from exc

    # DESC's loader rebuilds its own objects and needs the class tags a real
    # eq.save() writes; a file assembled by hand (test/synthetic_desc.py) is
    # readable by us and not by it. Say which of the two failed.
    try:
        eq = load(path)
        eq = eq[-1] if hasattr(eq, "__len__") else eq
        m_grid, n_grid, nfp_eq, sym = eq.M_grid, eq.N_grid, eq.NFP, eq.sym
    except Exception as exc:
        raise ValueError(
            f"{path} is current-constrained, so its iota has to come from DESC, "
            f"but DESC could not load the file ({type(exc).__name__}: {exc}). "
            "Only a file written by DESC itself can take this path."
        ) from exc

    grid = LinearGrid(rho=rho[1:], M=m_grid, N=n_grid, NFP=nfp_eq, sym=sym)
    interior = np.asarray(grid.compress(eq.compute("iota", grid=grid)["iota"]))
    s = rho[1:] ** 2
    axis = interior[0] + (interior[1] - interior[0]) * (0.0 - s[0]) / (s[1] - s[0])
    return np.concatenate([[axis], interior])


def _open(path: str) -> Any:
    """The DESC file opened for reading, with a readable error for a non-HDF5
    input (``h5py`` otherwise raises a bare ``OSError`` naming no cause).

    Args:
        path: the file to open.

    Returns:
        The open :class:`h5py.File`.

    Raises:
        ValueError: if the file does not carry the HDF5 signature.
    """
    import h5py  # noqa: PLC0415
    with open(path, "rb") as fh:
        if fh.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError(f"{path}: not an HDF5 file, so not a DESC output")
    return h5py.File(path, "r")


def _equilibrium(fh: Any, path: str) -> Any:
    """The converged equilibrium in an open DESC file, whichever way it is stored.

    DESC writes two layouts and both are in the wild. An
    ``EquilibriaFamily`` -- what the shipped examples and any continuation
    run are -- nests its members under ``/_equilibria``, and only the LAST
    is converged; ``eq.save(...)`` of a single ``Equilibrium``, which is
    what ``VMECIO.load`` hands back, writes that equilibrium flat at the
    top level with no family group at all.

    Args:
        fh: the open :class:`h5py.File`.
        path: the file path, for error messages.

    Returns:
        The HDF5 group of the converged equilibrium.

    Raises:
        ValueError: if the file is neither layout.
    """
    if "_equilibria" in fh:
        family = fh["_equilibria"]
        steps = sorted((k for k in family if k.isdigit()), key=int)
        if not steps:
            raise ValueError(f"{path}: '_equilibria' holds no equilibrium")
        return family[steps[-1]]
    if "_R_lmn" in fh:                         # a single saved Equilibrium
        return fh
    raise ValueError(f"{path}: neither an '_equilibria' family nor a single saved "
                     f"equilibrium ('_R_lmn'); this is not a DESC equilibrium "
                     f"output (top-level keys: {sorted(fh)})")


def read_desc(path: str, n_rho: int | None = None, deg: int = DEG) -> dict[str, Any]:
    """Parse a DESC output file into the dict shape of :func:`mrx.gvec.read_state`.

    Args:
        path: a DESC ``.h5`` output file. The LAST equilibrium of its
            continuation family is read.
        n_rho: radial samples of the spline refit; ``None`` scales with the
            file's radial resolution ``L`` (:data:`NODES_PER_L` per degree,
            at least :data:`MIN_NODES`).
        deg: spline degree of the refit blocks.

    Returns:
        ``nfp``, ``deg``, the DESC resolutions ``L``, ``M``, ``N``, the
        toroidal flux ``Psi``, the blocks ``X1``, ``X2``, ``LA`` (each
        carrying its knot vector ``T``) and ``profiles`` (``rho``, ``phi``,
        ``iota``, ``pressure`` at the sample nodes, flux in GVEC units).

    Raises:
        ValueError: for a file with no equilibrium family, or one that is
            not HDF5 at all.
        NotImplementedError: for a non-stellarator-symmetric file.
        ImportError: for a current-constrained file when DESC is absent.
    """
    with _open(path) as fh:
        eq = _equilibrium(fh, path)
        if not bool(eq["_sym"][()]):
            raise NotImplementedError(
                f"{path}: _sym = False (non-stellarator-symmetric) needs the "
                "missing-parity partners of R, Z and lambda")
        nfp = int(eq["_NFP"][()])
        L, M, N = (int(eq[f"_{k}"][()]) for k in "LMN")
        psi = float(eq["_Psi"][()])
        rho = _nodes(n_rho if n_rho is not None
                     else int(np.clip(NODES_PER_L * L + 1, MIN_NODES, MAX_NODES)))
        blocks = {}
        for block_name, prefix in (("X1", "R"), ("X2", "Z"), ("LA", "L")):
            modes = np.asarray(eq[f"_{prefix}_basis"]["_modes"][()])
            coef = np.asarray(eq[f"_{prefix}_lmn"][()], dtype=np.float64)
            blocks[block_name] = _convert_block(modes, coef, nfp, rho, prefix, deg)
        pressure = _profile_values(eq["_pressure"], rho, "pressure")
        iota = _profile_values(eq["_iota"], rho, "iota")

    if iota is None:
        iota = _iota_from_desc(path, rho)
    if pressure is None:
        pressure = np.zeros_like(rho)
    return dict(
        nfp=nfp, deg=deg, L=L, M=M, N=N, Psi=psi, n_rho=len(rho),
        mnmax=blocks["X1"]["coef"].shape[0], **blocks,
        profiles=dict(rho=rho, phi=psi * rho ** 2 / TWO_PI,
                      iota=iota, pressure=pressure))


def flip_poloidal_angle(st: dict[str, Any]) -> dict[str, Any]:
    """The same equilibrium re-expressed in ``theta -> -theta``.

    DESC insists on a positive coordinate Jacobian: ``VMECIO.load`` runs
    ``ensure_positive_jacobian``, which for a LEFT-handed wout -- most of
    them -- silently flips the sign of theta, negating every ``m < 0`` mode
    of ``R`` and ``Z``, every ``m >= 0`` mode of lambda, and ``iota``. The
    resulting DESC equilibrium is the same torus with the opposite poloidal
    orientation, so comparing it with the wout it came from at equal
    ``theta`` compares two different points. This undoes that.

    In MRX's single-angle blocks the substitution is exact and local:
    ``trig(m theta - n zeta)`` at ``-theta`` is ``trig(m theta + n zeta)``,
    i.e. the mode ``(m, -n)``, picking up a minus sign in the sine blocks.
    With lambda additionally negated -- ``theta* = theta + lambda`` must
    flip with theta -- the three blocks come out as

    * ``X1`` (cosine): ``n -> -n``
    * ``X2`` (sine): ``n -> -n``, coefficients negated
    * ``LA`` (sine): ``n -> -n``, the two sign flips cancelling

    and ``iota``, a ratio of the two angles' fluxes, negates. ``phi`` and
    ``pressure`` are untouched: neither knows about theta.

    Nothing in MRX needs this to read a DESC file -- ``build_gvec_map``
    measures the handedness that makes ``det DF > 0`` either way. It is for
    holding a DESC state against a VMEC one, where the labels must agree.

    Args:
        st: a state dict from :func:`read_desc`.

    Returns:
        A new state dict; the input is not modified.
    """
    out = dict(st)
    for name, flip_sign in (("X1", False), ("X2", True), ("LA", False)):
        blk = dict(st[name])
        order = np.lexsort((-blk["n"], blk["m"]))
        blk["m"], blk["n"] = blk["m"][order], -blk["n"][order]
        blk["coef"] = blk["coef"][order] * (-1.0 if flip_sign else 1.0)
        out[name] = blk
    out["profiles"] = dict(st["profiles"], iota=-np.asarray(st["profiles"]["iota"]))
    return out


def match_orientation(st: dict[str, Any], ref: dict[str, Any],
                      n_probe: int = 16) -> tuple[dict[str, Any], int]:
    """``st`` turned to ``ref``'s poloidal orientation, measured not assumed.

    Which way ``VMECIO.load`` flipped depends on the wout's handedness, so
    the orientation is decided rather than predicted: ``R`` AND ``Z`` are
    evaluated both ways against the reference and the closer wins. Both are
    needed. ``R`` alone cannot see the flip on an up-down-symmetric
    cross-section, where it is even in theta -- the circular torus of
    ``test/synthetic_desc.py`` is exactly that case -- while ``Z``, a sine
    block, changes sign there. No stellarator-symmetric shape hides from
    the pair, since that would need ``Z`` identically zero.

    Args:
        st: the DESC state to orient.
        ref: the reference state, typically :func:`mrx.vmec.read_wout`.
        n_probe: samples per angle of the probe grid.

    Returns:
        ``(state, sign)``: the state in ``ref``'s orientation (``st``
        itself when they already agree) and ``+1`` or ``-1``.
    """
    from mrx.gvec import evaluate  # noqa: PLC0415  (imports this module)

    rho = np.linspace(0.2, 1.0, 5)
    theta = 2.0 * np.pi * (np.arange(n_probe) + 0.5) / n_probe
    zeta = 2.0 * np.pi * (np.arange(n_probe) + 0.5) / (n_probe * ref["nfp"])

    def distance(state):
        total = 0.0
        for name in ("X1", "X2"):
            want = evaluate(ref[name], rho, theta, zeta)
            got = evaluate(state[name], rho, theta, zeta)
            total += float(np.abs(got - want).max()) / max(float(np.abs(want).max()), 1e-30)
        return total

    flipped = flip_poloidal_angle(st)
    return (st, 1) if distance(st) <= distance(flipped) else (flipped, -1)


def read_nfp(path: str) -> int:
    """Just ``nfp``, without fitting anything.

    Args:
        path: a DESC ``.h5`` output file.

    Returns:
        The number of field periods.
    """
    with _open(path) as fh:
        return int(_equilibrium(fh, path)["_NFP"][()])


def profile_spline(st: dict[str, Any], name: str) -> BSpline:
    """The radial spline through a DESC profile's sampled values, in ``rho``.

    The mirror of :func:`mrx.vmec.profile_spline`: a profile is a function of
    ``s = rho^2``, i.e. even in ``rho``, so it is fit with the ``m = 0``
    parity conditions of :func:`mrx.vmec._fit_block` on its own knots.
    ``phi`` is exactly ``Psi rho^2 / 2 pi``, which the fit reproduces and
    whose derivative therefore vanishes at the axis.

    Args:
        st: the state dict of :func:`read_desc`.
        name: ``"phi"``, ``"iota"`` or ``"pressure"``.

    Returns:
        The profile as a :class:`scipy.interpolate.BSpline` in ``rho``.
    """
    prof, deg = st["profiles"], st["deg"]
    values = np.asarray(prof[name], dtype=np.float64)[:, None]
    blk = _fit_block(prof["rho"], values, 2, np.zeros(1, dtype=int),
                     np.zeros(1, dtype=int), deg)
    return BSpline(blk["T"], blk["coef"][0], deg)
