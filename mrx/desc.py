"""DESC equilibria: the ``.h5`` output read in closed form.

Turns a DESC Fourier-Zernike series into the same block dict
:func:`mrx.gvec.read_state` produces, so ``build_gvec_map``,
``load_clebsch`` and ``clebsch_potential_form`` apply verbatim. Only
``h5py``, ``numpy`` and ``scipy`` are imported: a file is read without
DESC installed.

Three conversions happen here and nowhere else: the product-to-sum
angular basis (each DESC mode becomes the pair ``(m, ±n·nfp)`` at half
weight), a Chebyshev-Lobatto radial sample refit by
:func:`mrx.vmec._fit_block`, and the flux-unit factor ``1 / 2π``. A
current-constrained file stores no iota and falls back to DESC itself
(:func:`_iota_from_desc`).

Guards: non-stellarator-symmetric files, mixed-parity modes, an unknown
profile class, an empty ``_equilibria`` family, and a midpoint refit
above :data:`REFIT_TOL`. The orientation pair
:func:`flip_poloidal_angle` / :func:`match_orientation` lives on the
shared block dict in :mod:`mrx.gvec` and is re-exported here.

The full derivation, the lambda-half-mesh finding, and the reader-parity
list are in ``docs/source/concepts/external_interfaces.md``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import eval_jacobi

from mrx.gvec import flip_poloidal_angle, match_orientation  # noqa: F401  (re-exported)
from mrx.vmec import _fit_block, profile_spline as _vmec_profile_spline

TWO_PI = 2.0 * np.pi

#: Spline degree of the refit blocks, matching :mod:`mrx.vmec`.
DEG = 3

#: Smallest radial sample count, and the multiple of DESC's radial
#: resolution ``L`` used above it. A degree-``L`` Zernike polynomial needs
#: the cubic interpolant to resolve its ``L/2`` interior extrema; four
#: nodes per extremum is the measured knee (``test_desc.py``).
MIN_NODES, NODES_PER_L = 65, 4
MAX_NODES = 257

#: Ceiling on the midpoint refit error of the R and Z blocks. The spline is
#: interpolatory at the sample nodes, so only the midpoints see an
#: under-resolved ``n_rho``; this refuses that rather than returning a
#: smooth wrong answer. A decade above the worst measured default-read
#: error on the tracked fixtures.
REFIT_TOL = 1e-4


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


def _accumulate_modes(modes: np.ndarray, coef: np.ndarray, nfp: int,
                      rho: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """DESC modes accumulated onto single-angle ``(m, n)`` columns at ``rho``.

    This is the product-to-sum half of :func:`_convert_block`, shared with
    :func:`_block_refit_error` so the midpoint check uses the same
    identities as the fit.

    Args:
        modes: ``(K, 3)`` table of ``(l, m, n)``.
        coef: ``(K,)`` coefficients.
        nfp: field periods, to turn ``n`` into the full-turn index.
        rho: radial sample nodes.

    Returns:
        ``{(m, n): values}`` at ``rho``, ``values`` of shape ``(len(rho),)``.
    """
    ell, m, n = (modes[:, i].astype(int) for i in range(3))
    radial = _zernike_radial(rho[:, None], ell[None, :], m[None, :]) * coef[None, :]
    w_plus, w_minus = _split_weights(m, n)
    m_abs, n_full = np.abs(m), np.abs(n) * nfp
    columns: dict[tuple[int, int], np.ndarray] = {}
    for key_n, weight in ((n_full, w_plus), (-n_full, w_minus)):
        for i in range(len(ell)):
            key = (int(m_abs[i]), int(key_n[i]))
            col = columns.get(key)
            if col is None:
                col = columns[key] = np.zeros(len(rho))
            col += weight[i] * radial[:, i]
    return columns


def _block_refit_error(modes: np.ndarray, coef: np.ndarray, nfp: int,
                       rho: np.ndarray, blk: dict[str, Any]) -> float:
    """Relative sup error of a fitted block at the node midpoints.

    The spline is interpolatory at ``rho``, so the nodes themselves cannot
    see an under-resolved ``n_rho``. The midpoints can.

    Args:
        modes: ``(K, 3)`` table of ``(l, m, n)``.
        coef: ``(K,)`` coefficients.
        nfp: field periods.
        rho: the nodes the block was fit at.
        blk: the fitted block dict.

    Returns:
        ``max |spline - exact| / max |exact|`` on the midpoints, or 0
        when the field is identically zero.
    """
    mid = 0.5 * (rho[:-1] + rho[1:])
    columns = _accumulate_modes(modes, coef, nfp, mid)
    design = BSpline.design_matrix(mid, blk["T"], blk["deg"]).toarray()
    got = design @ blk["coef"].T
    want = np.stack(
        [columns.get((int(mm), int(nn)), np.zeros(len(mid)))
         for mm, nn in zip(blk["m"], blk["n"])],
        axis=1)
    scale = max(float(np.abs(want).max()), 1e-30)
    return float(np.abs(got - want).max()) / scale


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
    m, n = modes[:, 1].astype(int), modes[:, 2].astype(int)
    cosine = _is_cosine(m, n)
    live = np.abs(coef) > 0.0
    if live.any() and not (cosine[live].all() or (~cosine[live]).all()):
        raise ValueError(f"{name}: mixes cos and sin modes of (m theta - n zeta); "
                         "only stellarator-symmetric DESC files are supported")
    sin_cos = 2 if (not live.any() or cosine[live][0]) else 1
    columns = _accumulate_modes(modes, coef, nfp, rho)
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
        carrying its knot vector ``T``), ``profiles`` (``rho``, ``phi``,
        ``iota``, ``pressure`` at the sample nodes, flux in GVEC units)
        and ``refit_error``, the relative midpoint error of the R and Z
        spline refit.

    Raises:
        ValueError: for a file with no equilibrium family, one that is
            not HDF5 at all, or whose R/Z spline refit exceeds
            :data:`REFIT_TOL`.
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
        blocks: dict[str, Any] = {}
        raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for block_name, prefix in (("X1", "R"), ("X2", "Z"), ("LA", "L")):
            modes = np.asarray(eq[f"_{prefix}_basis"]["_modes"][()])
            coef = np.asarray(eq[f"_{prefix}_lmn"][()], dtype=np.float64)
            blocks[block_name] = _convert_block(modes, coef, nfp, rho, prefix, deg)
            raw[block_name] = (modes, coef)
        pressure = _profile_values(eq["_pressure"], rho, "pressure")
        iota = _profile_values(eq["_iota"], rho, "iota")

    if iota is None:
        iota = _iota_from_desc(path, rho)
    if pressure is None:
        pressure = np.zeros_like(rho)
    refit_error = max(
        _block_refit_error(*raw["X1"], nfp, rho, blocks["X1"]),
        _block_refit_error(*raw["X2"], nfp, rho, blocks["X2"]),
    )
    if refit_error > REFIT_TOL:
        raise ValueError(
            f"{path}: R/Z spline refit error {refit_error:.3e} > {REFIT_TOL} "
            f"at n_rho={len(rho)}; raise n_rho")
    return dict(
        nfp=nfp, deg=deg, L=L, M=M, N=N, Psi=psi, n_rho=len(rho),
        mnmax=blocks["X1"]["coef"].shape[0], refit_error=refit_error, **blocks,
        profiles=dict(rho=rho, phi=psi * rho ** 2 / TWO_PI,
                      iota=iota, pressure=pressure))


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
    return _vmec_profile_spline(st, name)
