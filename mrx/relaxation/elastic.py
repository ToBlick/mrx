"""The elastic regularisation of the relaxation, after Burby, Cho & Ruth, *Elasticized magnetohydrostatics*
(arXiv:2610.36241), and Ruth, Burby, Sengupta & Brown, *Statistical equilibrium model for stellarators*
(arXiv:2604.08878).

Both papers write the equilibrium as a minimiser of an energy of the back-to-labels map ``G``, with the field
the pull-back ``B = adj(DG) B_0(G)`` of a reference field ``B_0``. Averaging the magnetic energy over small-scale
fluctuations of variance ``Sigma_0`` adds the elastic energy

    W_el = 1/2 int tr(adj(DG) Sigma_0(G) adj(DG)^T) dx = 1/2 sum_k int |S_k|^2 dx,

with ``Sigma_0 = sum_k s_k s_k^T`` and ``S_k = adj(DG) s_k(G)``. This is the linear term of the second paper's
phenomenological model, the one its numerical examples suggest is enough. Each ``S_k`` is the pull-back of a fixed
2-form ``s_k`` of label space, a flux density frozen into the flow. Its transport is the Lie derivative of a 2-form,
``dS_k/dt = curl(u x S_k) - u div S_k``, so ``div S_k`` moves as a density. The relaxation never forms ``G``. It
carries the ``S_k`` as further 2-forms, the *elastic fields*, moves them with the velocity that moves ``B`` and
descends the total energy ``||B||^2 / 2 + sum_k ||S_k||^2 / 2``. The force gains
``sum_k (curl S_k x S_k + S_k div S_k)`` and the Hessian the terms of every ``S_k``
(:func:`mrx.relaxation.newton.second_variation`). The papers' ``lambda^2`` is ``||S_k||^2 / ||B||^2``.

The magnetic energy alone does not see flows along the field lines of ``B``, and it allows the current sheets of
ideal MHD on rational surfaces. The elastic energy also charges a flow for the strain it puts on the labels, which
makes the second variation elliptic and spreads each current sheet over a layer, of width ``~ lambda`` in the
estimate of arXiv:2604.08878 (with ``lambda`` from 0.01 to 0.1 in its scans).

The fields come in kinds (:class:`Kind`), which differ in parity, in the wall condition and in whether they are
closed. The state of the relaxation holds them in one dict, the fields of each kind as an array of shape
``(K, n_2)`` under the name of the kind. Each step of :mod:`mrx.relaxation.loop` calls :func:`lorentz` for their
currents and force and :func:`transport` for their increments, and the Newton step takes them as frozen-in
fields from :func:`frozen`. The kinds present and their number of fields are compiled in, the fields are not.

The reference to use is :func:`perpendicular_fields`. The others remain for comparison. :data:`REFERENCES` names
them as ``--elastic.reference`` does.

- :func:`label_fields` gives the fields of that paper, ``d psi ^ d theta`` and ``d psi ^ d zeta`` of the start
  flux coordinates, with ``psi = r^2 / 2`` (:data:`LABELS`). They are odd like ``B``, tangent to the wall and
  closed. Their divergence stays exactly zero and their transport is ``curl(u x S_k)`` alone.
- :func:`normal_field` gives the normal part of ``Sigma_0``, ``f dtheta ^ dzeta``, which couples neighbouring flux
  surfaces (:data:`NORMAL`). It is even like the velocity, tangent to the wall and not closed, so it moves with the
  full transport.
- :func:`isotropic_fields` gives the isotropic reference ``Sigma_0 = a^2 I``, the cylindrical frame
  ``a e_R``, ``a e_phi``, ``a e_Z`` (:data:`ISOTROPIC_ODD`, :data:`ISOTROPIC_EVEN`). A constant ``Sigma_0`` exerts no
  force, ``div Sigma_0 - grad(tr Sigma_0) / 2 = 0``, so only the strain the flow creates acts on ``B``. The label
  fields instead exert a force at zero strain. The frame fields cross the wall, so they live in the 2-forms without
  the wall condition. ``u . n = 0`` keeps them consistent with the wall, and their weak currents carry the traction
  on it.
- :func:`perpendicular_fields` gives the fluctuations across the start field ``B_0``, ``Sigma_0 = lam^2 (|B_0|^2 I -
  B_0 B_0^T)`` (:data:`PERPENDICULAR_ODD`, :data:`PERPENDICULAR_EVEN`). At zero strain their force is
  ``-lam^2 (B_0 . grad) B_0``, which is ``-lam^2 J x B_0`` up to a gradient. An equilibrium of ``B_0`` stays one of
  the full force. Their stress at zero strain is ``-lam^2 B_0 B_0^T``, which puts no traction on a wall with
  ``B_0 . n = 0``. These fields cross the wall and are not closed. They are not tapered there and live in the 2-forms
  without the wall condition, so at zero strain the discrete force differs from this one by an error that falls with the
  mesh.

The elastic fields act in the Newton solve (:mod:`mrx.relaxation.newton`) as a regulariser through the force and the
operator.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.geometry import map_jacobian_at
from mrx.relaxation.newton import divergence_at_quadrature

@dataclass(frozen=True)
class Kind:
    """A kind of elastic fields: the ``name`` its fields go under in the state, the ``parity`` of the fields under the
    stellarator symmetry (``-1`` odd like ``B``, ``+1`` even like the velocity), whether they satisfy the wall
    condition ``S . n = 0`` (``wall``) and whether they are closed, ``div S = 0`` (``closed``)."""
    name: str
    parity: int
    wall: bool
    closed: bool

    def view(self, seq: DeRhamSequence) -> DeRhamSequence:
        """The space of the fields: the parity view of the full sequence ``seq``, without the wall condition for
        fields that cross the wall. Their electric fields ``u x S`` and currents live in the 1-forms of that view."""
        view = seq.parity_view(self.parity)
        return view if self.wall else view.free


#: The label fields of :func:`label_fields`.
LABELS = Kind("labels", -1, True, True)
#: The normal field of :func:`normal_field`.
NORMAL = Kind("normal", 1, True, False)
#: The odd fields ``a e_phi`` and ``a e_Z`` of :func:`isotropic_fields`.
ISOTROPIC_ODD = Kind("isotropic_odd", -1, False, True)
#: The even field ``a e_R`` of :func:`isotropic_fields`, with ``div e_R = 1 / R``.
ISOTROPIC_EVEN = Kind("isotropic_even", 1, False, False)
#: The odd fields of :func:`perpendicular_fields`, from ``e_phi`` and ``e_Z``.
PERPENDICULAR_ODD = Kind("perpendicular_odd", -1, False, False)
#: The even field of :func:`perpendicular_fields`, from ``e_R``.
PERPENDICULAR_EVEN = Kind("perpendicular_even", 1, False, False)
#: All kinds, by name.
KINDS = {kind.name: kind for kind in (LABELS, NORMAL, ISOTROPIC_ODD, ISOTROPIC_EVEN, PERPENDICULAR_ODD,
                                      PERPENDICULAR_EVEN)}


def _label_potential(i):
    """The 1-form ``psi d x^i`` (logical components) with ``psi = r^2 / 2``, a function of a logical point. Its curl is
    ``d psi ^ d theta`` for ``i = 1`` and ``d psi ^ d zeta`` for ``i = 2``."""
    def A(x):
        return jnp.zeros(3, dtype=x.dtype).at[i].set(0.5 * x[0] ** 2)
    return A


def _normal_form(x):
    """The 2-form ``s (1 - s) dtheta ^ dzeta`` (logical components) with ``s = r^2``, a function of a logical point."""
    s = x[0] ** 2
    return jnp.zeros(3, dtype=x.dtype).at[0].set(s * (1.0 - s))


def label_fields(seq: DeRhamSequence, lam: float, B: jnp.ndarray) -> dict:
    """Return the elastic fields of the logical flux coordinates, two odd 2-forms of :data:`LABELS`, in a dict by
    kind name.

    The fields are the label 2-forms ``d psi ^ d theta`` and ``d psi ^ d zeta`` with ``psi = r^2 / 2``. Away from
    the axis, ``r = const`` are the flux surfaces of the start field (see :mod:`mrx.relaxation.initial_conditions`).
    So these are ``e_T = grad v x grad theta`` and ``-e_P = grad v x grad zeta`` of arXiv:2604.08878 with
    ``v = psi``. ``psi`` rather than ``r`` keeps them regular at the axis, where ``d r ^ d theta`` grows like
    ``1 / r``. Their field lines lie in the flux surfaces, so ``S . n = 0`` holds on the wall. Each field is the
    discrete curl of its potential, so ``div S = 0`` holds exactly. Each is scaled to ``||S_k|| = lam ||B||``.
    """
    odd = seq.odd
    free = odd.free
    fields = []
    for i in (1, 2):
        S = odd.restrict(free.G[1] @ free.interpolate(_label_potential(i), 1), 2)
        fields.append(S / odd.l2_norm(S, 2))
    return {LABELS.name: lam * odd.l2_norm(B, 2) * jnp.stack(fields)}


def normal_field(seq: DeRhamSequence, lam_perp: float, B: jnp.ndarray) -> dict:
    """Return the normal elastic field ``f dtheta ^ dzeta`` of the logical flux coordinates, one even 2-form of
    :data:`NORMAL`, in a dict by kind name.

    Its field lines cross the flux surfaces ``r = const`` of the start field, so it charges flows that compress
    neighbouring surfaces or shear them against each other. The weight is ``f = s (1 - s)`` with ``s = r^2``.
    ``dtheta ^ dzeta`` grows like ``1 / r`` at the axis and is not square integrable there. With ``f ~ r^2`` the
    field is the smooth ``(x, y, 0) / R`` near the axis, while ``f ~ r`` would leave a field whose direction jumps
    at the axis. ``f(1) = 0`` gives ``S . n = 0`` on the wall. ``s (1 - s)`` is the lowest polynomial in ``s`` with
    both. The field is the interpolant of that form, so its divergence is the interpolant of
    ``f' dr ^ dtheta ^ dzeta``, which is not zero. It is scaled to ``||S|| = lam_perp ||B||``.
    """
    even = seq.even
    S = even.restrict(even.free.interpolate(_normal_form, 2), 2)
    return {NORMAL.name: (lam_perp * seq.odd.l2_norm(B, 2) / even.l2_norm(S, 2) * S)[None]}


def _frame(seq: DeRhamSequence, x: jnp.ndarray):
    """``(DPhi, p, R)`` at the logical point ``x``: the Jacobian of the map, the physical point and its major radius."""
    p = seq.map(x)
    return jax.jacfwd(seq.map)(x), p, jnp.hypot(p[0], p[1])


def _adjugate(D: jnp.ndarray) -> jnp.ndarray:
    """``det(D) D^-1`` of a 3 x 3 matrix, whose rows are the cross products of the columns of ``D``."""
    return jnp.stack([jnp.cross(D[:, 1], D[:, 2]), jnp.cross(D[:, 2], D[:, 0]), jnp.cross(D[:, 0], D[:, 1])])


def isotropic_fields(seq: DeRhamSequence, lam: float, B: jnp.ndarray) -> dict:
    """Return the isotropic elastic reference in a dict by kind name: the frame ``a e_phi``, ``a e_Z`` (two odd
    2-forms of :data:`ISOTROPIC_ODD`) and ``a e_R`` (one even 2-form of :data:`ISOTROPIC_EVEN`), on the 2-forms without
    the wall condition.

    The frame is the cylindrical one of the map, so ``sum_k S_k S_k^T = a^2 I`` everywhere. The Cartesian frame
    ``e_x, e_y, e_z`` would do the same, but ``e_x`` and ``e_y`` do not repeat from one field period to the next. The
    frame rotates by ``pi`` about the x axis like the stellarator symmetry, ``e_R`` stays and ``e_phi`` and ``e_Z``
    change sign, which makes ``e_R`` even and the other two odd. ``e_phi`` and ``e_Z`` are the discrete curls of
    their potentials ``-R e_Z`` and ``R e_phi / 2``, so their divergence is exactly zero. ``e_R`` is the interpolant
    of the 2-form, and its divergence is the interpolant of ``div e_R = 1 / R``. At zero strain the force of each
    field inside the domain is a gradient, ``e_R / R``, ``-e_R / R`` and zero. On the wall each weak current carries
    the traction ``S x n``, and the tractions of the three fields add up to a normal force, which no flow with
    ``u . n = 0`` feels. Each field is scaled to ``||S_k|| = lam ||B||``.
    """
    odd, even = ISOTROPIC_ODD.view(seq), ISOTROPIC_EVEN.view(seq)

    def potential(phi):
        """The logical components ``DPhi^T A`` of the potential ``A`` of ``e_phi`` (``phi``) or of ``e_Z``."""
        def A(x):
            D, p, R = _frame(seq, x)
            zero = jnp.zeros((), dtype=x.dtype)
            return D.T @ (jnp.stack([zero, zero, -R]) if phi else 0.5 * jnp.stack([-p[1], p[0], zero]))
        return A

    def radial(x):
        """The logical components ``J DPhi^-1 e_R`` of the 2-form ``e_R``, formed with the adjugate, which stays
        finite at the axis where ``DPhi`` is singular."""
        D, p, R = _frame(seq, x)
        e_R = jnp.stack([p[0] / R, p[1] / R, jnp.zeros((), dtype=x.dtype)])
        return _adjugate(D) @ e_R
    scale = lam * seq.odd.l2_norm(B, 2)
    S_odd = jnp.stack([odd.G[1] @ odd.interpolate(potential(phi), 1) for phi in (True, False)])
    S_even = even.interpolate(radial, 2)[None]
    return {ISOTROPIC_ODD.name: scale * S_odd / jnp.stack([odd.l2_norm(S_k, 2) for S_k in S_odd])[:, None],
            ISOTROPIC_EVEN.name: scale * S_even / even.l2_norm(S_even[0], 2)}


def perpendicular_fields(seq: DeRhamSequence, lam: float, B: jnp.ndarray) -> dict:
    """Return the elastic fields across the field ``B``, ``S_k = lam |B| (e_k - (e_k . b) b)`` with ``b = B / |B|``
    and ``e_k`` the cylindrical frame of :func:`isotropic_fields`, in a dict by kind name: the odd fields of
    ``e_phi`` and ``e_Z`` (:data:`PERPENDICULAR_ODD`) and the even one of ``e_R`` (:data:`PERPENDICULAR_EVEN`), on the
    2-forms without the wall condition.

    Then ``sum_k S_k S_k^T = lam^2 (|B|^2 I - B B^T)``, which is smooth wherever ``B`` is. The amplitude follows
    ``|B|`` at every point. A constant one would leave a force along the curvature of the field lines. The
    fluctuation level in each direction across ``B`` is ``lam^2 |B|^2``, the level of the isotropic reference of the
    same ``lam`` up to the volume average. Their total energy is ``lam^2 ||B||^2``, that of the two label fields.
    ``e_k . b`` flips sign with ``B`` under the stellarator symmetry, so ``(e_k . b) b`` has the parity of ``e_k``.
    The fields are the L2 projections of these vector fields. They are not closed.
    """
    odd, even = PERPENDICULAR_ODD.view(seq), PERPENDICULAR_EVEN.view(seq)
    D = map_jacobian_at(seq.map, seq.quad.x)
    p = jax.lax.map(seq.map, seq.quad.x, batch_size=mrx.MAP_BATCH_SIZE_INNER)
    R = jnp.hypot(p[:, 0], p[:, 1])
    zero = jnp.zeros_like(R)
    J = seq.jacobian_j[:, None]
    B_jk = seq.odd.evaluate_at_quadrature(B, 2)
    B_phys = jnp.einsum('qij,qj->qi', D, B_jk) / J
    B_abs = jnp.linalg.norm(B_phys, axis=1, keepdims=True)
    frame = {"R": jnp.stack([p[:, 0] / R, p[:, 1] / R, zero], axis=1),
             "phi": jnp.stack([-p[:, 1] / R, p[:, 0] / R, zero], axis=1),
             "Z": jnp.stack([zero, zero, zero + 1.0], axis=1)}

    def project(view, e):
        """The 2-form ``lam |B| (e - (e . b) b)`` on ``view``: the L2 projection of its logical components."""
        X = lam * (B_abs * e - jnp.sum(e * B_phys, axis=1, keepdims=True) * B_phys / B_abs)
        X_jk = J * jnp.linalg.solve(D, X[..., None])[..., 0]
        return view.M[2].solve(view.vector_load_values(X_jk, 2, 2))
    return {PERPENDICULAR_ODD.name: jnp.stack([project(odd, frame[c]) for c in ("phi", "Z")]),
            PERPENDICULAR_EVEN.name: project(even, frame["R"])[None]}


#: The builders of the references of ``--elastic.reference``, each ``(seq, lam, B) -> fields``.
REFERENCES = {"labels": label_fields, "isotropic": isotropic_fields, "perpendicular": perpendicular_fields}


def frozen(seq: DeRhamSequence, S: dict) -> list:
    """The fields of a dict ``S`` by kind name one by one, as triples ``(view, S_k, closed)`` of the space of their
    kind, the field and whether it is closed. These are the frozen-in fields of
    :func:`mrx.relaxation.newton.second_variation`."""
    return [(KINDS[name].view(seq), S_k, KINDS[name].closed) for name, S_kind in S.items() for S_k in S_kind]


def lorentz(seq: DeRhamSequence, S: dict, guess: dict | None) -> tuple[dict, jnp.ndarray, dict]:
    """Return ``(J, load, rho)`` of the elastic fields ``S``. ``J`` holds their weak currents ``M_1 J_k = D_1^T S_k``
    on the spaces of their kind, shape ``(K, n_1)`` per kind, the solves warm-started from ``guess`` (a dict like
    ``J``, or ``None``). ``load`` is their force ``sum_k (curl S_k x S_k + S_k div S_k)`` as a dual 2-form of the even
    view, zero without fields. ``rho`` holds the values of ``div S_k`` at the quadrature points for the kinds that are
    not closed, shape ``(K, n_q, 1)``, for :func:`transport`."""
    even = seq.even
    J, rho, load = {}, {}, 0.0
    for name, S_kind in S.items():
        kind = KINDS[name]
        view = kind.view(seq)
        J_kind, rho_kind = [], []
        for k, S_k in enumerate(S_kind):
            J_k = view.M[1].solve(view.D[1].T @ S_k, guess=None if guess is None else guess[name][k])
            S_jk = view.evaluate_at_quadrature(S_k, 2)
            load = load + even.cross_product_load_values(view.evaluate_at_quadrature(J_k, 1), S_jk, 2, 1, 2)
            if not kind.closed:
                rho_kind.append(divergence_at_quadrature(view, S_k))
                load = load + even.vector_load_values(S_jk * rho_kind[-1], 2, 2)
            J_kind.append(J_k)
        J[name] = jnp.stack(J_kind)
        if rho_kind:
            rho[name] = jnp.stack(rho_kind)
    return J, load, rho


def transport(seq: DeRhamSequence, S: dict, u_jk: jnp.ndarray, rho: dict, guess_E: dict,
              guess_P: dict) -> tuple[dict, dict, dict]:
    """Return ``(dS, E, P)``, the increments ``dS_k = curl E_k - P_k`` of the elastic fields ``S`` along the velocity
    with the values ``u_jk`` at the quadrature points, in dicts like ``S``. ``E_k`` is the field ``u x S_k`` on the
    1-forms of the kind. ``P_k = M_2^-1 load(u div S_k)``, the drag of the density ``div S_k`` with the values ``rho``
    of :func:`lorentz`, is there for the kinds that are not closed only. The solves are warm-started from ``guess_E``
    and ``guess_P``. The curl is the topological one, so a closed field stays closed exactly."""
    dS, E, P = {}, {}, {}
    for name, S_kind in S.items():
        kind = KINDS[name]
        view = kind.view(seq)
        E_kind, P_kind, dS_kind = [], [], []
        for k, S_k in enumerate(S_kind):
            E_k = view.M[1].solve(view.cross_product_load_values(u_jk, view.evaluate_at_quadrature(S_k, 2), 1, 2, 2),
                                  guess=guess_E[name][k])
            dS_k = view.G[1] @ E_k
            if not kind.closed:
                # div S_k changes by the divergence of the flux P_k, so it moves as a density
                P_kind.append(view.M[2].solve(view.vector_load_values(u_jk * rho[name][k], 2, 2),
                                              guess=guess_P[name][k]))
                dS_k = dS_k - P_kind[-1]
            E_kind.append(E_k)
            dS_kind.append(dS_k)
        dS[name], E[name] = jnp.stack(dS_kind), jnp.stack(E_kind)
        if P_kind:
            P[name] = jnp.stack(P_kind)
    return dS, E, P

