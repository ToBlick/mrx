"""The Newton step of the relaxation: the Hessian of the magnetic energy and the Newton direction.

The relaxation moves the field ``B`` (a 2-form) with a divergence-free velocity ``u``, so that ``B_t = curl E``,
``E = Pi_1(u x h)`` with ``h = Pi_1 B_t`` (:func:`mrx.relaxation.physics.induction_field`). With ``Q = curl Pi_1(u x
h)`` and ``R = curl Pi_1(u x Pi_1 Q)`` the energy is ``||B + Q + R/2||_M^2 / 2`` to second order in ``u``. Its
gradient is ``-load(J x h)`` and its symmetric Hessian is

    (u, H v) = (Q_u, Q_v)_M + [(J, u x Pi_1 Q_v) + (J, v x Pi_1 Q_u)] / 2,

the bracket from ``(B, curl Y)_M = (J, Y)_{M_1}``.

At an equilibrium ``H`` is minus the ideal-MHD force operator at zero pressure.

* :func:`second_variation` returns the map ``u -> H u``, optionally with a penalty on flows along the field, or
  its Gauss-Newton part without the two terms in the current ``J``.
* :func:`newton_direction` computes one Gauss-Newton step. It writes ``u = curl a``, so that ``u`` is exactly
  divergence-free, and solves ``curl^T H curl a = curl^T M_2 F`` with the Gauss-Newton part of ``H`` for the
  potential ``a`` by Newton-MR, a MINRES solve that keeps and reorthogonalises its Lanczos vectors. The
  preconditioner is :func:`spline_preconditioner`. The two mass solves of every Hessian apply run to a
  tolerance inside the operator, which MINRES tolerates once they are accurate enough.
* :func:`target_direction` is :func:`newton_direction` with further unknowns: the amplitude of the harmonic flow
  ``h`` of the domain, the volumes of the flux shells, for a prescribed pressure (:mod:`mrx.relaxation.shells`), and
  the poloidal flux slips between the flux surfaces, for a prescribed current (:class:`mrx.relaxation.twist.Slips`).
  The curls do not reach ``h``, a divergence-free flow tangent to the wall, and only ``u = curl a + c h`` spans every
  incompressible flow. With stellarator symmetry ``h`` is odd and the force even, so the force has no harmonic part
  and the parity view of the flows holds no ``h``.

The module constants are the production defaults of the Newton solve. :class:`mrx.relaxation.config.Newton` uses
them as its defaults.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mrx.operators import _mass_solve
from mrx.optimization.shape_ad import map_jacobian
from mrx.quadrature import evaluate_at_xq
from mrx.relaxation.physics import induction_field
from mrx.solvers import minres
from mrx.spline_bases import basis_derivative_table

#: Production defaults: the weight of the parallel-flow penalty (in units of the field's strain), the relative
#: tolerance of the Newton solve in the norm of its preconditioner, the maximum number of MINRES iterations per
#: solve and the relative tolerance of the two 1-form mass solves inside every apply of the Gauss-Newton operator.
#: li383 graded (48,64,24), p = 2, from the file's field (2026-10-18): mass solves to 1e-6 and to 1e-8 give the same
#: trajectory, 1e-5 reaches the force residual 1e-8 four steps later and 1e-4 stalls at 2e-7. MINRES is not a
#: flexible method, and the error of the mass solves enters its operator. With tolerance 0.1 and 300 iterations the
#: solve takes the steps of the block Gauss-Newton system under flexible GCR on li383 M, L, p = 3 and W7-X, at the
#: same or 5-11 % less time. Without the reorthogonalisation it needs 1.6-2.6 times the iterations. Tolerance sweep
#: on li383 (48,64,24), (64,80,32) and W7-X (32,64,36), graded, 30 steps (2026-10-19): 0.1 reaches the force residual
#: 1e-8 first or level (17.5, 27.4, 19.7 s). 0.05 and 0.03 take fewer steps at 1.4-1.7 times the cost per step and end
#: 2-5 times lower after 30 steps, 0.2 needs 18-21 steps.
NEWTON_PENALTY = 3.0
NEWTON_TOL = 0.1
NEWTON_MAXITER = 300
NEWTON_MASS_SOLVE_TOL = 1e-6


def _derivatives_at_quadrature(view, dofs, k):
    """The logical derivatives ``d_c`` of the reference components of the k-form ``dofs`` (k = 1, 2) at the
    quadrature points, ``(n_q, 3, 3)`` with ``[q, c, i]`` the derivative of component ``i`` along ``c``: the form
    evaluated with the derivatives of its 1-D splines on the axis ``c``, exact."""
    form = view.basis_1 if k == 1 else view.basis_2
    x = (view.quad.x_x, view.quad.x_y, view.quad.x_z)
    c_dofs = view.E(k).T @ dofs
    out = []
    for c in range(3):
        info = []
        for comp in range(3):
            deriv = form.derivative_axes(comp)
            tables = []
            for a in range(3):
                if a != c:
                    tables.append((view.d_basis_jk if a in deriv else view.basis_jk)[a])
                else:
                    tables.append(view.dd_basis_jk[a] if a in deriv
                                  else basis_derivative_table(view.basis_0.Lambda[a], x[a]))
            info.append((comp, *tables))
        out.append(evaluate_at_xq(c_dofs, info, list(form.shape), view.quad.shape, 3))
    return jnp.stack(out, axis=1)


def atom_profiles(seq, field):
    """Radial profiles of a 2-form ``field``, on the radial quadrature points, used to build the preconditioner.

    Returns ``(prof_t, prof_z, strain)``. ``prof_t`` and ``prof_z`` are the averages over ``theta`` and ``zeta``
    of the field's logical ``theta`` and ``zeta`` components ``f = Bh / J``. ``strain`` has shape ``(nq_r, 3)`` and
    holds, for each logical direction ``c``, the angle average of ``sum_i (d_c f^i)^2``, i.e. how fast the field
    varies along that direction. The derivatives are exact: those of the 2-form's splines and of the map's
    Jacobian determinant, ``d_c f = (d_c Bh - f d_c J) / J`` with ``d_c J / J = tr(DPhi^-1 d_c DPhi)``.
    """
    shape = tuple(int(v) for v in seq.quad.shape)
    J = seq.jacobian_j
    B_jk = seq.odd.evaluate_at_quadrature(field, 2)
    f_jk = B_jk / J[:, None]
    D, dD, _ = map_jacobian(seq, seq.map, second=True)
    dlogJ = jnp.einsum("qij,qkji->qk", jnp.linalg.inv(D), dD)
    grads = (_derivatives_at_quadrature(seq.odd, field, 2) - f_jk[:, None, :] * dlogJ[:, :, None] * J[:, None, None]
             ) / J[:, None, None]
    f_jk = f_jk.reshape(shape + (3,))
    prof_t = f_jk[..., 1].mean(axis=(1, 2))
    prof_z = f_jk[..., 2].mean(axis=(1, 2))
    strain = (grads ** 2).sum(axis=-1).reshape(shape + (3,)).mean(axis=(1, 2))
    return prof_t, prof_z, strain


def parallel_penalty_profile(profiles, kappa):
    """The radial weight ``w(r)`` of the parallel-flow penalty: ``kappa`` times the field's strain along the field.

    With ``(h_t, h_z, s)`` the output of :func:`atom_profiles`, the strain along the field is
    ``(h_t^2 s_t + h_z^2 s_z) / (h_t^2 + h_z^2)``.
    """
    prof_t, prof_z, strain = profiles
    return kappa * (prof_t ** 2 * strain[:, 1] + prof_z ** 2 * strain[:, 2]) / (prof_t ** 2 + prof_z ** 2)


def _cardinal(q, x):
    """The cardinal B-spline of degree ``q``, with the knots ``0, 1, ..., q + 1``, at ``x``."""
    if q == 0:
        return ((x >= 0) & (x < 1)).astype(float)
    return (x * _cardinal(q - 1, x) + (q + 1 - x) * _cardinal(q - 1, x - 1)) / q


def _angular_symbols(n, q):
    """The Fourier symbols ``(mass, derivative, stiffness)`` of the 1-D spline matrices ``int N_i N_k``,
    ``int N_i' N_k`` and ``int N_i' N_k'`` of the periodic splines of degree ``q`` on ``n`` equal cells of
    ``[0, 1)``, per mode ``m`` in the order of ``np.fft.fftfreq``. On uniform knots the matrices are circulant, so the
    discrete Fourier transform diagonalises them exactly: each matrix maps the coefficients ``exp(2 pi i m j / n)``
    to its symbol times them. They are NumPy arrays that depend on ``n`` and ``q`` alone."""
    g, w = np.polynomial.legendre.leggauss(q + 1)
    x = (np.arange(q + 1)[:, None] + 0.5 * (g[None, :] + 1.0)).ravel()      # Gauss points of the cells of one spline
    wq = np.tile(0.5 * w, q + 1)
    j = np.arange(-q, q + 1)                                                # the splines that overlap it
    shifted = x[None, :] - j[:, None]

    def derivative(y):
        return _cardinal(q - 1, y) - _cardinal(q - 1, y - 1) if q else np.zeros_like(y)
    N, dN = _cardinal(q, x), derivative(x)
    phase = np.exp(2j * np.pi * np.fft.fftfreq(n)[:, None] * j[None, :])
    return (phase @ (_cardinal(q, shifted) @ (wq * N)) / n, phase @ (_cardinal(q, shifted) @ (wq * dN)),
            phase @ (derivative(shifted) @ (wq * dN)) * n)


def spline_preconditioner(seq, profiles, penalty):
    """The preconditioner of the Newton system built from spline operators, a :class:`SplineAtom`: an SPD approximate
    inverse of ``curl^T H curl``.

    It is the preconditioner of the 1-form Laplacian, scaled on both sides by ``lambda^{-1/2}``, where ``lambda`` is
    a model of the Hessian discretised on the splines: on each component of the raw 1-forms the form
    ``int (h_t d_theta u + h_z d_zeta u)^2 + (s_c + w) u^2``, the squared derivative along the field (``h_t``, ``h_z``
    from ``profiles``, see :func:`atom_profiles`), ``s_c`` the strain along the component's direction and
    ``w`` the parallel-flow ``penalty``, assembled on the component's tensor-product splines, relative to its mass
    matrix. The angular matrices are circulant, so the Fourier modes ``(m, n)`` of the coefficients diagonalise them,
    and per mode the model is a pencil of two radial spline matrices, ``K u = lambda M u``. ``M`` is the radial mass
    matrix and ``K`` the radial mass matrix weighted by the model's symbol at ``(m, n)``, a function of ``r``. The
    scaling takes the pencil's diagonal: ``lambda = K_ii / M_ii`` for the radial spline ``i``, its Rayleigh quotient
    on that spline. So the atom holds one number per degree of freedom of the raw 1-forms.

    The angular symbols assume uniform periodic knots in ``theta`` and ``zeta``. On other angular knots the matrices
    are not circulant and the atom is the one of the uniform knots with the same number of cells.

    Build it once per Newton step.
    """
    prof_t, prof_z, strain = profiles
    w_r = jnp.asarray(seq.quad.w_x)
    shapes = [tuple(int(v) for v in s) for s in seq.basis_1.shape]
    lam, dlam = seq.basis_0.Lambda, seq.basis_0.dLambda
    radial = (seq.d_basis_jk[0], seq.basis_jk[0], seq.basis_jk[0])
    scale = []
    for c in range(3):
        theta, zeta = (dlam if c == 1 else lam)[1], (dlam if c == 2 else lam)[2]
        mu_t, d_t, s_t = _angular_symbols(int(theta.n), int(theta.p))
        mu_z, d_z, s_z = _angular_symbols(int(zeta.n), int(zeta.p))
        a1 = (s_t.real / mu_t.real)[None, :, None]
        a2 = (s_z.real / mu_z.real)[None, None, :]
        a3 = (2 * (d_t[:, None] * np.conj(d_z[None, :])).real / (mu_t.real[:, None] * mu_z.real[None, :]))[None]
        # each row averages a radial weight over one radial spline: K_ii / M_ii of the weighted mass matrix K
        average = radial[c] ** 2 * w_r
        average = average / average.sum(axis=1, keepdims=True)

        def diagonal(weight):
            return (average @ weight)[:, None, None]
        # a1 a2 >= (a3 / 2)^2 on each mode and (average h_t h_z)^2 <= (average h_t^2) (average h_z^2), both by the
        # Cauchy-Schwarz inequality, so the model is at least the averaged strain and penalty
        model = (a1 * diagonal(prof_t ** 2) + a2 * diagonal(prof_z ** 2) + a3 * diagonal(prof_t * prof_z)
                 + diagonal(strain[:, c] + penalty))
        scale.append(1.0 / jnp.sqrt(model))
    return SplineAtom(scale=tuple(scale), shapes=tuple(shapes))


class SplineAtom(eqx.Module):
    """The Newton preconditioner of :func:`spline_preconditioner`, applied to a dual 1-form ``x`` as
    ``atom(seq, x)``.

    It is a pytree that holds only the scales per radial spline and Fourier mode, so a new field does not trigger a
    recompile. The sequence is passed at each call.
    """

    scale: tuple
    shapes: tuple = eqx.field(static=True)

    def _C(self, x):
        out, off = [], 0
        for sc, s in zip(self.scale, self.shapes):
            n_c = s[0] * s[1] * s[2]
            X = x[off:off + n_c].reshape(s)
            out.append(jnp.fft.ifft2(jnp.fft.fft2(X, axes=(1, 2)) * sc, axes=(1, 2)).real.ravel())
            off += n_c
        return jnp.concatenate(out)

    def __call__(self, seq, x):
        E = seq.E(1)
        y = E @ self._C(E.T @ x)
        y = seq.L[1].precondition(y)
        return E @ self._C(E.T @ y)


def gauss_newton_parts(seq, B, penalty=None, mass_tol=None):
    """``(change, change_t, parallel)``: the pieces of the Gauss-Newton part of the Hessian at ``B``.

    ``change(u_jk)`` is the field change ``Q = curl M_1^-1 load(u x h)`` (a 2-form of ``seq.odd``) of the velocity
    with the values ``u_jk`` at the quadrature points, ``change_t(X)`` is ``Q^T M_2 X`` (a dual 2-form of
    ``seq.even``) and ``parallel(u_jk)`` the parallel-flow penalty of :func:`second_variation` (zero for
    ``penalty=None``). So ``change_t(change(u_jk)) + parallel(u_jk)`` is ``H u`` without the terms in ``J``. Each of
    ``change`` and ``change_t`` costs one 1-form mass solve to the relative tolerance ``mass_tol``, and ``h`` one free
    1-form mass solve, once."""
    odd, even = seq.odd, seq.even
    B_jk = odd.evaluate_at_quadrature(B, 2)
    tol = seq.tol if mass_tol is None else mass_tol
    h_jk = induction_field(B, seq, tol)
    curl = odd.G[1]

    def m1_inv(x):
        return _mass_solve(odd, x, 1, tol=tol)

    def change(u_jk):
        return curl @ m1_inv(odd.cross_product_load_values(u_jk, h_jk, 1, 2, 1))

    def change_t(X):
        Y = m1_inv(odd.D[1].T @ X)
        return even.cross_product_load_values(h_jk, odd.evaluate_at_quadrature(Y, 1), 2, 1, 1)

    if penalty is None:
        return change, change_t, lambda u_jk: 0.0
    Bsq_over_J2 = jnp.einsum('qi,qij,qj->q', B_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2
    weight = jnp.repeat(penalty, int(seq.quad.shape[1]) * int(seq.quad.shape[2]))

    def parallel(u_jk):
        s = jnp.einsum('qi,qij,qj->q', u_jk, seq.metric_jkl, B_jk) / seq.jacobian_j ** 2 / Bsq_over_J2
        return even.vector_load_values(B_jk * (weight * s)[:, None], 2, 2)
    return change, change_t, parallel


def second_variation(seq, B, J=None, penalty=None, mass_tol=None):
    """The map ``u -> H u`` that applies the energy Hessian at ``B`` to a velocity 2-form ``u``.

    ``J`` is the weak curl of ``B``. The result is a dual 2-form. Each application costs two 1-form mass solves,
    to the relative tolerance ``mass_tol`` (``seq.tol`` if ``None``), and with ``J`` two more free 1-form mass solves
    (``Pi_1 Q`` and ``Pi_1(J x u)``). ``J=None`` leaves out the two terms in ``J``.
    What remains is the Gauss-Newton part ``(u, H v) = (Q_u, Q_v)_M``, which is positive semidefinite.
    Flows along the field, ``u = f B``, do not change ``B`` and form a null space of ``H``. ``penalty`` (the
    weight ``w(r)`` of :func:`parallel_penalty_profile`) removes it by adding
    ``(v, M_par u) = int w(r) (v . B)(u . B) / |B|^2 dx``, which leaves flows across the field unchanged.
    ``penalty=None`` gives the bare Hessian.
    """
    # B, J and the intermediate forms live in the odd parity view, u and H u in the even one
    odd, even = seq.odd, seq.even
    change, change_t, parallel = gauss_newton_parts(seq, B, penalty, mass_tol)
    if J is None:
        def apply(u):
            u_jk = even.evaluate_at_quadrature(u, 2)
            Hu = change_t(change(u_jk))
            return Hu if penalty is None else Hu + parallel(u_jk)
        return apply

    curl = odd.G[1]
    free = odd.free
    J_jk = odd.evaluate_at_quadrature(J, 1)
    tol = seq.tol if mass_tol is None else mass_tol
    h_jk = induction_field(B, seq, tol)

    def apply(u):
        u_jk = even.evaluate_at_quadrature(u, 2)
        Q = change(u_jk)
        hQ_jk = induction_field(Q, seq, tol)
        # (J, u x Pi_1 Q_v) = (Pi_1(J x u), Q_v): the free projection z of J x u, tested against the 2-forms
        z = _mass_solve(free, free.cross_product_load_values(J_jk, u_jk, 1, 1, 2), 1, tol=tol)
        Jxu = odd.vector_load_values(free.evaluate_at_quadrature(z, 1), 1, 2)
        # dJ + W / 2 with dJ = M_1^-1 D_1^T Q and W = M_1^-1 curl^T Pi_1(J x u), in one solve
        Y = _mass_solve(odd, odd.D[1].T @ Q + 0.5 * (curl.T @ Jxu), 1, tol=tol)
        Y_jk = odd.evaluate_at_quadrature(Y, 1)
        Hu = (even.cross_product_load_values(h_jk, Y_jk, 2, 1, 1)
              + 0.5 * even.cross_product_load_values(hQ_jk, J_jk, 2, 1, 1))
        return Hu if penalty is None else Hu + parallel(u_jk)

    return apply


def newton_direction(seq, B, load, a_guess, kappa=NEWTON_PENALTY, tol=NEWTON_TOL, maxiter=NEWTON_MAXITER,
                     mass_tol=NEWTON_MASS_SOLVE_TOL):
    """The Gauss-Newton direction at ``B`` among the curls. Returns ``(u, a, info)`` with ``u = curl a``.

    It is the step of a domain whose flows hold no harmonic field, the even flows of a stellarator-symmetric
    sequence. Otherwise :func:`target_direction` adds the harmonic flow.
    ``load`` is the Lorentz force ``J x h`` as a dual 2-form. It need not be projected: the right-hand side
    ``curl^T load`` is blind to its gradient and harmonic parts. ``a_guess`` is the warm start for the potential
    ``a`` (pass the previous step's ``a``) and ``kappa`` the weight of the parallel-flow penalty. The operator is
    the Gauss-Newton part of the Hessian (:func:`second_variation` without ``J``) with its mass solves to
    ``mass_tol``. The solve (:func:`newton_mr`) stops when its residual, in the norm of the preconditioner, is
    below ``tol`` times the right-hand side or after ``maxiter`` MINRES iterations. ``info`` is the number of MINRES
    iterations, positive when ``tol`` was met and negative when not.
    """
    even = seq.even                           # a and u = curl a live in the even parity view
    # Both parity views are taken from a full sequence. From a view the other one is reached through its static
    # base, and its geometry would be compiled in as a constant that a later set_geometry_from_map does not reach.
    profiles = atom_profiles(seq, B)
    penalty = parallel_penalty_profile(profiles, kappa)
    curl, curl_t, A = _newton_system(seq, B, penalty, mass_tol)
    atom = spline_preconditioner(seq, profiles, penalty)

    def P(x):
        return atom(even, x)

    def norm(r):
        return jnp.sqrt(r @ P(r))
    a, info = newton_mr(A, P, curl_t(load), a_guess, tol, maxiter, norm)
    return curl(a), a, jnp.asarray(info, dtype=jnp.int32)


def target_direction(seq, B, load, a_guess, harmonic=None, shells=None, slips=None, kappa=NEWTON_PENALTY,
                     tol=NEWTON_TOL, maxiter=NEWTON_MAXITER, mass_tol=NEWTON_MASS_SOLVE_TOL):
    """The Gauss-Newton direction with scalar borders: the amplitudes of the harmonic flows ``harmonic``, the shell
    volumes (:mod:`mrx.relaxation.shells`) and the poloidal flux slips (:class:`mrx.relaxation.twist.Slips`). Returns
    ``(u, a, info, c_h, c, d, dB_slip)``: the velocity ``u = curl a + h^T c_h + W c``, the field change of the slips
    ``dB_slip = T d``, so that the step's field change is ``curl Pi_1(u x h) + T d``, and the coefficients ``c_h``, ``c`` and
    ``d`` (``None`` for what is not given).

    ``harmonic`` holds the harmonic 2-forms ``h`` of the flows as rows (``seq.even.nullspace(2)``), the one of a solid
    torus without stellarator symmetry. With the symmetry the even flows hold none and the force has no harmonic part.
    ``shells`` is a :class:`~mrx.relaxation.shells.Shells`, with the shell velocities ``W c`` and their transpose
    ``W_t`` (products with its stored gradient velocities). ``slips`` is a
    :class:`~mrx.relaxation.twist.Slips`, with the field changes ``T d`` of the twists and their transpose ``T_t``.
    The system is the Gauss-Newton part of the Hessian of the energy on that space, with ``Q`` the field change of a
    velocity and ``M`` the 2-form mass matrix (:func:`gauss_newton_parts`),

        [ curl^T H curl    curl^T H h^T    curl^T H W    curl^T Q^T M T ] [a  ]   [ curl^T load        ]
        [ h H curl         h H h^T         h H W         h Q^T M T      ] [c_h] = [ h load             ]
        [ W_t H curl       W_t H h^T       W_t H W       W_t Q^T M T    ] [c  ]   [ W_t load + g_p     ]
        [ T_t M Q curl     T_t M Q h^T     T_t M Q W     T_t M T        ] [d  ]   [ -T_t M B + g_slip  ],

    solved by :func:`newton_mr` with the block-diagonal preconditioner of the atom and the inverses of the diagonal
    blocks: ``h H h^T`` (one apply of the system per harmonic form) and the dense blocks of the shells and the slips.
    The curls do not reach ``h``, and the harmonic row ``h load`` is the harmonic part of the force, which the
    potential alone cannot remove. The rows of the shells and the slips are the derivatives of the energy less the
    targets' work along them. ``load`` is the unprojected Lorentz force ``J x h``: along the shells its gradient
    part is what counts. The slip block ``T_t M T`` is the energy's own second derivative along the slips, since the
    field is linear in ``d``."""
    even = seq.even
    n_a = a_guess.size
    profiles = atom_profiles(seq, B)
    penalty = parallel_penalty_profile(profiles, kappa)
    atom = spline_preconditioner(seq, profiles, penalty)
    curl = even.G[1]
    n_h = 0 if harmonic is None else harmonic.shape[0]
    n_c = 0 if shells is None else shells.g_p.size
    n_d = 0 if slips is None else slips.g.size
    if shells is not None:
        W, W_t = shells.velocity, shells.velocity_t

    def velocity(z):
        u = curl @ z[:n_a]
        if harmonic is not None:
            u = u + harmonic.T @ z[n_a:n_a + n_h]
        return u if shells is None else u + W(z[n_a + n_h:n_a + n_h + n_c])

    def rows(Hu):
        """The rows of the potential, the harmonic forms and the shells of the dual 2-form ``Hu``."""
        return ([curl.T @ Hu] + ([harmonic @ Hu] if harmonic is not None else [])
                + ([W_t(Hu)] if shells is not None else []))

    rhs = rows(load)
    if shells is not None:
        rhs[-1] = rhs[-1] + shells.g_p
    blocks = [shells.D_inv] if shells is not None else []
    if slips is None:
        Hs = second_variation(seq, B, None, penalty, mass_tol)

        def A(z):
            return jnp.concatenate(rows(Hs(velocity(z))))
    else:
        change, change_t, parallel = gauss_newton_parts(seq, B, penalty, mass_tol)
        T, T_t = slips.maps(seq)
        M2 = seq.odd.M[2]
        rhs.append(slips.g - T_t(M2 @ B))
        blocks.append(slips.D_inv)

        def A(z):
            u = velocity(z)
            u_jk = even.evaluate_at_quadrature(u, 2)
            X = change(u_jk) + T(z[n_a + n_h + n_c:])
            return jnp.concatenate(rows(change_t(X) + parallel(u_jk)) + [T_t(M2 @ X)])
    n = n_a + n_h + n_c + n_d
    if harmonic is not None:
        # h H h^T is positive: the penalty alone holds it up where h is along B (the vacuum field)
        e = jnp.eye(n_h, n, n_a, dtype=a_guess.dtype)
        blocks.insert(0, jnp.linalg.inv(jax.lax.map(lambda x: A(x)[n_a:n_a + n_h], e)))

    def P(z):
        out, off = [atom(even, z[:n_a])], n_a
        for D_inv in blocks:
            out.append(D_inv @ z[off:off + D_inv.shape[0]])
            off += D_inv.shape[0]
        return jnp.concatenate(out)

    def norm(r):
        return jnp.sqrt(r @ P(r))
    z0 = jnp.concatenate([a_guess, jnp.zeros(n - n_a, a_guess.dtype)])
    z, info = newton_mr(A, P, jnp.concatenate(rhs), z0, tol, maxiter, norm)
    c_h = z[n_a:n_a + n_h] if harmonic is not None else None
    c = z[n_a + n_h:n_a + n_h + n_c] if shells is not None else None
    d = z[n_a + n_h + n_c:] if slips is not None else None
    dB_slip = T(d) if slips is not None else None
    return velocity(z), z[:n_a], jnp.asarray(info, dtype=jnp.int32), c_h, c, d, dB_slip


def _newton_system(seq, B, penalty, mass_tol):
    """The maps ``curl``, ``curl^T`` and ``a -> curl^T H curl a`` of the Gauss-Newton system on the even view of the
    full sequence ``seq``."""
    Hs = second_variation(seq, B, None, penalty, mass_tol)
    even = seq.even

    def curl(a):
        return even.G[1] @ a

    def curl_t(y):
        return even.G[1].T @ y

    def A(a):
        return curl_t(Hs(curl(a)))
    return curl, curl_t, A


def newton_mr(A, P, b, x0, tol, maxiter, norm):
    """Solve the symmetric ``A x = b`` by Newton-MR (Liu & Roosta 2022). Returns ``(x, info)``.

    If the warm start ``x0`` already meets ``norm(b - A x0) <= tol norm(b)`` it is returned unchanged.
    Otherwise one preconditioned MINRES solve computes the correction to ``x0``. It stops when its own residual
    estimate (in the preconditioner's norm) meets the tolerance, or after ``maxiter`` iterations. MINRES
    stops early at a direction of nonpositive curvature. Such a direction is a descent
    direction only for the right-hand side it was found with, so the solve is then repeated for ``b`` itself
    from zero. MINRES keeps its Lanczos vectors and reorthogonalises each new one (``reorthogonalize`` of
    :func:`mrx.solvers.minres`), which stores ``2 maxiter`` vectors. ``info`` is the number of MINRES iterations,
    positive when the tolerance was met and negative when not.
    """
    x = x0
    bnorm = norm(b)
    r0 = b - A(x)

    def solve(_):
        rnorm = norm(r0)
        d, info, npc = minres(A, r0 / rnorm, M=P, tol=tol * bnorm / rnorm, maxiter=maxiter,
                              npc_exit=True, norm=norm, reorthogonalize=True)

        def from_zero(_):
            d0, info0, _ = minres(A, b / bnorm, M=P, tol=tol, maxiter=maxiter, npc_exit=True,
                                  norm=norm, reorthogonalize=True)
            return d0 * bnorm, jnp.abs(info0).astype(jnp.int32)

        x_new, its_npc = jax.lax.cond(npc, from_zero,
                                      lambda _: (x + d * rnorm, jnp.zeros((), jnp.int32)),
                                      None)
        return x_new, b - A(x_new), (jnp.abs(info) + its_npc).astype(jnp.int32)

    def skip(_):
        return x, r0, jnp.zeros((), jnp.int32)

    x, r, its = jax.lax.cond(norm(r0) > tol * bnorm, solve, skip, None)
    return x, jnp.where(norm(r) <= tol * bnorm, its, -its)
