# Laplacian Preconditioner Notes

This note records the settled production policy for the Hodge-Laplacian
preconditioners after the recent `k = 0` shifted/nullspace cleanup.

The target operators are

$$
(L_k + \varepsilon M_k) u = f
$$

for shifted solves, and the Moore-Penrose inverse action of `L_k` for the
singular `\varepsilon = 0` case.

Legacy code and scripts may still refer to `hx` or `fd` for the scalar helper.
In the current implementation and in this note, the preferred term is
`tensor` for the structured scalar `k = 0` Laplacian-side preconditioner.

## 1. General Rule

The harmonic space must be treated differently in the shifted and unshifted
problems.

For `\varepsilon = 0`, the correct operation is nullspace deflation:

$$
L_k^{+} = P_{\perp} L_k^{-1} P_{\perp}
$$

on the range of `L_k`, interpreted as the pseudoinverse action.

For `\varepsilon > 0`, deflation is wrong. The harmonic mode is no longer in
the kernel; it becomes an eigenmode with eigenvalue `\varepsilon`. The correct
structure is

$$
(L_k + \varepsilon M_k)^{-1}
\approx
\left(I - Z Z^T M_k\right) B_{\perp}
\left(I - M_k Z Z^T\right)
+ \frac{1}{\varepsilon} Z Z^T,
$$

where the columns of `Z` form an `M_k`-orthonormal basis of the harmonic
subspace.

So the policy is:

- `\varepsilon = 0`: use nullspace deflation.
- `\varepsilon > 0`: use a complement preconditioner plus an explicit harmonic
	coarse correction.

## 2. Settled `k = 0` Policy

### 2.1 When the harmonic mode is known

For free `k = 0`, the preferred shifted preconditioner is:

- the structured tensor preconditioner on the complement,
- plus the exact harmonic coarse term `(1 / \varepsilon) z_0 z_0^T`.

In formula form,

$$
P_{0,\mathrm{shift}}^{-1}
=
\left(I - z_0 z_0^T M_0\right) B_{0,\perp}
\left(I - M_0 z_0 z_0^T\right)
+ \frac{1}{\varepsilon} z_0 z_0^T.
$$

This is the intended end state for ordinary shifted scalar solves.

### 2.2 When the harmonic mode is not yet known

During inverse iteration, or in any early shifted solve before a valid stored
coarse vector exists, the solver must stay robust without a nullspace guess.

The current implementation therefore uses this rule:

- inverse iteration explicitly disables the harmonic coarse correction;
- the shifted free `k = 0` tensor branch falls back to shifted Jacobi until a
	real coarse vector is available;
- once the coarse vector is available, ordinary shifted solves switch back to
	the tensor-plus-coarse path.

This is the key control-flow split in the codebase now.

### 2.3 What not to do

An `L^{-1}`-based expansion such as

$$
L^{+} - \varepsilon L^{+} M L^{+}
$$

is not a complete shifted preconditioner on the full space. It is only a
complement approximation. On the harmonic mode it misses the required
`1 / \varepsilon` scaling entirely.

The same applies to variants like

$$
L^{+} - \varepsilon L^{+} M L^{+} + \varepsilon M^{-1}.
$$

Those expansions may be useful as complement refinements, but they are not a
replacement for the explicit harmonic coarse term.

## 3. Settled `k = 3` Policy

For shifted `k = 3`, the practical production default remains conservative:

- use shifted Jacobi on the complement,
- add the explicit harmonic coarse correction when the harmonic mode is known.

The scalar-duality / tensor-duality round trip is still available as an
experimental upper-block ingredient, but it is not the default practical
recommendation at this point.

So the current reading is:

- `k = 0`: tensor complement preconditioner is a production-quality building
	block.
- `k = 3`: explicit coarse handling is production-quality; the transferred
	scalar-duality correction is still experimental.

## 4. Nullspace Construction Policy

The nullspace arrays are now always initialized on the operator bundle, with
zero vectors as placeholders until real harmonic vectors are computed.

That simplifies the control flow, but it does **not** change the solver rules:

- unshifted solves still use nullspace deflation;
- shifted solves still avoid deflation;
- inverse iteration still disables harmonic coarse correction while it is
	discovering the harmonic vectors.

So the zero placeholders are a storage convention, not a license to treat the
shifted problem as if its harmonic space should be projected out.

## 5. Practical Summary

- Unshifted `k = 0`: deflate the constant mode and use the tensor complement
	preconditioner.
- Shifted `k = 0`, harmonic mode known: tensor complement plus exact harmonic
	coarse correction.
- Shifted `k = 0`, harmonic mode unknown: no harmonic coarse correction; use
	the robust shifted fallback until the nullspace vector is available.
- Shifted `k = 3`: Jacobi plus explicit harmonic coarse correction remains the
	practical default.