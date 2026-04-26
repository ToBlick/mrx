# Laplacian preconditioner notes

This note records what we learned while debugging the shifted Hodge-Laplacian preconditioners, mainly for the toroidal test problems with Betti numbers `(1, 1, 0, 0)`.

The main target operators were the shifted systems

$$
(L_k + \varepsilon M_k) u = f,
$$

with special attention to `k = 0` without Dirichlet BCs and `k = 3` with Dirichlet BCs, since these are the cases with one-dimensional harmonic spaces.

## Main conclusions

### 1. Shifted `k = 3` needs an explicit harmonic coarse correction

For shifted `k = 3`, the important source-level improvement was the explicit rank-1 correction on the stored harmonic mode. If `z_3` is the stored `M_3`-normalised nullspace vector,

$$
z_3^T M_3 z_3 = 1,
$$

then the coarse-aware preconditioner is

$$
P_{3,\mathrm{coarse}}^{-1}
=
\left(I - z_3 z_3^T M_3\right) B_3^{-1} \left(I - M_3 z_3 z_3^T\right)
+ \frac{1}{\varepsilon} z_3 z_3^T.
$$

This makes the shifted preconditioner exact on the harmonic direction:

$$
P_{3,\mathrm{coarse}}^{-1}(\varepsilon M_3 z_3) = z_3.
$$

In practice this was essential. Without it, shifted `k = 3` solves stall badly.

### 2. Shifted `k = 0` also benefits from the same exact rank-1 fix

For `k = 0`, `dirichlet = False`, the harmonic space is the constant mode. The same rank-1 wrapper gives

$$
P_{0,\mathrm{coarse}}^{-1}
=
\left(I - z_0 z_0^T M_0\right) B_0^{-1} \left(I - M_0 z_0 z_0^T\right)
+ \frac{1}{\varepsilon} z_0 z_0^T,
$$

where `z_0` is the stored `M_0`-normalised constant mode.

This was worthwhile to keep in `src`. It makes the scalar preconditioner exact on the constant mode and improves the standalone shifted `k = 0` solves.

### 3. The scalar HX/FD preconditioner is genuinely good for `k = 0`

The shifted scalar fast-diagonalisation preconditioner is clearly stronger than shifted Jacobi on its own problem. In the debug runs it reduced iteration counts substantially for random and mixed right-hand sides, while also solving the pure harmonic case in essentially one or two iterations once the explicit coarse correction was enabled.

So there is no evidence that the scalar `k = 0` auxiliary solve itself is broken.

### 4. Better scalar preconditioning does not fix the shifted `k = 3` HX preconditioner

We tried to reuse the improved coarse-aware shifted `k = 0` preconditioner inside the `k = 3` HX auxiliary-space sandwich. This did not materially improve the shifted `k = 3` solve, and that change was removed again.

The reason is that the outer `k = 3` coarse correction already handles the dangerous harmonic mode. Improving the scalar auxiliary solve on its own constant mode does not address the main failure on the `k = 3` complement.

### 5. The hard part is the `3 -> 0 -> 3` auxiliary correction, not the scalar solve

The `k = 3` HX correction has the form

$$
\widetilde M_3^{-1} M_{03} P_0^{-1} M_{30} \widetilde M_3^{-1},
$$

where `P_0^{-1}` is the shifted scalar inverse or preconditioner.

The experiments indicate that the weak point is this whole sandwich, especially how the transferred scalar correction acts on the non-harmonic `k = 3` modes.

The evidence for this is:

- `k = 0` HX/FD is strong on the scalar problem.
- `k = 3` harmonic-only right-hand sides are already solved in one iteration once the outer coarse correction is present.
- Yet `k = 3` HX remains much worse than `jacobi + coarse` on random and mixed right-hand sides.

### 6. Exact dense transfer helps relative to ordinary HX, but still loses to `jacobi + coarse`

We also tested a debug-only dense exact-lift version of the `0 -> 3` transfer. That improved the auxiliary-space correction somewhat compared with the standard HX path, especially when used with a small hybrid weight, but it still did not beat plain `jacobi + coarse`.

This is important because it rules out the simplest failure mode:

- it is not just a bug from using an approximate transfer,
- it is not just the scalar nullspace handling,
- and it is probably not a single missing global geometry factor.

The current interpretation is that the auxiliary-space correction is not spectrally well matched to the shifted `k = 3` complement, even when the transfer is applied exactly in dense form.

## Implications for the non-shifted `eps = 0` case

The experiments above targeted shifted systems, so they do not transfer verbatim to the singular `eps = 0` problem. Still, they do suggest a fairly clear picture for the pure Hodge-Laplacian solve.

### 1. The harmonic space should be handled by nullspace deflation, not by a shifted coarse term

For `eps > 0`, the rank-1 coarse correction works because the harmonic mode becomes an eigenvector with eigenvalue `eps`. For `eps = 0`, that formula is no longer meaningful because the factor `1 / eps` blows up.

So the correct analogue is not a shifted coarse correction but explicit nullspace deflation: project the right-hand side into the range of `L_k`, solve on that complement, and interpret the result as the Moore-Penrose inverse action.

This is already the right conceptual framework for the non-shifted solver.

### 2. The good `k = 0` scalar behaviour should still matter at `eps = 0`

The shifted experiments support the view that the scalar FD/HX preconditioner is fundamentally good on the scalar complement, not just on the lifted constant mode. That should carry over to the pseudoinverse problem as well.

So for `k = 0`, the non-shifted conclusion is still favorable: after deflating the constant mode, the scalar HX/FD inverse remains a sensible and likely strong preconditioner for the complement solve.

Later dense extracted experiments made this sharper. On the polar extracted
scalar problem, the free-case nullspace is not removed merely by inverting the
bulk block; it reappears in the Schur complement. A naive dense Schur inverse
therefore breaks positivity even when the bulk model itself is accurate.

The practical lesson is:

- project the right-hand side into the range of the operator,
- use mass-orthogonal normalization for the harmonic mode,
- and if an extracted block preconditioner is used, treat the free-case Schur
	block as a deflated / pseudoinverse solve rather than an ordinary inverse.

Once that Schur nullspace handling was added, the higher-rank diagonal scalar
preconditioners behaved as expected and significantly outperformed Jacobi.

### 3. The weak point for `k = 3` is still likely the auxiliary transfer sandwich

The shifted data strongly suggests that the weakness of the `k = 3` HX approach is not the harmonic mode itself and not the scalar auxiliary inverse in isolation, but the transferred correction

$$
\widetilde M_3^{-1} M_{03} P_0^{-1} M_{30} \widetilde M_3^{-1}
$$

acting on the `k = 3` complement.

That observation should remain relevant at `eps = 0`. Once the harmonic mode has been removed by nullspace deflation, the non-shifted solve is again controlled by how well the preconditioner acts on the complement. The shifted experiments therefore make it plausible that the same `3 -> 0 -> 3` auxiliary correction may remain the weak component in the pure `k = 3` solve.

This is a suggestive implication, not a proof. The shift changes the spectrum, so the non-shifted complement could still behave differently. But the current evidence points to the same structural bottleneck.

### 4. For `k = 3`, skepticism about HX should carry over unless new evidence appears

The safest practical reading is:

- for `eps = 0`, treat the harmonic mode by deflation,
- then judge the preconditioner only on the complement,
- and keep in mind that the shifted experiments gave no indication that the HX-style `3 -> 0 -> 3` correction dominates `jacobi` on that complement.

So the shifted results do not prove that non-shifted `k = 3` HX is bad, but they do remove one optimistic explanation: it is probably not enough to say "the scalar helper is good, therefore the 3-form auxiliary preconditioner must also be good."

### 5. What should be tested separately at `eps = 0`

If we want a clean non-shifted conclusion, the right tests are:

- compare `jacobi` vs `hx` after exact nullspace deflation,
- focus on random right-hand sides in the range of `L_3`,
- and measure whether the auxiliary correction improves the complement solve rather than the harmonic direction.

That would turn the current implication into a direct result.

## Practical recommendations

### Shifted `k = 3`, Dirichlet

Use shifted Jacobi plus explicit harmonic coarse correction as the default practical preconditioner:

$$
P^{-1} \approx P_{\mathrm{jacobi+coarse}}^{-1}.
$$

At the moment this is the most robust option we tested.

### Shifted `k = 0`, no Dirichlet

Use the shifted FD/HX scalar preconditioner with the explicit rank-1 coarse correction. This is a real improvement and should stay in the source implementation.

## What we are not claiming

- We are not claiming that auxiliary-space methods for `k = 3` are impossible.
- We are not claiming that geometry plays no role.
- We are not claiming that the current dense exact-lift experiment is the final word.

What we are claiming is narrower: with the current discrete operators and current experiments, the shifted `k = 3` HX-style correction is inferior to `jacobi + coarse`, while the shifted `k = 0` scalar HX/FD preconditioner is clearly useful.

## Open question

The remaining conceptual question is whether the exact-lift hybrid is simply mis-scaled, or whether it is correcting the wrong `k = 3` modes altogether. A scalar hybrid weight helps slightly when it is small, but not enough to beat `jacobi + coarse`. That suggests the current auxiliary correction may contain useful information, but not in a form that is competitive as the main shifted `k = 3` preconditioner.