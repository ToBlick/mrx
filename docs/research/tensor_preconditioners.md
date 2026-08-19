# Tensor preconditioners for the mass and Hodge Laplacian

**Status: experimental**, all in `mrx/experimental/block_jacobi_laplacian.py`
and opt-in. Production is unchanged (raw_kron masses, closed-form Jacobi
Laplacians). §1-6 are the reasoning; §7-10 the design, results and traps.

---

## 1. The situation

We approximately invert mass and Hodge-Laplacian operators repeatedly inside
Krylov solves. The operators are matrix-free and carry `10^5`-`10^6` DOFs, so a
direct factorisation is out and anything costing more than a few operator
applies to build is suspect. The cheap fallback is Jacobi — invert the diagonal —
which costs nothing and is weak.

How weak: the unshifted verification (`eps = 0`, tol 1e-10, all eight
`(k, dirichlet)` cases) needs 5,000-20,000 CG iterations for `k >= 1`, and two
W7-X cases do not converge inside 20,000 at all. The question is what else is
nearly free.

---

## 2. The asset: everything is a tensor product

The discretisation lives on a **logical box** with a tensor-product spline
basis, `phi_{ijk} = R_i(r) T_j(t) Z_k(z)`. Before the geometry gets involved,
that makes the operators separable:

* **Mass**: `M = A_r (x) A_t (x) A_z` — one Kronecker **product**.
* **Laplacian**: three terms, each differentiating one axis and leaving the
  others alone —

  ```
  K_r (x) M_t (x) M_z  +  M_r (x) K_t (x) M_z  +  M_r (x) M_t (x) K_z
  ```

  a Kronecker **sum**.

Both are almost free to invert. A Kronecker product inverts factor by factor:
three 1-D solves applied along the axes of the coefficient tensor, `O(n^2)`
storage per axis and `O(N^{4/3})` per apply. A Kronecker sum does not factor
like that, but it yields to fast diagonalisation.

Everything below is about how much of this survives contact with a real
geometry.

---

## 3. Fast diagonalisation

The tool that makes the Laplacian tractable (Lynch, Rice & Thomas 1964; in the
code `_fd_apply_3d` and `_simultaneous_diagonalize_pair`).

Per axis, solve the small generalised eigenproblem `K_a v = lambda M_a v` and
normalise the eigenvectors so that

```
V_a^T M_a V_a = I ,        V_a^T K_a V_a = diag(lambda_a)
```

Mass and stiffness are diagonalised **simultaneously** — mass to the identity,
stiffness to a diagonal. Applying `V_r (x) V_t (x) V_z` to the three-term sum
turns each term into a diagonal on one axis and identities on the others, so the
whole sum collapses to

```
lambda_r(i) + lambda_t(j) + lambda_z(k)
```

Inverting is then: transform along each axis, divide pointwise, transform back.
Three small dense matrices, `O(N^{4/3})` per apply, and the inverse is **exact**
for the operator it was given — no inner iteration, no fill-in, no tolerance.

Two conditions, which constrain everything downstream:

1. **The same `M_a` in all three terms.** Different masses per term means no
   shared eigenbasis and the trick fails.
2. **The operator must be a sum of separable terms** to begin with.

---

## 4. The obstacle: the metric is not separable

With the map `F` in place the integrals pick up the metric and Jacobian —
weights like `J`, `g^{ab}J`, `g_ab/J`. A weighted 1-D integral no longer
factors, so `M` is not a Kronecker product and the Laplacian is not a Kronecker
sum. The asset is gone unless we put it back by hand.

That is the whole design problem: **approximate a non-separable weight so the
tensor structure survives, and choose where to put the error.**

---

## 5. Three moves

### Move 1 — push the metric into a diagonal

A diagonal is compatible with tensor structure in a way a general weight is not:
`D (A_r (x) A_t (x) A_z) D` is still trivially invertible. So carry as much of
the metric as possible as a *pointwise scaling* rather than inside the integrals.

raw_kron does exactly this for the mass — `M ~ Lam (A (x) A (x) A) Lam` with
`Lam` chosen to reproduce the exact mass diagonal, so the metric is captured
pointwise and only the *coupling* between neighbouring DOFs is approximated. The
same move reappears for the Laplacian at `k >= 1`, where the weight factors into
a per-component and a per-axis part and the component part becomes a diagonal
sandwich.

**Why this is the good kind of approximation:** a diagonal keeps the *field*
exactly and gives up only its correlation with the rest of the operator.
Collapsing the same factor to a single scalar instead — the obvious "lumping" —
costs a factor of 2-8, because averaging destroys the variation a diagonal
preserves.

### Move 2 — lump what is left, one profile per axis

The metric sitting *between two derivatives* cannot be pushed outside. There we
do the crude thing deliberately: replace the weight by a product of 1-D profiles,

```
w(r,t,z) ~ w_r(r) . w_t(t) . w_z(z)
```

each profile being the average over the other two directions. A rank-one
approximation of the weight field — chosen not because it is accurate but
because it is the only thing that restores the structure the solver needs.

Two rules for doing it, neither obvious a priori:

**Average the product, not its parts.** The weights are products like
`g^{tt} J`. In toroidal coordinates `g^{tt} ~ 1/r^2` and `J ~ r`, so the bare
metric component is not integrable near the axis while the product `~ 1/r` is.
Averaging them separately manufactures a divergence the operator does not have.

**Weight the stiffnesses, not the masses.** In `K_r (x) M_t (x) M_z` the masses
are just "the identity in the other two directions" — `g^{rr}J` has *already*
been folded into `K_r` by averaging over exactly those directions. Weighting the
masses too counts the metric twice and yields an atom worse than the diagonal it
was meant to beat.

### Move 3 — stop lumping where lumping cannot work

At the polar axis the coordinate system is singular: the metric does not merely
vary, it varies without bound, and no product of 1-D profiles can represent it.
Refinement does not help — the innermost element is always the worst.

So do not model it. **Solve it exactly.** The DOFs touching the axis are a ring,
`O(n_theta n_zeta)` of them, so that block can be probed with one operator apply
per row and inverted densely. The separable model then owns only the region where
it is a reasonable description. The two blocks are treated as independent (block
Jacobi rather than a Schur complement) — a deliberate simplification that keeps
the construction uniform and avoids a second approximate inverse in the middle.

This split is the most effective knob in the design. Widening the exact region by
a few rings is worth as much as any refinement of the separable model, and the
two ways to widen it — adding rings, or grading the radial mesh so the first
element is fatter — are the same mechanism from different directions.

---

## 6. The de Rham complication, and natural BCs

For `k >= 1` the spaces are vector-valued and each component uses a *derivative*
spline on some axes and a primal spline on others. The tensor picture survives:
`d` and its adjoint act on **one axis at a time**, and which axis is fixed by
which basis is already differentiated. Each component still sees a three-term
Kronecker sum; on an axis whose basis is already a derivative spline, the
"stiffness" is a stiffness *of those splines*.

**Natural boundary conditions are a feature of the weak block, and they are the
whole story behind the free-BC cases.** `S_k = D_k^T M_{k+1} D_k` is the strong
part -- `d` applied directly, no integration by parts, no boundary term. `W_k`
contains `D_{k-1}^T`, the adjoint of `d`, defined by
`<delta_h w, tau> = <w, d tau>`. Integrating by parts leaves a surface term:

| k | `<w, d tau>` | boundary term involves |
| --- | --- | --- |
| 0 | -- (`W_0 = 0`) | **nothing** |
| 1 | `<u, grad tau>` | `u_r`, the NORMAL component |
| 2 | `<w, curl tau>` | `w_t, w_z`, the TANGENTIAL components |
| 3 | `<omega, div tau>` | `omega` at the boundary |

Under an essential (Dirichlet) condition `tau` vanishes on the boundary and every
one of those dies. That is why the dbc cases were uniformly strong from the start
while the free ones lagged 4-6x at every degree with a weak block -- k=0 excepted,
which is the control that rules out every other explanation.

The term is a surface integral on `r = 1`, so for a tensor basis it is

```
alpha . (e e^T) (x) M_t (x) M_z ,    e = dLam(1),  alpha = <w>_{theta,zeta}(1)
```

-- the same shape as the FIRST Kronecker term, so it merges into `K_r` as a
rank-one update. Nothing about the sum, the eigenbasis, the cost or the storage
changes, and `alpha` needs no fitting: it is the weight the stiffness already
uses, evaluated AT the boundary instead of averaged over `r`. This is
`bc_entry`, on by default, and exactly zero under Dirichlet and at k=0.

**Its limit is the coefficient, not the structure.** Folding the term into the
sum forces the face weight `w(1,theta,zeta)` down to a SCALAR. On a toroid,
where `J ~ R_0 + eps r cos theta` varies about +-33% across the face, that
scalar captures most of it (k=1 free 442 -> 128, and no rings needed at all). On
W7-X the face weight varies far more and it captures little (1660 -> 1101).
Carrying the variation would need `B_t`, `B_z` with real angular profiles -- but
then they are no longer proportional to `M_t`, `M_z`, the eigenbasis is no longer
shared, and fast diagonalisation breaks. An exact OUTER ring is precisely the
object that can carry an arbitrary `theta`-`zeta` dependence on one radial index,
which is why it is needed at k>=2 and on W7-X.

Confirmed directly by the dense spectrum probe: at k=1 free the preconditioned
operator has **44 outlier eigenvalues** out of 894 and condition 768, with the
extreme mode living **entirely** on the outer radial boundary; at k=1 dbc there
are **zero** outliers and condition 12.7.

## 7. The design, concretely

**Shape** (both operators): densely-probed **core** (polar ring + `extra_rings`)
plus a **separable bulk** on the remaining radial window, uncoupled.

**Laplacian.** One three-term Kronecker sum per component, fast-diagonalised,
masses unweighted. An axis is a *derivative axis* where the component's basis is
a derivative spline — k=0 none, k=1 axis `c`, k=2 all but `c`, k=3 all three —
and gets the stiffness of the derivative splines (`p >= 2`; at `p = 1` a DG-0
jump seminorm stands in). All weights follow one formula,
`w(k,c,a) = [mass weight of component c] . g^{aa}`:

| k | mass weight | w(c,a) |
| --- | --- | --- |
| 0 | `J` | `g^{aa} J` |
| 1 | `g^{cc} J` | `g^{cc} g^{aa} J` |
| 2 | `g_cc / J` | `g_cc g^{aa} / J` |
| 3 | `1 / J` | `g^{aa} / J` |

This reproduces every term derived separately (at k=1, `a = c` gives the div-div
weight `(g^{rr})^2 J`, `a != c` gives the curl-curl weight `g_dd/J` with `d` the
third axis). It assumes an **orthogonal** metric — exact on the toroid,
approximate on W7-X.

Because `w` factors into component and axis parts, the **bulk bracket is the same
at every degree** and the component factor rides along as a diagonal sandwich
`D_c^{-1/2} [bracket]^-1 D_c^{-1/2}`. So any improvement to the k=0 bracket
propagates to all four degrees unchanged.

**Mass.** Same shape, easier — a single Kronecker product, so the bulk inverse is
three 1-D solves and no fast diagonalisation is involved. The bulk model is
*identical* to raw_kron's; only the core changes, from the `E+` pseudoinverse to
a dense probe.

**Settings.** Two independent degeneracies, two knobs:

| what is degenerate | knob | when |
| --- | --- | --- |
| polar core | `extra_rings = 3` | every k, both BCs (knee at 3; 1-2 do nothing) |
| natural-BC face | `bc_entry` | ON by default; exactly zero under Dirichlet and at k=0 |
| natural-BC face, strong angular variation | `outer_rings = 2` | free BC at k>=2, and W7-X k=1 |

The mass wants `extra_rings = 0` -- extra rings hurt it, since it is local, well
conditioned, and its separable bulk is already accurate.

**Two bracket variants.** `averaged` shares one mass per axis so all three
directions fast-diagonalise together, at the cost of averaging away the radial
dependence of the `t`/`z` terms. `modal` keeps each weight's radial profile and
solves the radial pencil exactly per angular mode; more accurate and its
advantage grows with refinement, but storage goes `O(n_zeta n_r^2)` instead of
`O(n_r^2 + n_t^2 + n_z^2)`. Separable + 4 rings recovers most of the gap.

---

## 8. Measured

Full unshifted verification: `eps = 0`, tol 1e-10, all eight `(k, dirichlet)`
cases, deflated CG on the four singular ones, 12x24x12.
**Every case converges, and every case is 3.5-9.3x.**

| | | jacobi | best | config | speedup |
| --- | --- | --- | --- | --- | --- |
| toroid | k=0 free / dbc | 400 / 234 | 43 / 32 | r3 | **9.3x / 7.3x** |
| | k=1 free / dbc | 442 / 376 | 128 / 62 | none / r3 | 3.5x / **6.1x** |
| | k=2 free / dbc | 345 / 472 | 72 / 77 | r3o4 / r3 | **4.8x / 6.1x** |
| | k=3 free / dbc | 188 / 305 | 33 / 41 | r3o2 / r3 | **5.7x / 7.4x** |
| W7-X | k=0 free / dbc | 522 / 269 | 83 / 49 | r3 | **6.3x / 5.5x** |
| | k=1 free / dbc | 1660 / 858 | 321 / 176 | r3o4 / r3 | **5.2x / 4.9x** |
| | k=2 free / dbc | 1548 / 1008 | 322 / 212 | r3o4 / r3 | **4.8x / 4.8x** |
| | k=3 free / dbc | 307 / 439 | 49 / 75 | r3o2 / r3 | **6.3x / 5.9x** |

For scale, the production default needs 5,000-20,000 iterations at `k >= 1` and
fails outright on W7-X k=2 free.

The two knobs separate cleanly along the mechanism. **Dirichlet wants `r3` and
nothing else** -- the trace term is absent, so only the polar core needs help.
**Free wants outer rings at k>=2**, where they are worth 3-6x: k=3 free goes
200 -> 33 (toroid) and 275 -> 49 (W7-X) on two rings alone. `o2` suffices at k=3
and is actually better than `o4` there; `o4` buys a little more at k=1/2 on
W7-X. At k=1 free the toroid needs no rings at all once `bc_entry` is on
(442 -> 128 at 9.5 s of build), while W7-X still does.

Build: `r3` 28-38 s, `r3o2` 55-83 s, `r3o4` 96-125 s, against Jacobi's 3-6.5 s.
One-off, and at W7-X k=1 free it buys back ~1,300 iterations.

The advantage also **grows with refinement** -- W7-X k=1 dbc goes 2.5x / 2.8x /
3.2x over 8^3 / 12^3 / 16^3 for the bare separable atom -- because Jacobi roughly
doubles per refinement step while the atom grows about half as fast.

Mass, 8x16x8, against raw_kron: ~1.3x on the **W7-X vector masses** (k=1: 62 ->
48, k=2: 60 -> 47), a wash elsewhere, same build cost. That is exactly the case
recorded as the known failure -- the coupling half of the 31 -> 75 blowup, where
lumped-L/U block-SGS was tried and regressed.

---

## 9. Traps (do not repeat)

* **Weight the stiffnesses, not the masses** -- double counting makes the atom
  worse than a diagonal.
* **The curl-curl weight is indexed by the THIRD axis**, not the component:
  `d_t u_r` lands in `(curl u)_z` and carries `g_zz/J`. This one index was worth
  **20x** (k=1 toroid 1180 -> 57).
* **The polar cut belongs to the averaging, not the assembly.** Assembling 1-D
  matrices against a cut quadrature weight makes the radial mass singular. The
  core is excluded from the bulk by restricting the radial *window*.
* **Do not apply the component factor twice** -- once in the radial profiles and
  again in the diagonal sandwich. Invisible at k=0 (factor = 1), 5x at k=1; the
  signature looks exactly like "the method does not generalise".
* **Greville abscissae of a clamped basis include the endpoints**, where
  `det(DF) = 0`. Anything dividing by `J` there gets `inf` -> `NaN` -> an OOM
  kill. Quadrature points never touch the boundary, which is why this never
  shows up in assembly paths.
* **`F = M^d G A^-1 G^T M^d` is a Kronecker-PRODUCT factor, not a summand.**
  It is the exact 1-D weak factor and it carries the BC for free, so it is
  tempting -- but dropping it into the Kronecker SUM in place of a stiffness
  measured 425 vs 50 (toroid k=1 dbc) and 2691 vs 145 (W7-X). The undifferentiated
  axes would have to carry `M A^-1 M`, not `M`, and reconciling that with a
  scalar makes things worse, not better.
* **Do not benchmark a singular case in a script without deflation.** Four of the
  eight `(k, dbc)` pairs are singular at `eps = 0`; without projection they
  return residuals of 1e16 for *every* arm, which reads as a preconditioner
  failure and is not one.
* **Do not choose A/B cases that confound two variables.** Running `0t,1t,2f,3f`
  put dbc at k=0/1 and free at k=2/3, which made a boundary-condition effect look
  like a degree effect and produced a wrong mechanism that survived for hours.

---

## 10. Open

* **Promote to production?** Every case is now 3.5-9.3x on the unshifted
  operator, so the case is strong -- but the settings are BC- and
  geometry-dependent (`r3` under Dirichlet, outer rings under free at k>=2) and
  that needs a defaulting policy rather than a per-run flag.
* **The face weight.** `bc_entry` is limited by collapsing `w(1,theta,zeta)` to a
  scalar. Carrying an angular profile breaks the shared eigenbasis; whether a
  cheap two-term correction exists is untried, and it would retire outer rings.
* **Adopting the mass version redefines the operator** -- the mass preconditioner
  is the weak term's inner inverse, so swapping it changes `L_k` at `k >= 1` and
  the Laplacian baselines need redoing.
* **`compute_nullspaces` should verify before storing.** With the default
  `maxiter = 1000` its inner solves truncate and it silently returned a W7-X
  "harmonic" form with `||Lv||/||v|| = 2.6`.
* The **radially exact (modal)** bracket is more accurate and mesh-independent at
  k=0 but stores `O(n_zeta n_r^2)`; separable + rings recovers most of it. Not
  retested since `bc_entry` landed.
