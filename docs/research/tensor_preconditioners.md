# Tensor preconditioners for the mass and Hodge Laplacian

**Status: experimental**, all in `mrx/experimental/block_jacobi_laplacian.py`
and opt-in. Production is unchanged (raw_kron masses, closed-form Jacobi
Laplacians). §1-6 are the reasoning; §7-10 the design, results and traps.

Last measured 2026-08-20: h-sweep 6^3 -> 16^3 and degree sweep p=1,2,3 on both
geometries. The natural boundary term now has a closed form and costs nothing
(§6); it makes outer rings unnecessary on the toroid at every resolution. Two
p=1 normalisation bugs are fixed (§8.3). The one open failure is W7-X free BC at
k >= 1 (§8.2).

**2026-08-21, §6 revised.** The shipped `bc_entry="exact"` coefficient is
provably 8-14x too small: two metric factors were being applied implicitly and
each was cancelled by something fitted, so the wrong version passed its own
sweep (§6.3). The corrected form is `alpha = rho . mu_0 . <S> . <P>` — a
metric-FREE `1/h` times one transparent scalar (§6.1) times the cross-term
factor (§6.7). Implemented as `bc_entry="ibpd"`. It reproduces the tuned
toroid optimum with no tuning and takes W7-X k=3 free from 123 to 75, but
still loses at W7-X k=1/k=2 pending `rho`. DEFAULT REMAINS `exact`.

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

**Natural boundary conditions are a feature of the weak block, and they were the
whole story behind the free-BC cases.** `S_k = D_k^T M_{k+1} D_k` is the strong
part — `d` applied directly, no integration by parts, no boundary term. `W_k`
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
while the free ones lagged 4-6x at every degree with a weak block — k=0 excepted,
which is the control that rules out every other explanation.

This is a genuine natural condition (`u.n = 0` at k=1, `w x n = 0` at k=2),
enforced not by removing a DOF but by a **mesh-dependent penalty**: the boundary
distribution is projected into `V_{k-1}`, and its norm grows like `1/h`.

### 6.1 The closed form

Write the weak block's inner vector with integration by parts. With `G = D_0`,
`d_s` the coefficients of `P_{k-1}(d^* u)` (the STRONG adjoint derivative,
projected) and `E` the boundary pairing of the `V_k` trace against `V_{k-1}`:

```
G^T M_k u  =  -M_{k-1} d_s  +  E u
```

so the weak block splits into **three** terms, not two:

```
u^T W_k u = d_s^T M_{k-1} d_s  -  2 d_s^T E u  +  u^T E^T M_{k-1}^{-1} E u
            \_______________/     \_________/     \____________________/
             the honest Ktilde      CROSS (§6.7)    the rank-one update
```

The atom models the first with `Ktilde` and adds the third. The middle one is
real and is NOT small — see §6.7.

`E` factors as a Kronecker product — radial `dLam_i(1) Lam_j(1)`, angular
`M_t (x) M_z` — and when the `V_k` component's angular bases match those of the
`V_{k-1}` component it pairs with, **the angular masses cancel against the same
factors in `M_{k-1}^{-1}` and one scalar survives**. With `Lam(1) = e_last` for
a clamped spline, the reduction is: take each of the three matrices' weights at
`r = 1`, average over `theta,zeta`, and multiply.

| factor | weight |
| --- | --- |
| `E` | `m_k` = mass weight of `V_k` component `c` |
| `M_{k-1}^{-1}` | `1 / m_{k-1}`, the PARTNER's mass weight |
| `E^T` | `m_k` |

**Every metric factor belongs in that scalar, and nothing else does.** What is
left of `M_{k-1}^{-1}` is a purely logical number:

```
mu_0 = (M_r^{logical, unweighted})^{-1}[last, last]   ~ c(p)/h_last
```

with no `k`, no component and no geometry in it — verified bit-identical across
toroid / rot-ellipse / W7-X (`5.226074e+01` at n=8, `9.388387e+01` at n=12).

The metric scalar collapses. Without lumping it is `m_k^2/m_{k-1} = m_k g^rr`;
under `lumped="diag"` the component factor `w_comp = m_k/J` is carried OUTSIDE
as the `D^-1/2 . D^-1/2` sandwich, so it must come out of the face weight too,
leaving `m_k J / m_{k-1}`. Using `J^2 = prod g_aa` that is **the same
expression for every degree**:

| k | `m_k J / m_{k-1}` | |
| --- | --- | --- |
| 1 | `(g^rr J) J / J` | `= J g^rr` |
| 2 | `(g_cc/J) J / (g^{c'c'} J)` = `g_cc g_{c'c'} / J` | `= J g^rr` |
| 3 | `(1/J) J / (g_rr/J)` | `= J g^rr` |

and it factors into exactly the two geometric ingredients there are:

```
J g^rr  =  (J sqrt(g^rr))   x   (sqrt(g^rr))
            surface element      pullback of the normal component
```

So the whole coefficient is

```
alpha = mu_0 . <S>_{theta,zeta}(r=1) . <P>_{theta,zeta}(r=1) . rho
K_r  += alpha . e e^T ,      e = dLam_r(1)
```

with `rho` the cross-term factor of §6.7. It merges into `K_r` as a **rank-one
update**, so the shared eigenbasis, fast diagonalisation, cost and storage are
all unchanged.

**Averaging convention: product of averages, not average of products.** `<S><P>`
rather than `<S P>`; they differ by the covariance of the two factors over the
face. This is deliberately the OPPOSITE convention to `bundled_axis_profiles`,
whose bundling exists to keep `g^tt J ~ 1/r` integrable toward the AXIS — there
is no such singularity on the `r = 1` face, so that argument does not carry.
Measured: identical to 1.0000 on the toroid (where `g^rr` is constant on the
face, so the covariance vanishes identically — a free correctness check),
`0.945` on rot-ellipse, `0.912` on W7-X. Real but second order.

### 6.2 Which component, and which partner

The trace lives exactly where the component's RADIAL axis is a derivative axis.
The partner in `V_{k-1}` is **not** always the same index — at k=2 the cross
product in `int (w x n).tau` swaps the tangential components:

| k | component with a trace | pairs with |
| --- | --- | --- |
| 0 | none (`W_0 = 0`) | — |
| 1 | `c = r` (normal) | `V_0`, the only component |
| 2 | `c = theta, zeta` (tangential) | `V_1` at `3 - c` — the OTHER one |
| 3 | the single component | `V_2` at `c = r` |

This DERIVES rather than being asserted. On the logical cube the boundary term
is `oint tr(tau) ^ tr(*u)`; at k=2 that is a wedge of two 1-forms on the face,
`tau_t (*u)_z - tau_z (*u)_t`, which is the swap. The same computation gives the
E weight as `m_k`, the component's own mass weight, at every degree — the metric
raising and the surface measure always recombine into it.

Getting it wrong is not a perturbation: the two weights differ by `(R/a)^2` on a
toroid, and fixing it took toroid 12^3 k=2 free from 158 to 62 iterations.

### 6.3 Two hidden metric factors, and how they hid each other

**Superseded.** This section used to describe three corrections to the original
`direct` weight. Two of them were wrong, and they were wrong in a way that is
worth recording, because both passed their own validation.

The `direct` weight `m_k g^rr` was **right all along**. What was missing was
only `mu_0`. But `mu_0` was implemented as `(M_r^{(k-1)})^{-1}[last,last]` with
the partner's mass weight built INTO the matrix before inverting — so
`1/m_{k-1}` never appeared as a visible factor. To compensate for the resulting
stray `J`, the face weight was then "corrected" `g^rr -> sqrt(g^rr)` on
surface-element grounds. That reasoning double-counts: `u.n = sqrt(g^rr) u_r`
and `ds = J sqrt(g^rr) dtheta dzeta` MULTIPLY back to `g^rr J`, and the bare
`oint (u.n)^2 ds` carries one power of the measure where `E^T M^-1 E` carries
two and one inverse.

A second hidden factor sat underneath. `lumped="diag"` factors
`w(c,a) = w_comp . (g^aa J)` and builds the Kronecker factors from the `g^aa J`
half alone, returning `w_comp` as the `D` sandwich (`component_diagonal`). The
face weight was built from the full `m_k`, so it carried `w_comp` **twice**.
That is a factor of `g^rr = 9.0` on the toroid, `15.0` on rot-ellipse, `6.7` on
W7-X.

Net: the shipped `exact` coefficient is `w_comp . (1/rho)` off — 13.9x on the
toroid, 12.1x on W7-X. Measured against the exact 1-D round trip `F - Ktilde`,
`exact` is 8-14x too small.

**The dangerous part is not the wrong number, it is the fitted compensation.**
Both hidden factors were cancelled by something tuned empirically, so the wrong
version passed its sweep — §6.3 used to claim "the residual sweep peaks at x1
and the h-dependence is gone", and it did. An implicit normalisation
manufactures the evidence that hides it. **Rule: never apply a metric factor
inside a matrix that is later inverted, or as an outer diagonal sandwich. Every
term merged into an atom must state which normalisation it is in.**

The one correction from the old list that stands:

* **Axis.** The fallback branch lacked an `a == 0` guard and was adding entries
  on the PERIODIC theta/zeta axes, which have no boundary. Cost k=3 free 97
  against 87 for no correction at all, and perturbed the Dirichlet cases where
  the term must vanish identically.

**Regression invariant: every Dirichlet row must be identical across
`nobc` / `direct` / `exact` / `ibpd`.** The term does not exist there; if a
Dirichlet number moves, the boundary code is reaching somewhere it should not.
Holds in all runs to date.

### 6.4 At p = 1 the trace lives on the coefficient, not the value

At `p = 1` the derivative splines are degree 0 and the radial factor is a DG-0
jump stand-in (§7), but the boundary term is assembled the same way and needs no
special case — provided both are written in the same normalisation.
`DerivativeSpline` scales to unit **integral**, not unit height, so at degree 0
`D_i = 1_{cell i} / h_i` and `e = dLam(1) = (0, ..., 1/h_last)`. The rank-one
update `alpha e e^T` therefore penalises `(u_last / h_last)^2` — the boundary
*value*, which is what the surface integral asks for. The same convention is
what forces the `diag(1/h)` conjugation of the jump form itself; get one right
and the other follows. Worth 973 -> 706 at toroid 12^3 k=1 free on its own, and
it was simply absent before (the `bc_entry` block sat inside the `p >= 2`
branch).

### 6.5 What the closed form does not buy

**Its limit is the coefficient, not the structure.** Folding the term into the
Kronecker sum forces the face weight `w(1,theta,zeta)` down to a SCALAR. On a
toroid, where `J ~ R_0 + eps r cos theta` varies about +-33% across the face,
that scalar captures essentially all of it — `exact` beats the dense outer ring
`o2` in every one of the eight cases at 16^3 while building in half the time. On
W7-X the face weight varies far more, and at k>=1 **free** it captures little:
`exact` removes only ~25% of the Jacobi count there and the deficit widens with
refinement, while `o2` holds a clean ~4x (§8). Carrying the angular variation
would need `B_t`, `B_z` with real profiles — but then they are no longer
proportional to `M_t`, `M_z`, the eigenbasis is no longer shared, and fast
diagonalisation breaks. An exact OUTER ring is precisely the object that can
carry an arbitrary `theta`-`zeta` dependence on one radial index, which is why
it remains the only thing that works on W7-X free.

Confirmed directly by the dense spectrum probe: at k=1 free the preconditioned
operator has **44 outlier eigenvalues** out of 894 and condition 768, with the
extreme mode living **entirely** on the outer radial boundary; at k=1 dbc there
are **zero** outliers and condition 12.7.

### 6.6 Refuted

* **Hard `u.n = 0` as a replacement.** Penalty x1e4 gives 250 iterations at k=1
  free and 334 at k=2 — worse than no term at all. The atom wants the finite
  penalty, not an eliminated DOF. "Use the dbc scalar Laplacian on that
  component" is not the fix, even though the underlying BC reading is right.
* **A rank-1 fit of the face weight.** It breaks fast diagonalisation:
  `e e^T (x) Mt~ (x) Mz~` with `Mt~ != Mt` is not a summand of the Kronecker
  sum, and `V_t^T Mt~ V_t` is dense in the shared eigenbasis. Moot anyway — the
  partner fix (§6.2) closed the k=2 gap that motivated it.
* **`exact` + `o2` is exactly `o2`.** The dense outer ring already cuts those
  rows from the atom's radial window, so the two are alternatives, not additive.
  Measured identical to the iteration at every resolution on both geometries.

### 6.7 The cross term

§6.1 splits `W_k` into three pieces and the atom carries two of them. The
middle one,

```
-2 d_s^T E u  =  -2 oint_{r=1} (u.n) . P_0(div u)
```

is the interference between the boundary trace and the interior divergence.

**Why it exists.** The weak adjoint `delta_h u` is defined by
`<delta_h u, tau> = <u, d tau>`, and integration by parts says the continuous
object it represents is `-div u` PLUS a surface distribution on `r = 1`. `V_{k-1}`
cannot hold a delta, so it represents that distribution as a spike of finite
height `~ (u.n)/h` in the last cell. `W_k = ||delta_h u||^2` therefore contains
the spike squared (the term we add), the interior divergence squared (`Ktilde`),
and **the interference between them**.

**Why it is not negligible.** Near the boundary `div u ~ u_n/h`, so

```
E^T M^-1 E  ~  u_n^2 / h        cross  ~  |u_n| . |div u|  ~  u_n^2 / h
```

— the same order in `h`. It is not a lower-order boundary effect. Its sign is
negative for a mode with outflow (`u_n` and `div u` correlated), so it REDUCES
the penalty, which is the direction the measurements demand.

**What it does to the coefficient.** As a matrix the cross term is
`-(E^T Ddiv + Ddiv^T E)` with `d_s = Ddiv u`. Radially `E = e_last e^T`, so
`E^T Ddiv = e v^T` with `v = Ddiv^T e_last`, and symmetrised that is
`-(e v^T + v e^T)` — **rank two**, and not of the form `e e^T` unless `v || e`.
Split `v = gamma e + v_perp`:

```
F - Ktilde  =  (alpha - 2 gamma) e e^T  -  (e v_perp^T + v_perp e^T)
```

So the DOMINANT effect is not a new structure, it is a **shrunk coefficient**
along the same `e e^T` the atom already uses. Measured on the exact 1-D round
trip (`scripts/debug/bc_alpha_compare.py`, `roundtrip_reference`):

| geometry | n | `s2/s1` | `off_frac` | `rho = c_star / alpha` |
| --- | --- | --- | --- | --- |
| toroid | 8 | 0.024 | 0.208 | 0.647 |
| toroid | 12 | 0.023 | 0.214 | 0.629 |
| w7x | 12 | 0.061 | 0.282 | 0.396 |

`s2/s1 ~ 0.02-0.06` says the correction really is rank one — consistent with
`v_perp` being small. `off_frac ~ 0.2-0.3` is the `v_perp` residue, the rank-two
signature. And `rho ~ 0.63` on the toroid is **h-independent**, which is what
makes it a coefficient rather than a mesh artefact.

**How to account for it.** Three options, in increasing crudeness:

1. **Take `c_star` directly.** `F - Ktilde` projected onto `e e^T` is
   `e^T (F - Ktilde) e / (e^T e)^2`, and it contains the cross term already. All
   objects are `n_r x n_r` and the atom assembles them anyway, so the cost is one
   small solve. The catch: `F` is built from radial PROFILES, so this re-imports
   the bundling inconsistency the face-evaluated scalar was designed to avoid —
   which is why the old `bc_entry` fallback that did exactly this measured worse.

2. **Hybrid — take the metric from the face and only `rho` from the round trip.**
   `rho` is DIMENSIONLESS, and numerator and denominator use the same profiles,
   so the bundling cancels out of the ratio. Then
   `alpha = rho . mu_0 . <S> . <P>` keeps every metric factor transparent and
   picks up the cross term from a quantity that is neither a fit nor a knob.
   This is the recommended route.

3. **A fixed `rho`.** Only defensible if the sweep shows one number across
   geometry and degree. It does not look like one — 0.63 vs 0.40 — so this is a
   fallback, not a design.

Note what this does NOT change: the cross term is a scalar multiplying the same
rank-one update, so fast diagonalisation, the shared eigenbasis, cost and storage
are untouched. It is a coefficient, not a structure.

## 7. The design, concretely

**Shape** (both operators): densely-probed **core** (polar ring + `extra_rings`)
plus a **separable bulk** on the remaining radial window, uncoupled.

**Laplacian.** One three-term Kronecker sum per component, fast-diagonalised,
masses unweighted. An axis is a *derivative axis* where the component's basis is
a derivative spline — k=0 none, k=1 axis `c`, k=2 all but `c`, k=3 all three —
and gets the stiffness of the derivative splines. At `p = 1` those splines are
degree 0 and have no stiffness, so a DG-0 **jump seminorm** stands in — the
finite-volume form `D^T diag(t) D` with harmonic-mean transmissibilities, which
has the same constant kernel and drops into the same Kronecker sum. It must be
written on the cell VALUES: the derivative basis is normalised to unit integral,
`D_i = 1_{cell i} / h_i`, so the jump form has to be conjugated by `diag(1/h)`
(§9). All weights follow one formula,
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
| natural-BC face | `bc_entry = "exact"` | ON by default (§6); free of charge, exactly zero under Dirichlet and at k=0. Coefficient known wrong by 8-14x — see `"ibpd"` (§6.1) and `rho` (§6.7) |
| natural-BC face, strong angular variation | `outer_rings = 2` | W7-X free at k>=1 only — on the toroid `exact` has retired it |

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

All numbers: unshifted (`eps = 0`), tol 1e-10, deflated CG on the four singular
`(k, dbc)` pairs, `extra_rings = 3`, `bc_entry = "exact"` unless stated.
Reproduce with `scripts/debug/verify_block_jacobi.py`, e.g.

```
SCRIPT=scripts/debug/verify_block_jacobi.py \
  ARGS="--geometry toroid --ns 12,24,12 --p 3 \
        --arms jacobi,blockjac_r3exact,blockjac_r3o2" \
  JOB_NAME=v OUTSUB=v bash slurm/job_diag_run.sh
```

Arm suffixes: `rN` inner rings, `oN` outer rings, `exact`/`face`/`nobc` boundary
mode, `a2d` 2-D ring atoms, `modal` modal-radial, `rt` round-trip
derivative-axis factor. Diagnostic only: `bcsN` boundary penalty x N
(`MRX_BJ_BC_SCALE`), `d0sN` degree-0 stiffness x N/100 (`MRX_BJ_D0_SCALE`),
`d0old` the pre-fix coefficient-basis jump form (`MRX_BJ_D0_FORM=coef`).
The p=1 factor comparison in §8.3 is `scripts/debug/diag_p1_factors.py`; the
harmonic-form measurements in §8.6 are `scripts/debug/diag_nullspace_polish.py`.

### 8.1 Toroid, p = 3: the closed-form boundary term is the whole design

`jacobi -> exact`, free BC, across a factor of 2.7 in mesh size:

| | 6x12x6 | 8x16x8 | 12x24x12 | 16x32x16 |
| --- | --- | --- | --- | --- |
| k=0 free | 214 -> **31** | 297 -> **32** | 399 -> **43** | 460 -> **52** |
| k=1 free | 249 -> **49** | 315 -> **57** | 442 -> **77** | 597 -> **94** |
| k=2 free | 174 -> **43** | 212 -> **45** | 348 -> **62** | 499 -> **75** |
| k=3 free | 84 -> **25** | 113 -> **29** | 188 -> **36** | 299 -> **42** |

Not h-independent — the atom grows about `n^0.6` — but the ratio to Jacobi
improves monotonically with refinement (k=1: 5.1x -> 6.4x), because Jacobi grows
roughly twice as fast. Dirichlet is the same picture one step stronger
(12^3: 234/376/470/306 -> 32/62/77/41).

**`exact` retires the dense outer ring on the toroid.** At 16^3 it wins all eight
cases against `o2`, including the two (k=3 free, k=3 dbc) where `o2` was ahead at
lower resolution — that lead was a low-resolution artifact. It also builds in
half the time (k=2 free: 66 s vs 145 s).

### 8.2 W7-X, p = 3: free BC at k>=1 is the open case

`jacobi / exact / o2`:

| | 8x16x8 | 12x24x12 | 16x32x16 |
| --- | --- | --- | --- |
| k=1 free | 1064 / 608 / **223** | 1668 / 1209 / **390** | 2279 / 1784 / **562** |
| k=2 free | 893 / 578 / **201** | 1548 / 1216 / **367** | 2324 / 1860 / **551** |
| k=1 dbc | 519 / **117** / 126 | 859 / **177** / 191 | 1342 / **231** / 254 |
| k=2 dbc | 543 / **145** / 171 | 1010 / **212** / 271 | 1546 / **267** / 369 |
| k=0, k=3 (both BCs) | `exact` wins | `exact` wins | `exact` wins |

The scalar face weight removes only ~25% of the Jacobi count at k>=1 free and the
deficit **widens** with refinement, while `o2` holds ~4x at every resolution.
This is one failure mode, not two: the earlier reading that only k=1 free was
affected was a resolution artifact — k=2 free behaves identically. What the two
share is a nonzero trace under a free condition (normal at k=1, tangential at
k=2) reduced to a single scalar, on the geometry where the orthogonal-metric
assumption behind both the weight formula and the face measure is worst
(`g_tz` is W7-X's largest off-diagonal). `exacto2` is bit-identical to `o2` everywhere (§6.6).

### 8.3 Degree sweep, and the p = 1 fix

At 12x24x12, `p = 2` behaves like `p = 3` (toroid k=1 free 361 -> 70, k=2 free
275 -> 56). `p = 1` did not: the atom was 2.5-6x **worse than Jacobi** for every
`k >= 1` on both geometries, while `k = 0` was untouched — the signature of the
degree-0 stand-in, which is used exactly on derivative axes and only at `p = 1`.

Two bugs, both in the normalisation of that stand-in, both now fixed (§6.4, §9):
the jump form was assembled on D-spline **coefficients** rather than values
(under-scaling it by `h^2`, so no constant repairs it and the damage grows with
refinement), and the natural-BC block never ran at degree 0 at all.

`jacobi / fixed / before`, `p = 1`:

| | toroid 8x16x8 | toroid 12x24x12 | W7-X 12x24x12 |
| --- | --- | --- | --- |
| k=1 free | 236 / **79** / 334 | 395 / **98** / 706 | 1260 / **588** / 1852 |
| k=1 dbc | 186 / **64** / 267 | 292 / **80** / 571 | 623 / **182** / 1129 |
| k=2 free | 199 / **65** / 436 | 365 / **81** / 1069 | 1336 / **597** / 2744 |
| k=2 dbc | 295 / **92** / 667 | 518 / **107** / 1450 | 1094 / **255** / 2202 |
| k=3 free | 115 / **52** / 221 | 237 / **65** / 465 | 397 / **124** / 852 |
| k=3 dbc | 200 / **66** / 218 | 392 / **75** / 346 | 724 / **129** / 716 |

(`before` here already includes the degree-0 boundary term, so the column
isolates the `h^2` scaling alone; the pre-fix default was worse still — 973 at
toroid 12^3 k=1 free.) `p = 1` now lands in the same family as `p = 3`
(toroid 12^3 k=1 free: 98 at p=1 vs 77 at p=3) and is 2.1-5.6x over Jacobi,
weakest on W7-X free exactly as at p=3 (§8.2). A
`p = 2` control reproduces the earlier numbers exactly — nothing at `p >= 2`
moves.

The 1-D spectra say the same thing quantitatively, and independently of any
solve. Median generalized eigenvalue ratio of the stand-in against the exact
round-trip factor `F = M^d G A^-1 G^T M^d`, per axis, toroid 8x16x8
(`diag_p1_factors.py`):

| | radial | theta | zeta |
| --- | --- | --- | --- |
| `p=1` jump / F, as assembled | 0.15 | 0.015 | 0.0017 |
| `p=1` jump / F, after `diag(1/h)` | **7.4** | **3.7** | **0.11** |
| `p>=2` honest / F (the working case) | 10.9 | 6.5 | 0.21 |

Off by 1-3 orders and by a different amount per axis before; within a factor of
1.5-2 of where the `p >= 2` factor sits after, with the axis ordering preserved.
That is the whole bug: the missing `1/h^2` is per axis, so it is not a constant
and it is not the same constant twice. The round-trip
factor itself is NOT the fix — run as the atom's derivative-axis factor
(`ktilde_mode="roundtrip"`) it is worse than the repaired jump form by 5-8x
(toroid 8^3 k=1 free: 600 vs 79).

### 8.4 Mass

8x16x8, against raw_kron: ~1.3x on the **W7-X vector masses** (k=1: 62 -> 48,
k=2: 60 -> 47), a wash elsewhere, same build cost. That is exactly the case
recorded as the known failure — the coupling half of the 31 -> 75 blowup, where
lumped-L/U block-SGS was tried and regressed.

### 8.5 Cost

Build: `r3` 28-38 s, `r3o2` 55-83 s, `r3o4` 96-125 s, against Jacobi's 3-6.5 s.
One-off. `bc_entry="exact"` is free — one 1-D solve for
`(M_r^{(k-1)})^{-1}[last,last]` and one face average per component — which is why
it is preferable to `o2` wherever the two are close.

### 8.6 Harmonic forms and deflation

The four singular cases are deflated against harmonic forms from
`mrx.nullspace.compute_nullspaces`, and on W7-X the k=1 free one **degrades with
refinement**: Rayleigh quotient 1.2e-13 / 9.3e-08 / 1.6e-06 at 8^3 / 12^3 / 16^3,
against 1e-11 or better for every other `(k, dbc)` pair.

Cause: that construction strips the coexact part with an `L_2` **free** solve —
the very case in §8.2 — and it runs to `maxiter` unconverged at every
resolution. Tightening `seq.tol` (1e-13 -> 1e-14) changes nothing, and loosening
it to 1e-6 changes nothing either: the cap binds, not the tolerance. The quotient
is the same against `L_k` and against the `L_approx` the solves actually use, so
this is not a raw_kron artifact — fixing it improves the deflation as well as the
physics.

Two cures, both measured on the W7-X k=1 free vector:

| | 8x16x8 | 12x24x12 | 16x32x16 |
| --- | --- | --- | --- |
| strip 1 (today's direct route) | 1.2e-13 | 9.3e-08 | 1.6e-06 |
| + strip 2 | 1.0e-23 | 5.0e-13 | 1.2e-10 |
| + strip 3 | 9.8e-28 | 8.4e-18 | 1.8e-14 |
| cost per strip | 20 s | 48 s | 129 s |
| inverse iteration, 2 sweeps | **7.2e-24** | **3.1e-24** | **2.8e-24** |
| its cost | 11 s | 45 s | 191 s |

**Iterative refinement works** — the strip is a projection, so repeating it fixes
the result even though the inner solve never converges — but its accuracy is
h-dependent and it needs three passes at 16^3 to reach 1e-14.

**Inverse iteration wins at every resolution.** Seeded with the direct vector
(`find_nullspace_vectors(..., x0s=[v], inner_tol=1e-8)`) it takes 2 sweeps and
lands at ~3e-24 *independently of h* — the floor is set by `inner_tol`, exactly
as that function documents — for less than the cost of one extra strip at 12^3
and half the cost of the three strips needed at 16^3. `inner_tol=1e-6` is 4
orders looser (~2e-20) at the same price, so there is no reason to use it here.

---

## 9. Traps (do not repeat)

* **Weight the stiffnesses, not the masses** -- double counting makes the atom
  worse than a diagonal.
* **Coefficients are not values.** `DerivativeSpline` normalises to unit
  INTEGRAL, so at `p = 1` the basis is `D_i = 1_{cell i} / h_i` and a DOF is
  `h_i` times the function value it stands for. Any factor written as a
  functional of the values -- the DG-0 jump form, the boundary trace -- must be
  conjugated by `diag(1/h)` before it enters the atom. Skipping it under-scales
  by `h^2`, which is invisible in a single-resolution A/B (it looks like a
  tunable constant) and gets worse the finer you go. Cost: the whole of `p = 1`
  at every `k >= 1`, on both geometries (§8.3).
* **A missing case in a degree branch is silent.** The natural-BC block sat
  inside the `p >= 2` arm of the same `if`, so `p = 1` quietly ran with no
  boundary term at all while `bc_entry="exact"` was reported as on.
* **The k=2 boundary partner is the OTHER tangential component** (`3 - c`), not
  `c`: the cross product in `int (w x n).tau` swaps them, and the angular
  cancellation that makes the closed form a scalar only holds for that partner.
  On a toroid the two weights differ by `(R/a)^2` -- 158 vs 62 iterations.
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
* **Do not rank two variants at one resolution.** `o2` beat `exact` at k=3 on the
  toroid at 8^3 and 12^3 and lost at 16^3; on W7-X free the ordering is the
  reverse and the gap grows. Anything that claims a winner needs at least two
  mesh sizes.
* **A residual tolerance is not an accuracy.** `compute_nullspaces` asks its
  inner solves for 1e-13 and they exit at `maxiter` instead; the stored vector is
  then wrong at 1e-3 with nothing in the log to say so. Check the solver info,
  and measure the Rayleigh quotient rather than `||Lv||`, whose scale drifts with
  `h`.

---

## 10. Open

* **Promote to production?** On the toroid the design is now settled and cheap:
  `r3` + `bc_entry="exact"` wins every case at every resolution with no ring
  knob at all, and the advantage grows with refinement. What blocks a default is
  W7-X free at `k >= 1`, where `outer_rings = 2` is still the only thing that
  works and costs 2x the build -- so the policy is geometry-dependent, which is
  what a default should not be.
* **W7-X free at k >= 1 (the one real gap).** `exact` removes ~25% and the deficit
  widens with `h` (§8.2). Prime suspect is the orthogonal-metric assumption
  shared by the weight formula `w(k,c,a) = [mass weight] . g^{aa}` and the face
  measure: `g_tz` is W7-X's largest off-diagonal and both assume it away. The
  same case is flagged in `greville-k1-hx-status`.
* **The face weight.** The closed form is limited by collapsing
  `w(1,theta,zeta)` to a scalar (§6.5). Carrying an angular profile breaks the
  shared eigenbasis. `bc_entry="woodbury"` is the one route that could carry a
  non-scalar weight -- `U^T A^-1 U` IS diagonal in the angular eigenbasis, so it
  is affordable -- and it was measured as adding nothing, but that was with the
  OLD, wrong `alpha`. Worth one re-test now that the weight is right.
* **Harmonic forms: adopt the inverse-iteration polish.** §8.6 settles which
  cure, not where it lives. The direct route stays as the seed -- it is what
  makes 2 sweeps enough -- and `compute_nullspaces` should hand its result to
  `find_nullspace_vectors` and store the polished vector, or at minimum measure
  the Rayleigh quotient and refuse to store a bad one. Untested lever if the
  direct route is kept as-is: its cost is set by `maxiter`, not `tol` (the `L_2`
  free solve never converges), so 3 strips at a few thousand iterations each
  should beat 1 strip at 20,000.
* **Adopting the mass version redefines the operator** -- the mass preconditioner
  is the weak term's inner inverse, so swapping it changes `L_k` at `k >= 1` and
  the Laplacian baselines need redoing.
* **2-D ring atoms** (`core_mode="atom2d"`) work for INNER rings -- matching the
  dense probe at ~1.8x less build, with the gap shrinking as the mesh refines --
  and fail for outer rings (toroid k=3 free 65 vs 24, W7-X k=2 free 616 vs 200).
  The outer ring's value is nonlocal radial coupling (Steklov/DtN); no separable
  factor carries it.
* The **radially exact (modal)** bracket is more accurate and mesh-independent at
  k=0 but stores `O(n_zeta n_r^2)`; separable + rings recovers most of it. Not
  retested since `bc_entry="exact"` landed.
