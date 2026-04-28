# Surgery Schur Refactor Plan

This note records the intended refactor of the mass-preconditioner stack so
that the dense extracted-space surgery elimination is no longer hard-wired to
the `tensor` bulk model.

The core design change is:

- the bulk approximation choice and the Schur eliminations should be separate
  layers,
- the outer extracted-space surgery Schur should be usable with any bulk
  preconditioner,
- and for `k = 1` and `k = 2` the current inner block Schur should be treated
  as another composable inner layer rather than as a tensor-specific detail.

## 0. Current Status

As of the current refactor state, the main architectural split is in place.

Implemented:

- `MassPreconditionerSpec` now carries `kind` and `surgery_schur` separately.
- `MassPreconditionerSpec` also now carries the stored polynomial-generation
  hyperparameters: Richardson keeps the power-iteration/damping controls, and
  Chebyshev now stores guarded-Lanczos tuning controls explicitly.
- mass-preconditioner storage is split into separate Jacobi, surgery, tensor,
  and Kronecker payloads.
- the outer surgery Schur wrapper is now built generically from surgery data
  plus a bulk apply, rather than being hard-wired to tensor storage.
- the tensor preconditioner payloads for `k = 0`, `k = 1`, and `k = 2` now
  store only the bulk/local tensor factors; surgery data lives separately.
- the operator-level mass-preconditioner builder supports surgery-wrapped
  `jacobi`, `richardson`, `chebyshev`, and `tensor` bulk models, but the only
  currently admitted inner smoother under Schur is terminal `tensor`.
- for `k = 0`, the public operator-level interface has now been narrowed to a
  stable subset: bulk-only `none|jacobi|richardson|chebyshev`,
  `none/schur/tensor`, the legacy alias `tensor -> none/schur/tensor`, and
  `richardson/schur/tensor`.
- the `k = 0` public interface now rejects `jacobi/schur/*`,
  `chebyshev/schur/*`, any inner Jacobi under Schur, and `tensor/no-schur`.
- for `k = 1`, the public operator-level interface now uses that same reduced
  admissible surface, but the implementation interprets Schur recursively:
  outer surgery Schur, then the inner RT Schur, then terminal leaf blocks.
- for `k = 3`, the operator layer now accepts the uniform
  `outer/schur/inner`-shaped spec but ignores everything except the terminal
  inner choice; i.e. outer and Schur structure are normalized away there.
- for `k = 2`, the public operator-level interface now matches `k = 1`: the
  same reduced `outer/schur/inner` surface is admitted, while the
  implementation interprets the bulk under the outer surgery wrapper as three
  terminal scalar leaf blocks `r_bulk`, `theta`, and `zeta`.
- for transition compatibility, plain `kind='tensor'` is interpreted as the
  legacy compound route, i.e. `kind='tensor', surgery_schur=True`.
- sequence-level preconditioner ownership has been removed again; active
  consumers now read preconditioner data from `seq.get_operators()`.
- `DeRhamSequence.assemble_all_sparse(include_preconditioners=False)` is in
  place for sparse-only assembly.
- `SequenceOperators` now has an optional dense cache, and
  `DeRhamSequence.assemble_all_dense()` fills it after sparse assembly without
  eager preconditioners.
- the interactive `k = 0` validation script now uses explicit
  `outer/schur/inner` terminology in its admissibility table and now matches
  the reduced public k = 0 interface exactly.
- the interactive validation script now compares `k = 2` mass inverse
  preconditioners alongside `k = 0`, `k = 1`, and `k = 3`.
- the operator layer now rejects all non-tensor inners under Schur for
  `k = 0`, `k = 1`, and `k = 2`; the brief experimental reopening for
  `k = 1` did not converge and was removed again.
- the interactive mass benchmark/demo catalog has been brought back into line
  with that rule, i.e. Schur inners are now shown as tensor-only there as
  well.
- the sparse-operator rename is now in place on the active operator path:
  `m{k}`, `d{k}`, `g{k}`, `grad_grad`, `curl_curl`, `div_div`, `dd{k}_diaginv`,
  and the projection blocks no longer carry the `_sp` suffix.

Mass-side status is now comparatively clean, but the Hodge/Laplace side is not
yet at the same architectural endpoint:

- mass Jacobi / surgery / tensor payloads now live under
  `operators.mass_preconds`, while the Hodge-side Jacobi data still lives
  directly on `SequenceOperators` as `dd{k}_diaginv` and
  `dd{k}_diaginv_dbc`.
- the `k = 0` tensor Hodge payload also still lives directly on
  `SequenceOperators` as `k0_tensor_hodge_precond`, together with the FD scale
  data `dd0_fd_scale_K` and `dd3_fd_scale_K`.
- for `k >= 1`, the Hodge/Laplace solve path already has an intrinsic
  saddle-point / Schur-complement structure through the derivative coupling;
  this is separate from the extracted-space surgery Schur used on the mass
  side and should be treated as such in the next refactor step.
- after the mass-side restriction to tensor-only Schur inners, the lower-block
  and `schur.inner` mass routes used by the Hodge/Laplace saddle solver now
  also effectively inherit that tensor-only-inner policy.

Validated so far:

- solver-facing regression tests were moved into `test/test_operators.py` and
  extended with direct inverse-mass and warm-start checks.
- an interactive validation script now exists at
  `scripts/interactive/solver_preconditioner_validation.py`.
- user-run validation confirmed the production `k = 3` Hodge solve path and
  the default diffusion solves after the refactor.
- `k = 0` comparison runs showed that `richardson/schur`,
  `none/schur/tensor`, and `richardson/schur/tensor` converge cleanly on the
  narrowed admissible set.
- the interactive comparison script now prints each concrete `k = 0`
  comparison case as it finishes and then emits the final summary table.
- `k = 1` inner-Jacobi Schur experiments did not converge in the interactive
  comparison script, so that route remains outside the admitted public
  interface.

Important empirical result:

- the earlier implicitly-defined `jacobi/schur` path corresponds to the
  Jacobi-inner variant of the new `outer/schur/inner` notation. In the
  present `k = 0` construction that Jacobi-inner combination is not SPD and is
  therefore not a safe CG preconditioner. In practice it behaved as an invalid
  combination, not merely a weak one.
- outer Jacobi on top of a Schur-preconditioned operator was also found to be
  unhelpful: it builds Jacobi on the non-local left-preconditioned operator
  `P A` rather than on a local matrix approximation, and the resulting route
  was removed from the public k = 0 interface.
- `richardson/schur/tensor` is still an interesting algorithmic idea because
  it can drive the measured outer iteration counts very low, but the scaling is
  unfavorable. The reason is that one reported outer Richardson step is not a
  cheap local relaxation: each step applies the full Schur-wrapped operator,
  and that Schur apply already contains a terminal tensor inverse on the bulk
  block. So the apparent iteration win is bought by a more expensive operator
  application, and the total work grows poorly once the problem size increases
  or the same nesting is used recursively in `k = 1` and `k = 2`.
- the interactive mass benchmark/demo should now be read as a tuning sweep,
  not as the public-interface contract. In particular, it currently compares
  tensor ranks `1, 2, 4` and Richardson/Chebyshev step counts `2, 4, 8` in
  order to understand empirical behavior. Those variants are useful for
  benchmarking, but they should not be confused with the smaller mass-side
  public policy described above.
- the same conclusion now applies to the recursive `k = 1` and `k = 2`
  operator paths: inner Jacobi under Schur is kept disallowed there as well,
  rather than being treated as a merely weak option.

Still pending or intentionally deferred:

- decide whether the legacy `kind='tensor'` shorthand should remain public long
  term or whether the explicit `none/schur/tensor` spelling should replace it.
- propagate the settled k = 0 public interface story into any remaining docs or
  helper utilities that still present a broader comparison space.
- decide whether the shared reduced public interface for `k = 1` and `k = 2`
  should eventually be documented explicitly as the same admissible set as
  `k = 0`, or whether it should continue to be described more conceptually.
- clean up any remaining legacy preconditioner writes outside the new operator
  assembly path.
- add a dedicated runtime smoke test for the new dense-cache convenience path
  if that path is expected to be used routinely.
- refactor the Hodge/Laplace preconditioner payloads so that they are no
  longer stored ad hoc on `SequenceOperators`.
- decide whether the scalar Hodge polynomial paths should remain public at all,
  or whether the same anti-nested-iteration policy used on the mass side
  should narrow them further.

## 0.5 Follow-On Plan For Hodge / Laplace Preconditioners

The next stage should not be expressed as another mass-surgery story. The
important distinction is:

- on the mass side, `surgery_schur` is an optional extracted-space wrapper;
- on the Hodge/Laplace side for `k >= 1`, the Schur complement is part of the
  operator itself.

So the Laplace/Hodge refactor should be phrased in saddle-point terms, not by
trying to reuse the mass `surgery_schur` language.

Current structural split by degree:

- `k = 0`: scalar SPD / singular scalar problem, no intrinsic saddle Schur.
- `k = 1, 2, 3`: saddle-point problem with a lower `(k-1)`-form mass block and
  an upper Schur operator
  `K_k + eps M_k + D_{k-1} M_{k-1}^{-1} D_{k-1}^T`.

Planned refactor steps:

1. Introduce a dedicated Hodge/Laplace preconditioner storage bundle.

   This should hold at least:

   - the Hodge Jacobi diagonals `dd{k}_diaginv` / `dd{k}_diaginv_dbc`,
   - the `k = 0` tensor Hodge payload,
   - the FD scale data needed by the tensor Hodge routes,
   - and any future harmonic-coarse payloads that belong to the Hodge solver
     rather than to the bare assembled operators.

   The goal is the same separation already achieved on the mass side:

   - assembled operators stay in `SequenceOperators`,
   - solver payloads move into a dedicated preconditioner container.

2. Keep `k = 0` separate from `k >= 1` in both interface and implementation.

   For `k = 0`, there is no PDE-level Schur complement, so the scalar Hodge
   builder should remain a direct SPD/singular-scalar path. That path can then
   be cleaned up on its own terms without being forced into the saddle API.

3. Treat `k = 1, 2, 3` Laplace/Hodge solves as intrinsically saddle-point.

   The public and internal picture should be:

   - lower-block preconditioner for the `(k-1)`-form mass block,
   - `schur.inner` approximation for the embedded inverse mass
     `M_{k-1}^{-1}` inside the Schur operator,
   - `schur.outer` preconditioner for the upper Schur operator itself,
   - optional coupled completion.

   In other words, the Schur decomposition is not an optional wrapper here; it
   is the natural structure of the operator for `k >= 1`.

4. Import the current mass-side scaling policy into the Hodge/Laplace Schur
   inner slot.

   The present empirical conclusion is that nested iterative inners scale very
   badly. So the default and, for now, the only admitted `schur.inner` should
   remain terminal tensor mass inversion. That matches what the current code is
   already moving toward after the mass-side restriction.

5. Refine the outer Schur policy degree-by-degree.

   The likely short-term target is:

   - `k = 1, 2, 3`: keep `schur.outer` as a local approximation to the upper
     Schur operator, with Jacobi as the stable baseline and polynomial outers
     only if they are genuinely useful on top of a tensor inner;
   - `k = 0`: keep the scalar Hodge route separate, with its own admissible
     set.

6. Only after that storage and interface split, revisit the naming and public
   documentation.

   In particular, the Hodge/Laplace note should make the difference explicit:

   - mass preconditioners talk about optional surgery Schur wrappers,
   - Hodge/Laplace preconditioners for `k >= 1` talk about intrinsic saddle
     blocks and the induced Schur complement.

Short-term demo / validation rollout before the full refactor:

- the interactive Laplace demo now computes harmonic nullspaces again on the
  default built operator bundle and propagates those stored vectors to the
  rank-specific benchmark bundles.
- nullspace inverse iteration keeps Richardson as the shifted preconditioner
  path (`richardson-2`, with tensor on the inner/schur-inner slot) and
  explicitly rejects Chebyshev during nullspace construction.
- instead, benchmark only invertible operator settings:
  - `k = 0`: `dirichlet=True`,
  - `k = 1`: `dirichlet=True`,
  - `k = 2`: `dirichlet=False`,
  - `k = 3`: `dirichlet=False`.
- begin with the `k = 0` scalar Hodge/Laplace path only, so the demo can be
  brought back into sync with the current interface before widening again to
  the saddle cases `k = 1, 2, 3`.
- for the upper Schur block once the saddle demo is reopened, include the
  currently discussed outer options:
  - `jacobi`,
  - `richardson`,
  - `chebyshev`,
  - `exact_jacobi` as the setup-heavy probed competitor,
  with the lower block supplied by the user as a normal `MassPreconditionerSpec`
  and with `exact_jacobi` probing the Schur operator using that same lower
  block preconditioner.

## 1. Goal

Today `kind='tensor'` means two things at once:

- do a dense Schur elimination on the extracted surgery block,
- use tensor block models on the bulk.

That coupling is too strong. The surgery Schur is an extracted-space wrapper,
not a tensor method.

The refactor goal is to make the structure explicit:

- choose a bulk model,
- optionally wrap it in an outer surgery Schur,
- and, where needed, optionally wrap parts of the bulk in an inner block Schur.

## 2. Target Abstraction

At the mass-preconditioner specification level, the outer surgery elimination
should become a generic flag:

- `kind`: selects the bulk model,
- `surgery_schur`: if enabled, wrap the chosen bulk model in the dense outer
  surgery Schur.

For `k = 1` and `k = 2`, there is another structural layer inside the bulk:

- `k = 1`: outer surgery Schur, then inner block Schur on the coupled
  `(r, theta_bulk)` block, then block-local bulk models,
- `k = 2`: outer surgery Schur, then the block-local bulk models; in the long
  run this should still be expressed as a separate inner layer even where the
  current inner bulk is simpler than `k = 1`.

So the compositional picture is:

- outer surgery Schur,
- inner block Schur,
- block-local model.

This means combinations such as `schur-schur-jacobi` are valid in principle:

- outer extracted surgery Schur,
- inner block Schur,
- Jacobi as the block-local model.

The only ordering constraint is that the outer surgery Schur must sit outside
the inner block Schur. One can only do the inner block elimination after the
outer surgery rows have already been separated from the bulk.

For the implementation, the first rollout will be simpler than the full
abstraction: for `k = 1` and `k = 2` we will support either

- all Schur layers enabled, or
- no Schur layers enabled.

So the code will initially move as one bundle between

- `schur-schur-<bulk-model>`, and
- `<bulk-model>`.

The finer separation between outer surgery Schur and inner block Schur remains
the correct long-run abstraction, but it will not be independently selectable
in the first cut.

## 3. Structural Interpretation By Degree

### `k = 0`

Current structure:

- outer scalar surgery/core Schur,
- one scalar bulk model.

Target structure:

- optional outer surgery Schur,
- one bulk model chosen by `kind`.

No inner block Schur layer is needed.

### `k = 1`

Current structure:

- outer surgery Schur,
- inner block Schur on the coupled `(r, theta_bulk)` RT block,
- tensor models for `arr`, `theta`, and `zeta`.

Target structure:

- optional outer surgery Schur,
- optional inner block Schur,
- block-local model for each diagonal bulk block.

This is the clearest case where `tensor` currently fuses all three layers.

### `k = 2`

Current structure:

- outer surgery Schur,
- three diagonal bulk blocks `r_bulk`, `theta`, `zeta`.

Target structure:

- optional outer surgery Schur,
- optional inner block wrapper layer,
- block-local model for each diagonal bulk block.

Even though the present `k = 2` bulk is simpler than `k = 1`, the refactor
should preserve the same abstraction boundary: outer extracted wrapper first,
then inner block structure, then local bulk models.

### `k = 3`

Current structure:

- no surgery,
- one scalar bulk model.

Target structure remains unchanged apart from using the same generic API.

## 4. Data-Model Refactor

The present tensor data objects store both the outer Schur data and the bulk
tensor factors. That prevents the outer surgery Schur from being reused by
non-tensor bulk models.

The data should be split into separate payloads.

### Outer surgery payload

Store independently of the bulk model:

- `surgery_indices`,
- `bulk_indices`,
- `A_sb`,
- `A_bs`,
- `S^{-1}` for the chosen outer bulk approximation.

This payload is attached to the extracted degree/BC case, not to `tensor`
specifically.

### Inner block payload

For `k = 1` and eventually `k = 2`, store separately:

- the indices of the inner coupled blocks,
- the inner off-diagonal coupling blocks,
- the corresponding inner Schur inverse.

This payload belongs to the inner block structure, not to the outer surgery
layer and not to the block-local model itself.

### Bulk-model payload

Store independently per model kind:

- Jacobi data,
- tensor block factors,
- any future fast-diagonalisation or low-rank block model data,
- polynomial-iteration metadata if we later decide to materialize that layer
  separately.

## 5. Apply Pipeline

The apply path should become a composition of wrappers.

### Outer surgery wrapper

Given a bulk apply `B^{-1}` on the extracted bulk block,

$$
A =
\begin{pmatrix}
A_{ss} & A_{sb} \\
A_{bs} & A_{bb}
\end{pmatrix},
$$

define the outer Schur preconditioner by

$$
S = A_{ss} - A_{sb} B^{-1} A_{bs}.
$$

Then apply

$$
y = B^{-1} r_b,
\qquad
z = S^{-1}(r_s - A_{sb} y),
\qquad
x_b = y - B^{-1} A_{bs} z.
$$

This wrapper does not care whether `B^{-1}` is tensor, Jacobi, or something
else.

### Inner block wrapper

For `k = 1` and analogous future cases, the bulk apply itself can be another
Schur wrapper over the block-local model. In other words:

- outer surgery Schur wraps
- inner block Schur wraps
- local block model.

So `schur-schur-jacobi` is structurally valid.

## 6. API Sketch

The public spec should evolve toward something like:

```python
@dataclass(frozen=True)
class MassPreconditionerSpec:
    kind: str = "tensor"
    surgery_schur: bool = False
    steps: int = 4
    power_iterations: int = 30
    damping_safety: float = 0.8
    min_eig_fraction: float = 1e-3
  lanczos_iterations: int = 12
  lanczos_max_eig_inflation: float = 1.1
  lanczos_min_eig_deflation: float = 0.85
  lanczos_min_eig_floor_fraction: float = 1e-3
    smoother: Optional[MassPreconditionerSpec] = None
```

For the first implementation cut, one boolean is enough because the inner
block Schur is implied by the outer surgery choice:

- `surgery_schur=False` means no Schur layers,
- `surgery_schur=True` means all available Schur layers.

If we later want independent control of outer and inner Schur layers, then the
API will need a second structural flag.

## 7. Migration Strategy

### Step 1: Split storage

- Introduce storage for outer surgery data separate from tensor bulk factors.
- Keep current tensor behavior unchanged.

### Step 2: Generic outer wrapper

- Implement a generic outer surgery Schur apply that accepts any bulk apply.
- Route current tensor behavior through that wrapper.

### Step 3: Expose explicit flags

- Add `surgery_schur` to the mass preconditioner spec.
- Preserve old `kind='tensor'` behavior by mapping it to the current compound
  route during transition.

### Step 4: Separate inner block layer

- Keep the implementation behavior simple: if `surgery_schur=True`, enable all
  currently available Schur layers; if `surgery_schur=False`, enable none.
- Internally preserve the distinction between outer surgery Schur and inner
  block Schur so the code remains ready for a later split.

### Step 5: General combinations

- Defer arbitrary outer/inner Schur combinations until after the first
  rollout.
- In the first rollout, allow only
  - no Schur + bulk model,
  - all Schur layers + bulk model.
- Revisit the full combination space later.

## 8. Resolved And Open Questions

Resolved for the current rollout:

- the inner block Schur remains an internal detail in the first cut, so
  `surgery_schur=True` means all currently available Schur layers.
- during transition, plain `kind='tensor'` is treated as
  `kind='tensor', surgery_schur=True` where the legacy route used surgery.
- for `k = 0`, the intended display language is now `outer/schur/inner`; in
  that notation the historical scalar `tensor` route is really
  `none/schur/tensor`.
- the settled public k = 0 interface is now intentionally smaller than the
  internal implementation surface: only `none|jacobi|richardson|chebyshev`
  bulk-only, `none/schur/*(not jacobi)`, the legacy alias
  `tensor -> none/schur/tensor`, and `richardson/schur/*(not jacobi)` remain
  publicly admitted.
- for `k = 1`, the agreed long-run interpretation is recursive rather than
  flat: once Schur is on, the model keeps eliminating through the available
  surgery/RT block structure until it reaches terminal leaf blocks. Those leaf
  blocks should then reuse k = 0-like logic for `A_rr` and k = 3-like logic
  for scalar terminal blocks such as `A_tt` and `A_zz`.
- for `k = 1` and `k = 2`, inner Jacobi under Schur remains disallowed at the
  public operator layer.
- `k = 2` now follows the same public abstraction boundary as `k = 1` and the
  same reduced admissible surface as `k = 0`: outer surgery choice outside,
  bulk-local model inside, with the bulk realized directly as three scalar leaf
  blocks.

Still open:

- whether the public API should eventually expose separate flags for outer
  surgery Schur and inner block Schur.
- whether the public mass-preconditioner API should eventually expose the
  `outer/schur/inner` structure directly, instead of relying on the legacy
  `kind='tensor'` shorthand plus documentation.
- how the recursive `outer/schur/inner` story should be exposed for `k = 1`
  and `k = 2` without reopening the overly broad comparison space we just
  removed at `k = 0`.

## 9. Recommended First Cut

The smallest safe refactor is:

- add generic outer surgery storage,
- add `surgery_schur`,
- let `surgery_schur=True` mean all currently available Schur layers,
- keep the current `k = 1` and `k = 2` inner block Schur grouped with the
  outer surgery choice for one intermediate step,
- and only split the inner block Schur into its own public layer later if that
  extra control proves useful.

That keeps the first step local while still moving the code toward the correct
abstraction.