# Matrix-Free Operations in MRX

## Overview

Matrix-free operations evaluate operators (mass matrices, Laplacians, etc.) without explicitly storing the global matrix. Instead, we apply the operator on-the-fly via element-local sum-factorized quadrature, exploiting the tensor-product structure of spline bases on rectangular domains.

This approach:
- **Reduces memory**: No storage of dense matrices (which can be $O(n^6)$ in naive implementations).
- **Preserves structure**: Leverages separability of tensor-product bases and geometric Jacobian.
- **Enables scalability**: Works efficiently on GPUs via JAX's vmap and einsum optimizations.

---

## Spline Basis Setup

### 1D Basis Evaluation

For each coordinate axis (r, θ, ζ), we evaluate spline basis functions at Gauss quadrature points within each element:

$$B_{i,q} = \Phi_i(x_q)$$

where:
- $\Phi_i$ is the $i$-th B-spline basis function (degree $p$)
- $x_q$ are the Gauss quadrature points within the element
- Result shape: `(n_basis, n_quad_1d)` per axis

For **derivative bases** (used in 1-forms and higher), we evaluate:

$$D_{i,q} = \frac{d\Phi_i}{dx}(x_q)$$

### Global vs. Local DOFs

The code maintains two indexing schemes:

1. **Full tensor-product space**: All $n_r \times n_\theta \times n_\zeta$ DOFs (periodic/clamped basis).
2. **Extracted space** (via extraction matrices $E$): Subset after applying boundary conditions (Dirichlet, extraction).

Example:
- Full k=1 space: $n_1 = n_r \cdot n_\theta \cdot n_\zeta + n_{dr} \cdot n_\theta \cdot n_\zeta + \ldots$ (per component)
- After extraction: `n1_dbc` DOFs (reduced via polar coordinate singularity handling).

---

## Geometry: Jacobian and Metric Tensors

### Pullback of Differential Forms

For a physical domain mapping $F: \hat{\Omega} \to \Omega$:

$$F(r, \theta, \zeta) = \begin{pmatrix} x(r,\theta,\zeta) \\ y(r,\theta,\zeta) \\ z(r,\theta,\zeta) \end{pmatrix}$$

The Jacobian matrix is:

$$J = \frac{\partial F}{\partial (r, \theta, \zeta)} \in \mathbb{R}^{3 \times 3}$$

#### 0-Form (Scalar) Weight

For scalars, the pullback includes only the determinant:

$$\text{Weight}_{k=0} = \det(J) \quad \text{shape: } (n_q,)$$

#### 1-Form Weight

1-forms encode vector components with mixed primal/derivative bases. The component $\alpha$ uses the **contravariant (inverse) metric**:

$$G^{-1} = J^{-T} J^{-1} \in \mathbb{R}^{3 \times 3}$$

$$\text{Weight}_{k=1, \alpha\beta} = (G^{-1})_{\alpha\beta} \cdot \det(J) \quad \text{shape: } (n_q, 3, 3)$$

This matrix-valued weight is evaluated at each quadrature point.

#### 2-Form Weight

2-forms use the **covariant (direct) metric**:

$$G = J^T J \in \mathbb{R}^{3 \times 3}$$

$$\text{Weight}_{k=2, \alpha\beta} = G_{\alpha\beta} \cdot \frac{1}{\det(J)} \quad \text{shape: } (n_q, 3, 3)$$

#### 3-Form Weight

For 3-forms (pseudo-scalars):

$$\text{Weight}_{k=3} = \frac{1}{\det(J)} \quad \text{shape: } (n_q,)$$

### Storage in Geometry Object

The `seq.geometry` object precomputes and stores:

```python
geometry.jacobian_j          # shape: (nquad,), i.e., det(J)
geometry.metric_inv_jkl      # shape: (nquad, 3, 3), i.e., G^{-1}
geometry.metric_jkl          # shape: (nquad, 3, 3), i.e., G
```

These are **reference-domain values** (quadrature points in $[0,1]^3$), computed once per geometry and reused across all solve steps.

---

## Sum-Factorized Element-Local Assembly

### Single Element Block Contraction

For one element with row and column bases (possibly different for mixed blocks):

$$A_{abcdef} = \sum_{qrs} B^r_a(q) \cdot B^c_b(q) \cdot W(q,r,s) \cdot B^r_c(r) \cdot B^c_d(r) \cdot B^r_e(s) \cdot B^c_f(s)$$

where:
- Subscripts $a,b,c,d,e,f$ index local basis functions (per axis).
- Superscripts $r,c$ denote row/column basis (can differ).
- Indices $q,r,s$ run over Gauss quadrature points.
- $W(q,r,s)$ is the metric weight (scalar or matrix component).

**Sum-factorized form** (via einsum):

```python
# Step 1: Contract with row-x and col-x bases, weight along x-axis
A = einsum('qa,qb,q->ab', B_x_row, B_x_col, W)

# Step 2: Contract with row-y and col-y bases
B = einsum('rc,rd,abq->abcd', B_y_row, B_y_col, A)

# Step 3: Contract with row-z and col-z bases
C = einsum('se,sf,abcdr->abcdef', B_z_row, B_z_col, B)
```

**Complexity**: $O(q \cdot n_{\text{local}}^2 \cdot q^2 \cdot n_{\text{local}}^2 \cdot q^2 \cdot n_{\text{local}}^2)$ naive becomes $O(q^3 n_{\text{local}}^2 + q^2 n_{\text{local}}^4)$ with sum factorization.

### 2D Array of Elements

For the full mesh with $(n_e^r, n_e^\theta, n_e^\zeta)$ elements:

```python
# Vectorized over all elements
blocks = vmap(vmap(vmap(
    _elem_block_mixed,
    in_axes=(None, None, 0, 0, 0, None, None, 0)
    ),  # over z elements
    in_axes=(None, None, 0, 0, None, None, 0, None, 0)
    ),  # over y elements
    in_axes=(0, 0, None, None, None, None, 0, 0, None)
    )(...)  # over x elements
```

Result shape: `(ne_r, ne_y, ne_z, p+1, p+1, p+1, p+1, p+1, p+1)` or subset thereof for derivative bases.

---

## Matrix-Free Application (Matvec)

Instead of materializing the element blocks and scattering into a global matrix, we contract directly with the input vector.

### Scalar (k=0) Example

Given input coefficient vector $u \in \mathbb{R}^n$ with local DOFs per element:

$$(\text{output})_i = \sum_{\text{elements}} \sum_{j,q,r,s} E_i \cdot (\text{elem block})_{abcdef} \cdot u_j \cdot E_j^T$$

where $E$ is the extraction matrix accounting for boundary conditions.

**Matrix-free approach** (pseudocode):

```python
def matvec_mass(u_ext):
    # 1. Lift extracted DOFs to full tensor-product space
    u_full = e.T @ u_ext  # shape: (n_r * n_theta * n_zeta,)
    
    # 2. Reshape to element-wise blocks
    u_elem = reshape_to_elements(u_full, (ne_r, ne_y, ne_z, p+1, p+1, p+1))
    
    # 3. For each element, apply local contraction on-the-fly
    result_elem = vmap(...)(
        lambda Bx, By, Bz, W, u_loc: einsum(
            'qa,rc,se,q,abcde,u_abcde->q_rse',
            Bx, By, Bz, W, u_loc
        )
    )(Bx_all, By_all, Bz_all, W_all, u_elem)
    
    # 4. Integrate over quadrature points (Gauss weights fold in)
    result_full = integrate_elements(result_elem, quad_weights)
    
    # 5. Restrict back to extracted space
    return e @ result_full  # shape: (n_dbc,)
```

**Memory**: $O(n)$ input + $O(q^3 \cdot n_e)$ transient quadrature storage (per element, temporary).

---

## Key Einsum Patterns in MRX

### Pattern 1: Basis × Weight × Basis (Single Element)

```python
# For k=0 mass matrix: integral of (basis_i * basis_j * det(J)) dξ
A = einsum('qa,qb,q->ab', B_row, B_col, jacobian)
# Shape: (n_basis_row, n_basis_col)
```

### Pattern 2: Vectorial (k=1, k=2) with Metric

```python
# Component (α,β) of k=1 mass matrix integral
# (basis^r_α @ G^{-1}_αβ @ basis^c_β * det(J))
A = einsum('qa,qb,qαβ,q->ab', 
           B_row, B_col, metric_inv_component, jacobian)
# Shape: (n_basis_row, n_basis_col)
```

### Pattern 3: Batched Element Contractions (3D Tensor)

```python
# Full element block after sum factorization
blocks = einsum('qa,qa,q,rc,rd,se,sf,qrs->abcdef',
                B_x_row, B_x_col, w_x,
                B_y_row, B_y_col, w_y,
                B_z_row, B_z_col, w_z,
                # w_qrs = combined weight at (q,r,s)
                ...)
```

### Pattern 4: Matvec (Element-Local Solve)

```python
# Apply assembled block to local input vector
output = einsum('abcdef,abcdef->abcdef', blocks, input_local)
# Then sum over local indices after extraction handling
```

---

## Geometry Entry Points

### 1. **Quadrature Setup**
```python
# Physical quadrature points in reference domain
x_q, y_q, z_q, w_x, w_y, w_z = seq.quad
```

### 2. **Jacobian Computation**
```python
# Evaluate pullback F at quadrature points
jacobian_at_q = evaluate_pullback_jacobian(F, seq.quad)
# det(J), inverse, metric tensors computed here
geometry = seq.set_map(F)
```

### 3. **Weight Formation**
```python
# For k=1 form:
weight_k1 = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]
```

### 4. **Basis Pullback** (for vectorial forms)
```python
# For k=1 component α: derivative basis on axis α, primal elsewhere
bases_k1_comp0 = [d_basis_r, basis_theta, basis_zeta]  # dr ⊗ Λ_θ ⊗ Λ_ζ
bases_k1_comp1 = [basis_r, d_basis_theta, basis_zeta]  # Λ_r ⊗ dθ ⊗ Λ_ζ
```

---

## Data Flow Diagram

```
Reference Domain (parameter space [0,1]³)
         ↓
    Quadrature Points (seq.quad.x_x, etc.)
         ↓
    Spline Basis Evaluation → B[i, q] (once per assembly)
         ↓ (applies to all DOFs/elements)
    Physical Domain
         ↓
    Geometry Jacobian F'(ξ) → det(J), G, G⁻¹ (seq.geometry)
         ↓
    Form-Dependent Weights
    (k=0: det(J), k=1: G⁻¹·det(J), k=2: G/det(J), k=3: 1/det(J))
         ↓
    Element-Local Einsum Contraction
    (bases × weights → local element block or matvec result)
         ↓
    Global Assembly (scatter via extraction E, E.T)
         ↓
    Output (matrix entries or residual vector)
```

---

## Performance Characteristics

### Assembly Phase
- **Input**: Spline bases, geometry, quadrature.
- **Output**: Precomputed local blocks or cached basis/weight data.
- **Cost**: $O(n^3 \cdot p^6)$ naive → $O(n^3 \cdot q \cdot p^2)$ sum-factorized per element.
- **Storage**: $O(q^3 \cdot p^6)$ per element block (materialized for storage) or $O(q^3 \cdot p^2)$ (on-the-fly).

### Matrix-Free Matvec
- **Input**: Coefficient vector (size $n$).
- **Output**: Residual/result vector (size $n$).
- **Cost**: $O(n \cdot q^3)$ (quadrature evaluation over all elements).
- **Memory**: $O(n)$ + $O(q^3)$ transient per element (no full matrix stored).

### JAX/GPU Acceleration
- **vmap**: Parallelizes over elements or Gauss points.
- **einsum**: Fused tensor contraction on GPU (efficient memory layout).
- **block_until_ready**: Ensures JAX has completed async operations before timing.

---

## Example: k=1 Mass Matrix, One Element

For a 1-form basis $\Phi^1_i$ with components (derivative on one axis, primal on others):

$$M_{ij} = \int_{\Omega} \Phi^1_i \cdot G^{-1}(\xi) \cdot \Phi^1_j \, |\det(J(\xi))| \, d\xi$$

where $G^{-1}$ acts component-wise.

**Reference-domain computation**:

$$\tilde{M}_{ij} = \int_0^1 \int_0^1 \int_0^1 
  \left[ \frac{\partial \Phi_i^r}{\partial r} \, \Gamma^{-1}_{00} \, \frac{\partial \Phi_j^r}{\partial r}
         + \text{off-diagonal terms} \right] 
  J(\xi) \, d\xi_r d\xi_\theta d\xi_\zeta$$

**Gauss quadrature**:

$$\tilde{M}_{ij} \approx \sum_{q,r,s} 
  \left( \frac{d\Phi_i^r}{dr}\big|_q \right) (G^{-1}(\xi_{qrs}))_{00} 
  \left( \frac{d\Phi_j^r}{dr}\big|_r \right)
  (\Phi_i^\theta(\xi_r)) \cdots J(\xi_{qrs}) \, w_q w_r w_s$$

**Einsum form** (component 0 of k=1):

```python
M0 = einsum('qa,qb,q,rc,rd,se,sf,q->abcdef',
    d_B_r_row, d_B_r_col, w_x,       # derivative basis x-axis
    B_t_row,   B_t_col,   w_y,       # primal basis y-axis
    B_z_row,   B_z_col,   w_z,       # primal basis z-axis
    jacobian * metric_inv_00)        # weight: G⁻¹₀₀ · det(J)
```

---

## Exact k=1 Matrix-Free Matvec Code

This section provides the complete working code for applying a k=1 mass matrix without materializing it, including the vector-valued basis functions and metric tensor contractions.

### Vector-Valued Basis Functions for k=1

The **k=1 form** has three vector components. Each component uses a **mixed primal/derivative basis**:

```python
def _component_axis_bases_k1(form, c):
    """k=1 component c: derivative basis on axis c, primal elsewhere.
    
    For component 0 (r-component):   [dΛ_r, Λ_θ, Λ_ζ]
    For component 1 (θ-component):  [Λ_r, dΛ_θ, Λ_ζ]
    For component 2 (ζ-component):  [Λ_r, Λ_θ, dΛ_ζ]
    
    This ensures the pullback of the vector curl is correct.
    """
    bases = [form.Λ[0], form.Λ[1], form.Λ[2]]
    bases[c] = form.dΛ[c]  # Replace axis c with its derivative
    return bases
```

Where:
- `form.Λ[i]` is the primal (non-differentiated) spline basis on axis i
- `form.dΛ[i]` is the derivative spline basis on axis i

### Metric Tensor for k=1

The **contravariant metric** $G^{-1}$ (pulled back via the Jacobian) is:

```python
# K=1 setup: contravariant metric (inverse metric)
metric = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]
# Shape: (nquad, 3, 3)
# metric[q, α, β] = G⁻¹(q)_{α,β} · det(J(q))

# Extract the (row_component, col_component) pair weights
pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
weight_of = {(cr, cc): metric[:, cr, cc] for cr, cc in pairs}
```

This creates a 3×3 grid of **scalar weights**, one for each (α,β) pair:
- `weight_of[(0, 0)]` = $G^{-1}_{00} \cdot \det(J)$ (r-r block)
- `weight_of[(0, 1)]` = $G^{-1}_{01} \cdot \det(J)$ (r-θ block)
- `weight_of[(1, 0)]` = $G^{-1}_{10} \cdot \det(J)$ (θ-r block)
- ... (9 pairs total)

### Single Element-Local Contraction (Pseudocode)

For one element with **row component `cr`** and **column component `cc`**:

```python
def _element_apply(Bvals_r, Bvals_c, W, x_flat_c, gather_idx_c):
    """Apply mass matrix for one (row_comp, col_comp) pair on one element.
    
    Parameters:
    -----------
    Bvals_r : (Bxr, Byr, Bzr)
        Row basis values at quadrature: each (n_elem, q_per_axis, n_basis_row)
    Bvals_c : (Bxc, Byc, Bzc)
        Col basis values at quadrature: each (n_elem, q_per_axis, n_basis_col)
    W : (n_elem, qx, qy, qz)
        Metric weight G⁻¹[cr, cc] · det(J) folded with Gauss weights w_x ⊗ w_y ⊗ w_z
    x_flat_c : (n_total_dofs_comp_c,)
        Input DOF vector for column component c
    gather_idx_c : (n_elem, p+1, p+1, p+1)
        Pre-computed flat indices to gather element-local DOFs from x_flat_c
    """
    Bxr, Byr, Bzr = Bvals_r
    Bxc, Byc, Bzc = Bvals_c
    
    # -----  GATHER phase: extract element-local input vector -----
    x_local = x_flat_c[gather_idx_c]
    # Shape: (n_elem, n_basis_x_c, n_basis_y_c, n_basis_z_c)
    
    # ----- COLUMN basis -> quadrature (forward transform) -----
    # Apply column bases along each axis: x -> q, y -> r, z -> s
    
    # Einsum index legend:
    # x = element index in x-direction
    # y = element index in y-direction
    # z = element index in z-direction
    # q, r, s = quadrature point indices (along x, y, z)
    # b, d, f = column basis function indices
    # a, c, e = row basis function indices
    
    # Step 1: Apply col basis in x, collapse to quadrature points
    t1 = einsum('xqb,xyzbdf->xyzqdf', Bxc, x_local)
    # (x basis x q-points) ⊗ (element-local coeffs) -> quad-point values
    
    # Step 2: Apply col basis in y
    t2 = einsum('yrd,xyzqdf->xyzqrf', Byc, t1)
    
    # Step 3: Apply col basis in z
    u = einsum('zsf,xyzqrf->xyzqrs', Bzc, t2)
    # u[x,y,z,q,r,s] = ∑_b,d,f Bxc[x,q,b] Byc[y,r,d] Bzc[z,s,f] x[x,y,z,b,d,f]
    
    # ----- MULTIPLY by metric weight -----
    # W includes both the metric tensor component G⁻¹[cr,cc] and Gauss weights
    u = u * W  # broadcast W[x,y,z,q,r,s] over u
    
    # ----- ROW basis <- quadrature (reverse transform) -----
    # Apply row bases along each axis to get back to basis coefficients
    
    # Step 4: Apply row basis in x
    s1 = einsum('xqa,xyzqrs->xyzars', Bxr, u)
    
    # Step 5: Apply row basis in y
    s2 = einsum('yrc,xyzars->xyzacs', Byr, s1)
    
    # Step 6: Apply row basis in z
    y_local = einsum('zse,xyzacs->xyzace', Bzr, s2)
    # y_local[x,y,z,a,c,e] = ∑_q,r,s Bxr[x,q,a] Byr[y,r,c] Bzr[z,s,e] u[x,y,z,q,r,s]
    
    return y_local  # (n_elem, n_basis_x_r, n_basis_y_r, n_basis_z_r)
```

**Einsum Summary (Single Element Apply)**:

The full contraction for one (cr, cc) pair on one element can be written as:

$$y_{ace} = \sum_{bdf,qrs} \Phi^c_r[b,q] \, \Phi^c_r[d,r] \, \Phi^c_r[f,s] \, G^{-1}_{cr,cc}[q,r,s] \, \det(J)[q,r,s] \, w_q w_r w_s \, u_{bdf}$$

where:
- $\Phi^c_r[b,q]$ = `Bxc[q,b]` (column basis x on axis)
- $\Phi^r_r[a,q]$ = `Bxr[q,a]` (row basis x on axis)
- $G^{-1}_{cr,cc}$ is the (cr, cc) component of the **contravariant metric tensor**

### Full Jitted Matvec Implementation

```python
def build_matrixfree_mass_apply_k1(seq, geometry=None):
    """Build jitted matvec: x -> M_1 x (no matrix stored)."""
    
    geometry = seq.geometry if geometry is None else geometry
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz
    ne_x, ne_y, ne_z, qx, qy, qz = _elem_counts(seq)
    
    # ===== SETUP: Component basis functions =====
    form = seq.basis_1
    
    # Evaluate 1D basis functions at quadrature points (done once)
    comp = _bases_for_form(seq, form, _component_axis_bases_k1, n_comp=3)
    # comp[0] = (Bx_comp0, gx, By_comp0, gy, Bz_comp0, gz) where:
    #   Bx_comp0 = d_basis_r (derivative on r-axis)
    #   By_comp0 = basis_theta (primal)
    #   Bz_comp0 = basis_zeta (primal)
    # comp[1] = (basis_r, ..., d_basis_theta, ..., basis_zeta)
    # comp[2] = (basis_r, ..., basis_theta, ..., d_basis_zeta)
    
    # ===== SETUP: Contravariant metric tensor =====
    metric = geometry.metric_inv_jkl * geometry.jacobian_j[:, None, None]
    # Shape: (nquad, 3, 3)
    # metric[q, α, β] = G⁻¹(q)_{α,β} · det(J(q))
    
    # Pre-compute Gauss weights and fold into metric for each (cr, cc) pair
    gw = _quad_gauss_weight(seq)  # (ne_x, ne_y, ne_z, qx, qy, qz)
    
    pairs = [(cr, cc) for cr in range(3) for cc in range(3)]
    W_split = {}
    for (cr, cc) in pairs:
        # Extract component (cr, cc) of metric and reshape to element grid
        Wf = _split_field(metric[:, cr, cc], nx, ny, nz,
                          ne_x, ne_y, ne_z, qx, qy, qz)
        # Fold in Gauss weights
        W_split[(cr, cc)] = Wf * gw
        # Now W_split[(cr, cc)][e_x, e_y, e_z, q, r, s]
        #   = G⁻¹[cr, cc](q,r,s) · det(J(q,r,s)) · w_x[q] · w_y[r] · w_z[s]
    
    # ===== SETUP: Static gather/scatter index plans =====
    # (Pre-computed on host, reused every matvec)
    shapes = form.shape  # Per-component DOF-grid shapes
    Bvals = tuple((c[0], c[2], c[4]) for c in comp)
    gather_idx = tuple(
        _flat_dof_plan(comp[cc][1], comp[cc][3], comp[cc][5], shapes[cc])
        for cc in range(3))  # One per column component
    seg_idx = tuple(
        _flat_dof_plan(comp[cr][1], comp[cr][3], comp[cr][5], shapes[cr]).reshape(-1)
        for cr in range(3))  # One per row component
    nseg = tuple(int(np.prod(shapes[c])) for c in range(3))
    starts_t = (0, form.n1, form.n1 + form.n2)
    
    # ===== JITTED KERNEL =====
    @jax.jit
    def _impl(x, Bvals, W_split, gather_idx, seg_idx):
        """Apply M_1 @ x in tensor-product DOF space.
        
        Parameters:
        -----------
        x : (n1_full,)
            Input vector in full k=1 DOF space (3 component blocks)
        Bvals : tuple of (Bxc, Byc, Bzc) for each component
        W_split : dict of (cr, cc) -> weight array
        gather_idx : tuple of index plans (one per col component)
        seg_idx : tuple of segment-id arrays (one per row component)
        """
        # Split input into three component vectors
        x_comp = [x[starts_t[c]:starts_t[c+1]] for c in range(3)]
        
        out_parts = []
        
        # Loop over row components (3 output vectors)
        for cr in range(3):
            # Accumulator for this row component
            acc = jnp.zeros(nseg[cr], dtype=x.dtype)
            
            # Loop over column components (3 input vectors)
            for cc in range(3):
                # Apply M_1[(cr, cc)] block
                # This is: element_apply(...) operating on x_comp[cc]
                
                y_local = _element_apply(
                    Bvals[cr],              # Row basis (Bxr, Byr, Bzr)
                    Bvals[cc],              # Col basis (Bxc, Byc, Bzc)
                    W_split[(cr, cc)],      # Weight: G⁻¹[cr,cc] · det(J) · w_x w_y w_z
                    x_comp[cc],             # Input DOF vector (component cc)
                    gather_idx[cc])         # Index plan for gathering x_comp[cc]
                
                # Scatter element-local results into global output
                acc = acc + jax.ops.segment_sum(
                    y_local.reshape(-1),    # Flatten all (n_elem, p+1, p+1, p+1)
                    seg_idx[cr],            # Segment IDs for row component cr
                    num_segments=nseg[cr])
            
            out_parts.append(acc)
        
        # Concatenate all three components
        return jnp.concatenate(out_parts)
    
    def apply(x):
        """Public interface: apply matrix-free k=1 mass matvec."""
        return _impl(x, Bvals, W_split, gather_idx, seg_idx)
    
    return apply
```

### Data Flow Diagram for k=1 Matvec

```
Input vector x (k=1, full DOF space)
  ↓
  ├─ Split into x_comp[0], x_comp[1], x_comp[2]
  ↓
  For each row component cr in {0, 1, 2}:
    ├─ For each col component cc in {0, 1, 2}:
    │   ├─ Gather: x_comp[cc] -> local element vectors via gather_idx[cc]
    │   ├─ Element apply (einsum):
    │   │   ├─ Forward: apply Bvals[cc] (col bases) -> quad points
    │   │   ├─ Weight:  multiply by W_split[(cr,cc)] = G⁻¹[cr,cc] · det(J) · w
    │   │   └─ Reverse: apply Bvals[cr] (row bases) <- quad points
    │   └─ Scatter: element-local results -> output via seg_idx[cr]
    ├─ Accumulate all cc contributions
    └─ out_parts[cr] = final output for component cr
  ↓
  Concatenate -> output vector y (k=1, full DOF space)
```

### Key Insights

1. **Vector Basis Structure**: Each k=1 component uses **mixed bases** (derivative on one axis, primal on others).

2. **Metric Tensor Integration**: The contravariant metric $G^{-1}$ enters as a **point-wise scalar weight** for each (α,β) pair after pullback and Jacobian multiplication.

3. **9 Blocks**: The 3×3 metric induces a **3×3 block structure** in the discretized mass matrix:
   - Diagonal blocks (α=β): typically larger (have non-zero inverse metric).
   - Off-diagonal blocks (α≠β): smaller (non-orthogonal geometry).

4. **No Quadrature Stored**: The weight `W_split[(cr,cc)]` folds the Gauss weights directly into the geometric weight, so the kernel never separates them.

5. **GPU-Friendly**: The einsum operations, gather, and segment_sum are all JAX primitives optimized for GPU execution.

---

## Summary

**Matrix-free in MRX leverages**:

1. **Sum-factorized structure**: Tensor-product bases separate into 1D operations.
2. **Geometry via pullback**: Jacobian and metric encode the domain mapping.
3. **Efficient einsum**: JAX's fused tensor contraction minimizes memory bandwidth.
4. **No global matrix**: Only element-local blocks or on-the-fly matvec, enabling large problems.

The key insight is that geometry (through $\det(J)$, $G$, $G^{-1}$) enters as a **point-wise weight** during quadrature, multiplicatively attached to the basis product. This multiplicativity preserves the separable structure, so sum factorization remains valid and efficient even with nonlinear geometry.
