from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.derham_sequence import DeRhamSequence
from mrx.extraction_operators import get_xi
from mrx.mappings import rotating_ellipse_map
from mrx.operators import assemble_mass_operators, dense_mass_matrix


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ExperimentConfig:
    ns: tuple[int, int, int] = (6, 8, 4)
    p: int = 3
    tol: float = 1e-9
    maxiter: int = 1000
    betti: tuple[int, int, int, int] = (1, 1, 0, 0)
    rotating_eps: float = 0.33
    rotating_kappa: float = 1.4
    rotating_r0: float = 1.0
    rotating_nfp: int = 3
    dirichlet: bool = True


CONFIG = ExperimentConfig()


def build_case(config: ExperimentConfig = CONFIG):
    seq = DeRhamSequence(
        config.ns,
        (config.p, config.p, config.p),
        2 * config.p,
        ("clamped", "periodic", "periodic"),
        polar=True,
        tol=config.tol,
        maxiter=config.maxiter,
        betti_numbers=config.betti,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(
        rotating_ellipse_map(
            eps=config.rotating_eps,
            kappa=config.rotating_kappa,
            R0=config.rotating_r0,
            nfp=config.rotating_nfp,
        )
    )
    operators = assemble_mass_operators(seq, seq.geometry, ks=(1,))
    return seq, operators


def _k1_layout_sizes(seq, dirichlet: bool):
    boundary_offset = 1 if dirichlet else 0
    return {
        "theta_surgery": 2 * seq.basis_1.nz,
        "zeta_surgery": 3 * seq.basis_1.dz,
        "r": (seq.basis_1.dr - 1) * seq.basis_1.nt * seq.basis_1.nz,
        "theta_bulk": (seq.basis_1.nr - 2 - boundary_offset) * seq.basis_1.dt * seq.basis_1.nz,
        "zeta_bulk": (seq.basis_1.nr - 2 - boundary_offset) * seq.basis_1.nt * seq.basis_1.dz,
    }


def _current_surgery_slices_k1(seq, dirichlet: bool):
    sizes = _k1_layout_sizes(seq, dirichlet)
    theta_surgery = slice(0, sizes["theta_surgery"])
    zeta_surgery = slice(theta_surgery.stop, theta_surgery.stop + sizes["zeta_surgery"])
    r_slice = slice(zeta_surgery.stop, zeta_surgery.stop + sizes["r"])
    theta_bulk = slice(r_slice.stop, r_slice.stop + sizes["theta_bulk"])
    zeta_bulk = slice(theta_bulk.stop, theta_bulk.stop + sizes["zeta_bulk"])
    return {
        "r": r_slice,
        "theta_surgery": theta_surgery,
        "theta_bulk": theta_bulk,
        "zeta_surgery": zeta_surgery,
        "zeta_bulk": zeta_bulk,
    }


def _raw_component_slices_k1(seq):
    n_r = seq.basis_1.n1
    n_theta = seq.basis_1.n2
    n_zeta = seq.basis_1.n3
    r_slice = slice(0, n_r)
    theta_slice = slice(r_slice.stop, r_slice.stop + n_theta)
    zeta_slice = slice(theta_slice.stop, theta_slice.stop + n_zeta)
    return {
        "r": r_slice,
        "theta": theta_slice,
        "zeta": zeta_slice,
    }


def dense_extraction_k1(seq, *, dirichlet: bool) -> jnp.ndarray:
    extraction = seq.e1_dbc if dirichlet else seq.e1
    return jnp.asarray(extraction.todense())


def dense_raw_mass_k1(operators) -> jnp.ndarray:
    return jnp.asarray(operators.m1.todense())


def dense_extracted_mass_k1(seq, operators, *, dirichlet: bool) -> jnp.ndarray:
    return jnp.asarray(dense_mass_matrix(seq, operators, 1, dirichlet=dirichlet))


def current_surgery_rows(extraction: jnp.ndarray, seq, *, dirichlet: bool) -> jnp.ndarray:
    slices = _current_surgery_slices_k1(seq, dirichlet)
    row_indices = jnp.concatenate([
        jnp.arange(slices["theta_surgery"].start, slices["theta_surgery"].stop),
        jnp.arange(slices["zeta_surgery"].start, slices["zeta_surgery"].stop),
    ])
    return extraction[row_indices, :]


def current_surgery_coordinates(extraction: jnp.ndarray, seq, *, dirichlet: bool) -> jnp.ndarray:
    slices = _current_surgery_slices_k1(seq, dirichlet)
    row_indices = jnp.concatenate([
        jnp.arange(slices["theta_surgery"].start, slices["theta_surgery"].stop),
        jnp.arange(slices["zeta_surgery"].start, slices["zeta_surgery"].stop),
    ])
    return jax.nn.one_hot(row_indices, extraction.shape[0], dtype=extraction.dtype)


def build_inner_r_candidate_rows(seq, *, variant: str = "xi-delta") -> jnp.ndarray:
    xi = get_xi(seq.basis_1.nt)
    if variant == "xi-delta":
        angular_weights = xi[:, 1, :] - xi[:, 0, :]
    elif variant == "xi-left":
        angular_weights = xi[:, 0, :]
    elif variant == "xi-right":
        angular_weights = xi[:, 1, :]
    else:
        raise ValueError(f"Unsupported variant {variant!r}")

    nt = seq.basis_1.nt
    nz = seq.basis_1.nz
    raw_slices = _raw_component_slices_k1(seq)
    rows = []
    for p in range(3):
        for m in range(nz):
            row = jnp.zeros((seq.basis_1.n,), dtype=jnp.float64)
            column_indices = raw_slices["r"].start + jnp.ravel_multi_index(
                (
                    jnp.zeros((nt,), dtype=jnp.int32),
                    jnp.arange(nt, dtype=jnp.int32),
                    jnp.full((nt,), m, dtype=jnp.int32),
                ),
                (seq.basis_1.dr, seq.basis_1.nt, seq.basis_1.nz),
                mode="clip",
            )
            row = row.at[column_indices].set(angular_weights[p])
            rows.append(row)
    return jnp.stack(rows, axis=0)


def build_inner_r_selector_rows(seq) -> jnp.ndarray:
    nt = seq.basis_1.nt
    nz = seq.basis_1.nz
    raw_slices = _raw_component_slices_k1(seq)
    rows = []
    for j in range(nt):
        for m in range(nz):
            row = jnp.zeros((seq.basis_1.n,), dtype=jnp.float64)
            column_index = raw_slices["r"].start + jnp.ravel_multi_index(
                (
                    jnp.array(0, dtype=jnp.int32),
                    jnp.array(j, dtype=jnp.int32),
                    jnp.array(m, dtype=jnp.int32),
                ),
                (seq.basis_1.dr, seq.basis_1.nt, seq.basis_1.nz),
                mode="clip",
            )
            row = row.at[column_index].set(1.0)
            rows.append(row)
    return jnp.stack(rows, axis=0)


def _kron_block(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(left, right)


def _complete_qr_basis(columns: jnp.ndarray) -> tuple[jnp.ndarray, int]:
    q, r = jnp.linalg.qr(columns, mode="complete")
    diag = jnp.abs(jnp.diag(r[: columns.shape[1], : columns.shape[1]]))
    rank = int(jnp.sum(diag > 1e-10))
    return q, rank


def build_r_shell_split_transform(seq, *, dirichlet: bool, angular_variant: str = "xi-right") -> dict[str, object]:
    slices = _current_surgery_slices_k1(seq, dirichlet)
    xi = get_xi(seq.basis_1.nt)
    if angular_variant == "xi-right":
        angular_columns = xi[:, 1, :].T
    elif angular_variant == "xi-left":
        angular_columns = xi[:, 0, :].T
    elif angular_variant == "xi-delta":
        angular_columns = (xi[:, 1, :] - xi[:, 0, :]).T
    else:
        raise ValueError(f"Unsupported angular_variant {angular_variant!r}")

    q_theta, rank = _complete_qr_basis(angular_columns)
    identity_z = jnp.eye(seq.basis_1.nz, dtype=jnp.float64)
    shell_transform = _kron_block(q_theta.T, identity_z)

    r_size = slices["r"].stop - slices["r"].start
    shell_size = seq.basis_1.nt * seq.basis_1.nz
    rest_r_size = r_size - shell_size
    if rest_r_size < 0:
        raise ValueError("r block is smaller than a single radial shell")

    r_transform = jnp.eye(r_size, dtype=jnp.float64)
    r_transform = r_transform.at[:shell_size, :shell_size].set(shell_transform)

    total_size = slices["zeta_bulk"].stop
    transform = jnp.eye(total_size, dtype=jnp.float64)
    transform = transform.at[slices["r"], slices["r"]].set(r_transform)

    r_special = slice(slices["r"].start, slices["r"].start + rank * seq.basis_1.nz)
    r_bulk = slice(r_special.stop, slices["r"].stop)
    return {
        "transform": transform,
        "angular_basis": q_theta,
        "angular_rank": rank,
        "shell_count": 1,
        "r_special": r_special,
        "r_bulk": r_bulk,
        "surgery_prefix": slice(0, r_special.stop),
        "theta_bulk": slices["theta_bulk"],
        "zeta_bulk": slices["zeta_bulk"],
    }


def build_r_multishell_split_transform(
    seq,
    *,
    dirichlet: bool,
    angular_variant: str = "xi-right",
    shell_count: int = 1,
) -> dict[str, object]:
    slices = _current_surgery_slices_k1(seq, dirichlet)
    xi = get_xi(seq.basis_1.nt)
    if angular_variant == "xi-right":
        angular_columns = xi[:, 1, :].T
    elif angular_variant == "xi-left":
        angular_columns = xi[:, 0, :].T
    elif angular_variant == "xi-delta":
        angular_columns = (xi[:, 1, :] - xi[:, 0, :]).T
    else:
        raise ValueError(f"Unsupported angular_variant {angular_variant!r}")

    q_theta, rank = _complete_qr_basis(angular_columns)
    identity_z = jnp.eye(seq.basis_1.nz, dtype=jnp.float64)
    shell_transform = _kron_block(q_theta.T, identity_z)

    r_size = slices["r"].stop - slices["r"].start
    shell_size = seq.basis_1.nt * seq.basis_1.nz
    if shell_count < 1:
        raise ValueError("shell_count must be positive")
    if shell_count * shell_size > r_size:
        raise ValueError("Requested too many radial shells for the r block")

    r_transform = jnp.eye(r_size, dtype=jnp.float64)
    for shell in range(shell_count):
        shell_slice = slice(shell * shell_size, (shell + 1) * shell_size)
        r_transform = r_transform.at[shell_slice, shell_slice].set(shell_transform)

    total_size = slices["zeta_bulk"].stop
    transform = jnp.eye(total_size, dtype=jnp.float64)
    transform = transform.at[slices["r"], slices["r"]].set(r_transform)

    special_size = shell_count * rank * seq.basis_1.nz
    r_special = slice(slices["r"].start, slices["r"].start + special_size)
    r_bulk = slice(r_special.stop, slices["r"].stop)
    return {
        "transform": transform,
        "angular_basis": q_theta,
        "angular_rank": rank,
        "shell_count": shell_count,
        "r_special": r_special,
        "r_bulk": r_bulk,
        "surgery_prefix": slice(0, r_special.stop),
        "theta_bulk": slices["theta_bulk"],
        "zeta_bulk": slices["zeta_bulk"],
    }


def _relative_block_norm(matrix: jnp.ndarray, row_slice: slice, col_slice: slice) -> float:
    block = matrix[row_slice, col_slice]
    return float(jnp.linalg.norm(block))


def report_r_shell_split_effect(seq, extracted_mass: jnp.ndarray, *, dirichlet: bool, angular_variant: str = "xi-right"):
    split = build_r_shell_split_transform(seq, dirichlet=dirichlet, angular_variant=angular_variant)
    transformed = split["transform"] @ extracted_mass @ split["transform"].T
    slices = _current_surgery_slices_k1(seq, dirichlet)
    return {
        "split": split,
        "transformed_matrix": transformed,
        "old_rt_coupling_norm": _relative_block_norm(extracted_mass, slices["r"], slices["theta_bulk"]),
        "new_rbulk_theta_coupling_norm": _relative_block_norm(transformed, split["r_bulk"], split["theta_bulk"]),
        "new_rspecial_theta_coupling_norm": _relative_block_norm(transformed, split["r_special"], split["theta_bulk"]),
    }


def report_r_multishell_split_effect(
    seq,
    extracted_mass: jnp.ndarray,
    *,
    dirichlet: bool,
    angular_variant: str,
    shell_count: int,
):
    split = build_r_multishell_split_transform(
        seq,
        dirichlet=dirichlet,
        angular_variant=angular_variant,
        shell_count=shell_count,
    )
    transformed = split["transform"] @ extracted_mass @ split["transform"].T
    slices = _current_surgery_slices_k1(seq, dirichlet)
    old_norm = _relative_block_norm(extracted_mass, slices["r"], slices["theta_bulk"])
    new_bulk_norm = _relative_block_norm(transformed, split["r_bulk"], split["theta_bulk"])
    return {
        "split": split,
        "transformed_matrix": transformed,
        "old_rt_coupling_norm": old_norm,
        "new_rbulk_theta_coupling_norm": new_bulk_norm,
        "new_rspecial_theta_coupling_norm": _relative_block_norm(transformed, split["r_special"], split["theta_bulk"]),
        "bulk_coupling_retained_fraction": 0.0 if old_norm == 0.0 else new_bulk_norm / old_norm,
    }


def solve_row_coordinates(extraction: jnp.ndarray, target_rows: jnp.ndarray) -> jnp.ndarray:
    gram = extraction @ extraction.T
    return target_rows @ extraction.T @ jnp.linalg.pinv(gram)


def orthonormal_row_basis(rows: jnp.ndarray, *, tol: float = 1e-10) -> tuple[jnp.ndarray, jnp.ndarray]:
    if rows.size == 0:
        return jnp.zeros((0, rows.shape[1]), dtype=rows.dtype), jnp.zeros((0,), dtype=rows.dtype)
    _, singular_values, vh = jnp.linalg.svd(rows, full_matrices=False)
    rank = int(jnp.sum(singular_values > tol))
    return vh[:rank, :], singular_values


def remove_row_space_component(rows: jnp.ndarray, base_basis: jnp.ndarray) -> jnp.ndarray:
    if rows.size == 0 or base_basis.size == 0:
        return rows
    coefficients = rows @ base_basis.T
    return rows - coefficients @ base_basis


def build_transform_from_special_rows(special_rows: jnp.ndarray, *, tol: float = 1e-10) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    row_basis, singular_values = orthonormal_row_basis(special_rows, tol=tol)
    _, _, vh = jnp.linalg.svd(row_basis, full_matrices=True)
    return vh, row_basis, singular_values


def project_raw_rows_to_extracted(extraction: jnp.ndarray, target_rows: jnp.ndarray) -> dict[str, jnp.ndarray | float]:
    coordinates = solve_row_coordinates(extraction, target_rows)
    reconstruction = coordinates @ extraction
    residual = target_rows - reconstruction
    return {
        "coordinates": coordinates,
        "reconstruction": reconstruction,
        "residual": residual,
        "relative_residual": float(jnp.linalg.norm(residual) / jnp.where(jnp.linalg.norm(target_rows) > 0, jnp.linalg.norm(target_rows), 1.0)),
    }


def describe_candidate_space(
    extraction: jnp.ndarray,
    seq,
    *,
    dirichlet: bool,
    current_rows: jnp.ndarray,
    candidate_raw_rows: jnp.ndarray,
):
    candidate_projection = project_raw_rows_to_extracted(extraction, candidate_raw_rows)
    candidate_coords = candidate_projection["coordinates"]
    current_coords = current_surgery_coordinates(extraction, seq, dirichlet=dirichlet)
    current_basis, _ = orthonormal_row_basis(current_coords)
    candidate_basis, candidate_singular_values = orthonormal_row_basis(candidate_coords)
    candidate_novel_coords = remove_row_space_component(candidate_coords, current_basis)
    candidate_novel_basis, candidate_novel_singular_values = orthonormal_row_basis(candidate_novel_coords)
    combined_basis, combined_singular_values = orthonormal_row_basis(
        jnp.concatenate([current_basis, candidate_basis], axis=0)
    )
    overlap = current_basis @ candidate_basis.T if current_basis.size and candidate_basis.size else jnp.zeros((0, 0))
    return {
        "candidate_projection": candidate_projection,
        "current_basis": current_basis,
        "candidate_basis": candidate_basis,
        "candidate_singular_values": candidate_singular_values,
        "candidate_novel_basis": candidate_novel_basis,
        "candidate_novel_singular_values": candidate_novel_singular_values,
        "combined_basis": combined_basis,
        "combined_singular_values": combined_singular_values,
        "overlap": overlap,
    }


def regression_report(extraction: jnp.ndarray, raw_mass: jnp.ndarray, extracted_mass: jnp.ndarray, transform: jnp.ndarray):
    transformed_extraction = transform @ extraction
    transformed_from_extraction = transformed_extraction @ raw_mass @ transformed_extraction.T
    transformed_from_matrix = transform @ extracted_mass @ transform.T
    diff = transformed_from_extraction - transformed_from_matrix
    return {
        "transformed_extraction": transformed_extraction,
        "transformed_from_extraction": transformed_from_extraction,
        "transformed_from_matrix": transformed_from_matrix,
        "relative_difference": float(
            jnp.linalg.norm(diff)
            / jnp.where(jnp.linalg.norm(transformed_from_matrix) > 0, jnp.linalg.norm(transformed_from_matrix), 1.0)
        ),
        "orthogonality_defect": float(jnp.linalg.norm(transform @ transform.T - jnp.eye(transform.shape[0]))),
    }


def print_summary(seq, *, dirichlet: bool, diagnostics: dict[str, object], transform_report: dict[str, object], variant: str):
    current_slices = _current_surgery_slices_k1(seq, dirichlet)
    current_special_size = (
        current_slices["theta_surgery"].stop - current_slices["theta_surgery"].start
        + current_slices["zeta_surgery"].stop - current_slices["zeta_surgery"].start
    )
    print("=" * 112)
    print(
        f"k=1 extraction-transform diagnostics: ns={seq.ns}, p={seq.ps[0]}, dirichlet={dirichlet}, "
        f"candidate_variant={variant}"
    )
    print(f"current explicit surgery size = {current_special_size}")
    print(f"candidate projection relative residual = {diagnostics['candidate_projection']['relative_residual']:.3e}")
    print(f"current special row-space dimension = {diagnostics['current_basis'].shape[0]}")
    print(f"candidate extracted row-space dimension = {diagnostics['candidate_basis'].shape[0]}")
    print(f"candidate novel dimension beyond current surgery = {diagnostics['candidate_novel_basis'].shape[0]}")
    print(f"combined row-space dimension = {diagnostics['combined_basis'].shape[0]}")
    print(f"max abs overlap(current, candidate) = {float(jnp.max(jnp.abs(diagnostics['overlap']))) if diagnostics['overlap'].size else 0.0:.3e}")
    print(f"transform orthogonality defect = {transform_report['orthogonality_defect']:.3e}")
    print(f"matrix regression relative defect = {transform_report['relative_difference']:.3e}")
    print("candidate singular values:")
    print(jnp.asarray(diagnostics["candidate_singular_values"]))
    print("candidate novel singular values:")
    print(jnp.asarray(diagnostics["candidate_novel_singular_values"]))
    print("combined singular values:")
    print(jnp.asarray(diagnostics["combined_singular_values"]))


def print_r_shell_split_summary(report: dict[str, object], *, angular_variant: str):
    split = report["split"]
    print("-" * 112)
    print(f"r-shell split diagnostics: angular_variant={angular_variant}")
    print(f"angular rank = {split['angular_rank']}")
    print(f"enlarged surgery prefix = {split['surgery_prefix']}")
    print(f"r_special = {split['r_special']}")
    print(f"r_bulk = {split['r_bulk']}")
    print(f"||old A[r, theta_bulk]||_F = {report['old_rt_coupling_norm']:.6e}")
    print(f"||new A[r_special, theta_bulk]||_F = {report['new_rspecial_theta_coupling_norm']:.6e}")
    print(f"||new A[r_bulk, theta_bulk]||_F = {report['new_rbulk_theta_coupling_norm']:.6e}")


def print_r_multishell_sweep_summary(reports: list[dict[str, object]]):
    print("-" * 112)
    print("r-shell sweep summary")
    print(f"{'variant':<12} {'shells':>6} {'rank':>6} {'surgery':>10} {'old':>12} {'new bulk':>12} {'retained':>10}")
    for report in reports:
        split = report["split"]
        print(
            f"{split['angular_variant']:<12} {split['shell_count']:>6d} {split['angular_rank']:>6d} "
            f"{split['surgery_prefix'].stop:>10d} {report['old_rt_coupling_norm']:>12.6e} "
            f"{report['new_rbulk_theta_coupling_norm']:>12.6e} {report['bulk_coupling_retained_fraction']:>10.3%}"
        )


def plot_matrix(title: str, matrix: jnp.ndarray):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(jnp.log10(jnp.abs(matrix) + 1e-16), cmap="viridis", origin="lower")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.show()


def main(config: ExperimentConfig = CONFIG, *, candidate_variant: str = "xi-delta"):
    seq, operators = build_case(config)
    extraction = dense_extraction_k1(seq, dirichlet=config.dirichlet)
    raw_mass = dense_raw_mass_k1(operators)
    extracted_mass = dense_extracted_mass_k1(seq, operators, dirichlet=config.dirichlet)

    current_rows = current_surgery_rows(extraction, seq, dirichlet=config.dirichlet)
    if candidate_variant == "inner-r-selectors":
        candidate_raw_rows = build_inner_r_selector_rows(seq)
    else:
        candidate_raw_rows = build_inner_r_candidate_rows(seq, variant=candidate_variant)
    diagnostics = describe_candidate_space(
        extraction,
        seq,
        dirichlet=config.dirichlet,
        current_rows=current_rows,
        candidate_raw_rows=candidate_raw_rows,
    )

    transform, _, _ = build_transform_from_special_rows(diagnostics["combined_basis"])
    transform_report = regression_report(extraction, raw_mass, extracted_mass, transform)
    print_summary(seq, dirichlet=config.dirichlet, diagnostics=diagnostics, transform_report=transform_report, variant=candidate_variant)

    r_shell_report = report_r_shell_split_effect(
        seq,
        extracted_mass,
        dirichlet=config.dirichlet,
        angular_variant="xi-right",
    )
    print_r_shell_split_summary(r_shell_report, angular_variant="xi-right")

    sweep_reports = []
    for angular_variant in ("xi-right", "xi-left", "xi-delta"):
        for shell_count in (1, 2, 3):
            report = report_r_multishell_split_effect(
                seq,
                extracted_mass,
                dirichlet=config.dirichlet,
                angular_variant=angular_variant,
                shell_count=shell_count,
            )
            report["split"]["angular_variant"] = angular_variant
            sweep_reports.append(report)
    print_r_multishell_sweep_summary(sweep_reports)

    plot_matrix("log10 |E1|", extraction)
    plot_matrix("log10 |E1 M1 E1^T|", extracted_mass)
    plot_matrix("log10 |T E1|", transform_report["transformed_extraction"])
    plot_matrix("log10 |T (E1 M1 E1^T) T^T|", transform_report["transformed_from_matrix"])
    plot_matrix("log10 |T_rshell (E1 M1 E1^T) T_rshell^T|", r_shell_report["transformed_matrix"])


if __name__ == "__main__":
    main()