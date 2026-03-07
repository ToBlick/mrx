"""
Stellarator Relaxation script using Hydra for configuration management.

This uses an analytical rotating ellipse map and analytical initial B-field,
unlike relax_from_nfs.py which loads data from GVEC files.

Usage: 
    # Single run with defaults
    python scripts/config_scripts/relax_stell.py
    
    # Override parameters
    python scripts/config_scripts/relax_stell.py fem.ns_r=16 fem.ns_theta=32
    
    # Multirun sweep
    python scripts/config_scripts/relax_stell.py -m fem.ns_r=8,12,16 eta.max=1e-6,1e-7

    # With custom run name
    python scripts/config_scripts/relax_stell.py run_name=my_experiment
"""
import time
from pathlib import Path

import h5py
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

import mrx.config  # noqa: F401  —  register Hydra structured configs
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.io import unique_id
from mrx.mappings import rotating_ellipse_map
from mrx.plotting import integrate_fieldline, poincare_plot
from mrx.relaxation import (
    DescentMethod,
    IntegrationScheme,
    MRXDiagnostics,
    TimeStepChoice,
    TimeStepper,
    relaxation_loop,
)
from mrx.utils import default_trace_dict

jax.config.update("jax_enable_x64", True)


def create_eta_schedule(cfg: DictConfig):
    """Create resistivity schedule function based on config."""
    eta_max = cfg.eta.max
    num_iters_outer = cfg.relaxation.num_iters_outer
    schedule_type = cfg.eta.schedule_type

    if schedule_type == "tanh":
        def eta_schedule(iter_outer):
            return eta_max * 0.5 * (1 - jnp.tanh(4 * jnp.pi * (iter_outer / num_iters_outer - 0.5)))
    elif schedule_type == "constant":
        def eta_schedule(iter_outer):
            return eta_max
    elif schedule_type == "linear":
        def eta_schedule(iter_outer):
            return eta_max * (1 - iter_outer / num_iters_outer)
    else:
        raise ValueError(f"Unknown eta.schedule_type: {schedule_type}")

    return eta_schedule


def create_fieldline_callback(seq, diagnostics, nfp, cfg: DictConfig, outdir: Path):
    """
    Create a callback function for fieldline integration during relaxation.

    This callback:
    1. Integrates magnetic field lines
    2. Evaluates |B| along the field lines
    3. Evaluates pressure p along the field lines
    4. Saves results to HDF5
    5. Generates Poincare plots
    """
    fieldline_every = cfg.fieldline.every
    fieldline_T = cfg.fieldline.T
    fieldline_n_traj = cfg.fieldline.n_traj
    fieldline_rtol = cfg.fieldline.rtol
    fieldline_atol = cfg.fieldline.atol

    # Pre-compile functions for efficiency
    get_pressure = jax.jit(diagnostics.pressure)

    def compute_B_norm(B_dof):
        """Create a function to compute |B| at a point given B_dof."""
        B_h = DiscreteFunction(B_dof, seq.Lambda_2, seq.E2)

        @jax.jit
        def B_norm_fn(x):
            x = x % 1.0
            Bx = B_h(x)
            DFx = jax.jacfwd(seq.F)(x)
            return jnp.linalg.norm(DFx @ Bx) / jnp.linalg.det(DFx)

        return B_norm_fn

    def compute_pressure_fn(p_dof):
        """Create a function to compute pressure at a point given p_dof."""
        p_h = Pushforward(DiscreteFunction(
            p_dof, seq.Lambda_0, seq.E0), seq.F, 0)

        @jax.jit
        def p_fn(x):
            x = x % 1.0
            return p_h(x)

        return p_fn

    def fieldline_callback(state, iteration):
        """
        Callback function called after each outer iteration.
        Integrates fieldlines and evaluates B-norm and pressure along them.
        """
        if iteration % fieldline_every != 0:
            return state

        print(
            f"  [Callback] Integrating fieldlines at iteration {iteration}...")

        B_dof = state.B_n

        # Create discrete B field for integration
        B_h = jax.jit(DiscreteFunction(B_dof, seq.Lambda_2, seq.E2))

        # Integrate fieldlines
        logical_trajectories, physical_trajectories = integrate_fieldline(
            B_h, seq.F, nfp,
            T=fieldline_T,
            n_traj=fieldline_n_traj,
            rtol=fieldline_rtol,
            atol=fieldline_atol,
        )

        # Compute |B| along fieldlines
        B_norm_fn = compute_B_norm(B_dof)
        flat_logical = logical_trajectories.reshape(-1, 3)
        B_norm_values = jax.vmap(B_norm_fn)(flat_logical)
        B_norm_values = B_norm_values.reshape(logical_trajectories.shape[:-1])

        # Compute pressure along fieldlines
        p_dof = get_pressure(B_dof)
        p_fn = compute_pressure_fn(p_dof)
        p_values = jax.vmap(p_fn)(flat_logical)
        p_values = p_values.reshape(logical_trajectories.shape[:-1])

        # Save to HDF5
        fieldline_file = outdir / f"fieldlines_iter{iteration:05d}.h5"
        print(f"  [Callback] Saving fieldlines to {fieldline_file}")

        with h5py.File(fieldline_file, "w") as f:
            f.create_dataset("logical_trajectories",
                             data=jnp.array(logical_trajectories))
            f.create_dataset("physical_trajectories",
                             data=jnp.array(physical_trajectories))
            f.create_dataset("B_norm", data=jnp.array(B_norm_values))
            f.create_dataset("pressure", data=jnp.array(p_values))
            f.attrs["iteration"] = iteration
            f.attrs["nfp"] = nfp
            f.attrs["T"] = fieldline_T
            f.attrs["n_traj"] = fieldline_n_traj
            f.attrs["force_norm"] = float(state.F_norm)

        print(
            f"  [Callback] |B| range: [{float(jnp.min(B_norm_values)):.4f}, {float(jnp.max(B_norm_values)):.4f}]")
        print(
            f"  [Callback] p range: [{float(jnp.min(p_values)):.4f}, {float(jnp.max(p_values)):.4f}]")

        # Generate Poincare plot
        print("  [Callback] Generating Poincare plots...")
        try:
            # Get plot limits from config (convert to tuple or None)
            Rlim = tuple(
                cfg.plotting.Rlim) if cfg.plotting.Rlim is not None else None
            zlim = tuple(
                cfg.plotting.zlim) if cfg.plotting.zlim is not None else None

            for zeta_value in cfg.plotting.zeta_values:
                fig, axes = poincare_plot(
                    logical_trajectories,
                    seq.F,
                    nfp,
                    p_h=p_fn,
                    zeta_value=zeta_value,
                    interpolation_degree=cfg.plotting.interpolation_degree,
                    markersize=cfg.plotting.markersize,
                    cmap_iota=cfg.plotting.cmap_iota,
                    cmap_p=cfg.plotting.cmap_p,
                    ks_thresh=cfg.plotting.ks_thresh,
                    denom_max=cfg.plotting.denom_max,
                    Rlim=Rlim,
                    zlim=zlim,
                    rasterized=True,
                )
                plot_file = outdir / \
                    f"poincare_iter{iteration:05d}_zeta{zeta_value:.2f}.pdf"
                fig.savefig(plot_file, dpi=cfg.plotting.dpi,
                            bbox_inches="tight")
                plt.close(fig)
                print(f"  [Callback] Saved Poincare plot to {plot_file}")
        except Exception as e:
            print(
                f"  [Callback] Warning: Could not generate Poincare plot: {e}")

        return state

    return fieldline_callback


def create_initial_B_field(F, tau):
    """
    Create the initial magnetic field in physical space.

    This is an analytical initial condition for a stellarator field.

    Parameters
    ----------
    F : callable
        The mapping function from logical to physical coordinates.
    tau : float
        Safety factor / rotational transform parameter (q_star).

    Returns
    -------
    B_xyz : callable
        Function that computes B(x,y,z) given logical coordinates p.
    """
    def B_xyz(p):
        x, y, z = F(p)
        R = (x**2 + y**2)**0.5
        phi = jnp.arctan2(y, x)
        BR = z * R
        Bphi = tau / R
        Bz = - (1 / 2 * (R**2 - 1**2) + z**2)
        Bx = BR * jnp.cos(phi) - Bphi * jnp.sin(phi)
        By = BR * jnp.sin(phi) + Bphi * jnp.cos(phi)
        return jnp.array([Bx, By, Bz])

    return B_xyz


@hydra.main(version_base=None, config_name="config_stell")
def main(cfg: DictConfig) -> float:
    """
    Main entry point for stellarator relaxation with Hydra configuration.

    Returns the final force norm (useful for Hydra optimization).
    """
    # Hydra automatically changes to the output directory
    outdir = Path.cwd()

    # Generate run name if not provided
    run_name = cfg.run_name if cfg.run_name else unique_id(8)

    print("=" * 60)
    print(f"Stellarator Relaxation: {run_name}")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    start_time = time.time()

    # Initialize trace dictionary
    trace_dict = default_trace_dict.copy()
    trace_dict["start_time"] = start_time

    # Create analytical map
    eps = cfg.geometry.eps
    kappa = cfg.geometry.kappa
    nfp = cfg.geometry.nfp

    print(
        f"Creating rotating ellipse map with eps={eps}, kappa={kappa}, nfp={nfp}...")
    F = rotating_ellipse_map(eps, kappa, nfp)
    F = jax.jit(F)

    # Setup FEM spaces
    ns = (cfg.fem.ns_r, cfg.fem.ns_theta, cfg.fem.ns_zeta)
    ps = (cfg.fem.ps_r, cfg.fem.ps_theta, cfg.fem.ps_zeta)

    # Validate that ps_x <= ns_x - 1
    ps = tuple(min(p, n - 1) for p, n in zip(ps, ns))

    # Determine zeta types based on ns_zeta
    if ns[2] == 1:
        types = ("clamped", "periodic", "constant")
        ps = (ps[0], ps[1], 0)
    else:
        types = ("clamped", "periodic", "periodic")

    print(f"Setting up FEM spaces with ns={ns}, ps={ps}...")
    seq = DeRhamSequence(
        ns, ps, cfg.fem.quad_order,
        types,
        F, polar=True, dirichlet=True
    )

    # Check that mapping is not singular
    assert jnp.min(seq.jacobian_j) > 0, "Mapping is singular!"

    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()

    setup_time = time.time()
    trace_dict["setup_done_time"] = setup_time
    print(f"Setup completed in {setup_time - start_time:.2f}s")

    # Create initial B-field
    tau = cfg.initial_field.q_star
    print(f"Creating initial B-field with q_star={tau}...")
    B_xyz = create_initial_B_field(F, tau)

    # Project to FEM space
    B_dof_0 = jnp.linalg.solve(seq.m2, seq.P2(B_xyz))
    B_dof_0 = seq.P_Leray @ B_dof_0

    # Normalize
    B_norm_initial = (B_dof_0 @ seq.m2 @ B_dof_0)**0.5
    B_dof_0 /= B_norm_initial
    print(f"Initial B-field norm: {B_norm_initial:.6f}")

    B_dof = B_dof_0.copy()

    div_B_initial = float(((seq.strong_div @ B_dof_0) @
                          seq.m3 @ (seq.strong_div @ B_dof_0))**0.5)
    print(f"div B after projection: {div_B_initial:.2e}")

    # Setup relaxation
    print("Setting up time stepper...")
    ts = TimeStepper(
        seq=seq,
        descent_method=DescentMethod[cfg.relaxation.descent_method.upper()],
        dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
        timestep_mode=IntegrationScheme.EXPLICIT,
    )

    eta_schedule = create_eta_schedule(cfg)

    # Setup diagnostics (needed for callback)
    diagnostics = MRXDiagnostics(seq)

    # Setup fieldline callback if enabled
    callback = None
    if cfg.fieldline.enabled:
        print(
            f"Fieldline callback enabled: tracing every {cfg.fieldline.every} iterations")
        callback = create_fieldline_callback(
            seq, diagnostics, nfp, cfg, outdir)

    # Run relaxation loop
    print(
        f"Starting relaxation: {cfg.relaxation.num_iters_outer} outer x {cfg.relaxation.num_iters_inner} inner iterations...")

    final_state, traces = relaxation_loop(
        B_dof,
        ts,
        num_iters_inner=cfg.relaxation.num_iters_inner,
        num_iters_outer=cfg.relaxation.num_iters_outer,
        dt0=cfg.relaxation.dt0,
        force_tolerance=cfg.relaxation.force_tolerance,
        resistivity_schedule=eta_schedule,
        callback=callback,
    )

    end_time = time.time()
    trace_dict["end_time"] = end_time

    # Final diagnostics
    B_dof_final = final_state.B_n

    final_force = float(traces["force_norm"][-1])
    final_energy = float(traces["energy"][-1])
    final_helicity = float(traces["helicity"][-1])
    helicity_change = abs(
        (final_helicity - traces["helicity"][0]) / traces["helicity"][0])

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Final force norm: {final_force:.2e}")
    print(f"Final energy: {final_energy:.6f}")
    print(f"Relative helicity change: {helicity_change:.2e}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print("=" * 60)

    # Save results
    if cfg.output.save_final:
        output_file = outdir / f"{run_name}.h5"
        print(f"Saving results to {output_file}...")
        with h5py.File(output_file, "w") as f:
            f.create_dataset("B_dof_initial", data=B_dof_0)
            f.create_dataset("B_dof_final", data=B_dof_final)
            f.create_dataset(
                "force_trace", data=jnp.array(traces["force_norm"]))
            f.create_dataset("energy_trace", data=jnp.array(traces["energy"]))
            f.create_dataset("helicity_trace",
                             data=jnp.array(traces["helicity"]))
            f.create_dataset("timestep_trace",
                             data=jnp.array(traces["timestep"]))
            # Save geometry info
            f.attrs["eps"] = eps
            f.attrs["kappa"] = kappa
            f.attrs["nfp"] = nfp
            f.attrs["q_star"] = tau
            # Save flattened config
            f.attrs["config"] = OmegaConf.to_yaml(cfg)

    print("\nDone!")

    # Return final force norm for Hydra optimization
    return final_force


if __name__ == "__main__":
    main()
