"""
Hydra configuration dataclasses for MRX.

All configuration is defined here as dataclasses registered with Hydra's
ConfigStore.  Scripts use ``@hydra.main(config_name=..., version_base=None)``
with *no* ``config_path`` — defaults are picked up from the store.

Usage from CLI::

    python scripts/config_scripts/relax_from_nfs.py              # defaults
    python scripts/config_scripts/relax_from_nfs.py fem.ns_r=16
    python scripts/config_scripts/relax_from_nfs.py resolution=high
    python scripts/config_scripts/relax_from_nfs.py -m fem.ns_r=8,12,16

Resolution group overrides:  ``resolution=low`` / ``medium`` / ``high``
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

# ---------------------------------------------------------------------------
#  Shared sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FEMConfig:
    ns_r: int = 8
    ns_theta: int = 8
    ns_zeta: int = 8
    ps_r: int = 4
    ps_theta: int = 4
    ps_zeta: int = 4
    quad_order: int = 4


@dataclass
class MapConfig:
    ns_r: int = 8
    ns_theta: int = 8
    ns_zeta: int = 8
    ps_r: int = 4
    ps_theta: int = 4
    ps_zeta: int = 4
    quad_order: int = 4
    flip_zeta: bool = False


@dataclass
class RelaxationConfig:
    num_iters_inner: int = 100
    num_iters_outer: int = 100
    dt0: float = 1.0
    force_tolerance: float = 1e-9
    descent_method: str = "gradient"  # gradient | conjugate_gradient | newton | bfgs


@dataclass
class EtaConfig:
    max: float = 0.0
    schedule_type: str = "tanh"  # tanh | constant | linear


@dataclass
class NoiseConfig:
    max: float = 0.0
    schedule_type: str = "tanh"  # tanh | constant | linear
    key: int = 42


@dataclass
class InterpolationConfig:
    val_stride: int = 4
    exclude_axis_tol: float = 1e-3


@dataclass
class OutputConfig:
    dir: str = "out"
    save_every: int = 1
    save_final: bool = True
    verbose: bool = True
    print_every: int = 1


@dataclass
class GeometryConfig:
    eps: float = 0.33
    kappa: float = 1.1
    nfp: int = 3


@dataclass
class InitialFieldConfig:
    q_star: float = 1.54


@dataclass
class FieldlineConfig:
    enabled: bool = True
    every: int = 100
    T: float = 2500.0
    n_traj: int = 32
    rtol: float = 1e-5
    atol: float = 1e-5


@dataclass
class StellPlottingConfig:
    zeta_values: list[float] = field(default_factory=lambda: [0.33])
    interpolation_degree: int = 3
    markersize: float = 0.1
    cmap_iota: str = "berlin"
    cmap_p: str = "plasma"
    ks_thresh: int = 10
    denom_max: int = 15
    dpi: int = 150
    Rlim: list[float] = field(default_factory=lambda: [0.6, 1.4])
    zlim: list[float] = field(default_factory=lambda: [-0.4, 0.4])


@dataclass
class PoincareFieldlineConfig:
    n_scan: int = 1
    n_vmap: int = 32
    T_factor: int = 300
    axis_margin: float = 0.05


@dataclass
class PoincareClassificationConfig:
    zeta_values: list[float] = field(default_factory=lambda: [0.33])
    ks_thresh: int = 10


@dataclass
class PoincarePlottingConfig:
    markersize: float = 0.005
    dpi: int = 150
    denom_max: int = 15
    cmap_iota: str = "nipy_spectral"
    cmap_p: str = "plasma"
    plot_pressure: bool = True
    Rlim: Optional[list[float]] = None   # None -> auto
    zlim: Optional[list[float]] = None   # None -> auto


@dataclass
class PoincareOutputConfig:
    subdir: str = "poincare_plots"
    format: str = "pdf"
    verbose: bool = True


# ---------------------------------------------------------------------------
#  Top-level configs
# ---------------------------------------------------------------------------

@dataclass
class RelaxFromNFSConfig:
    """Config for ``relax_from_nfs.py``.

    By default the ``resolution: low`` preset is composed in (overrides
    ``map.ns_*`` and ``fem.ns_*``).  Pass ``resolution=medium`` etc. on the
    CLI to change.
    """
    defaults: list[Any] = field(default_factory=lambda: [
        "_self_",
        {"resolution": "low"},
    ])

    run_name: Optional[str] = None
    nfs_file: str = "data/gvec_w7x.h5"
    nfp: int = 5

    map: MapConfig = field(default_factory=MapConfig)
    fem: FEMConfig = field(default_factory=FEMConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    relaxation: RelaxationConfig = field(default_factory=RelaxationConfig)
    eta: EtaConfig = field(default_factory=EtaConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(
        dir="out/relax_from_nfs"))


@dataclass
class RelaxStellConfig:
    """Config for ``relax_stell.py``."""
    run_name: Optional[str] = None

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    initial_field: InitialFieldConfig = field(default_factory=InitialFieldConfig)
    fem: FEMConfig = field(default_factory=lambda: FEMConfig(
        ns_r=6, ns_theta=10, ns_zeta=6,
        ps_r=3, ps_theta=3, ps_zeta=3, quad_order=3))
    relaxation: RelaxationConfig = field(default_factory=lambda: RelaxationConfig(
        num_iters_inner=10, descent_method="conjugate_gradient",
        force_tolerance=1e-6))
    eta: EtaConfig = field(default_factory=lambda: EtaConfig(max=1e-6))
    fieldline: FieldlineConfig = field(default_factory=FieldlineConfig)
    plotting: StellPlottingConfig = field(default_factory=StellPlottingConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(
        dir="out/stell_relaxation", print_every=10))


@dataclass
class PoissonTestConfig:
    """Application parameters for ``test_torus_poisson_sparse.py``."""
    n: int = 8
    p: int = 3
    epsilon: float = 1 / 3
    cg_tol: float = 1e-9
    cg_maxiter: int = 100_000
    map_batch_size_inner: int = 0      # 0 corresponds to vmap
    map_batch_size_outer: Optional[int] = None    # None means no batching


@dataclass
class MCPoissonConfig:
    """Application parameters for ``scripts/dice/mc_poisson.py``."""
    n: int = 10
    p: int = 3
    N: list[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10_000])
    outer_batch_size: Optional[int] = None
    inner_batch_size: Optional[int] = 10_000
    seed: int = 42
    replicates: int = 1


@dataclass
class PoincarePlotsConfig:
    """Config for ``poincare_plots.py``.

    Tip: pass ``hydra.job.chdir=false hydra.run.dir=.`` on the CLI to
    keep the working directory unchanged and avoid creating an output dir.
    """
    run_dir: Optional[str] = None

    fieldline: PoincareFieldlineConfig = field(default_factory=PoincareFieldlineConfig)
    poincare: PoincareClassificationConfig = field(default_factory=PoincareClassificationConfig)
    plotting: PoincarePlottingConfig = field(default_factory=PoincarePlottingConfig)
    output: PoincareOutputConfig = field(default_factory=PoincareOutputConfig)


# ---------------------------------------------------------------------------
#  Resolution presets  (registered as group overrides at _global_ package)
# ---------------------------------------------------------------------------

_low_res = {"map": {"ns_r": 4, "ns_theta": 4, "ns_zeta": 4},
            "fem": {"ns_r": 4, "ns_theta": 4, "ns_zeta": 4}}

_medium_res = {"map": {"ns_r": 5, "ns_theta": 8, "ns_zeta": 8},
               "fem": {"ns_r": 6, "ns_theta": 10, "ns_zeta": 8}}

_high_res = {"map": {"ns_r": 6, "ns_theta": 12, "ns_zeta": 12},
             "fem": {"ns_r": 8, "ns_theta": 16, "ns_zeta": 12}}


# ---------------------------------------------------------------------------
#  ConfigStore registration  (executed on import)
# ---------------------------------------------------------------------------

def _register() -> None:
    cs = ConfigStore.instance()

    # Main configs
    cs.store(name="config_relax_from_nfs", node=RelaxFromNFSConfig)
    cs.store(name="config_stell",         node=RelaxStellConfig)
    cs.store(name="config_poincare",      node=PoincarePlotsConfig)
    cs.store(name="_poisson_test_schema",  node=PoissonTestConfig)
    cs.store(name="_mc_poisson_schema",     node=MCPoissonConfig)

    # Resolution group (usage: ``resolution=low``)
    for name, node in [("low", _low_res), ("medium", _medium_res), ("high", _high_res)]:
        cs.store(group="resolution", name=name, node=node, package="_global_")


_register()
