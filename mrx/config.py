"""Hydra configuration dataclasses for MRX.

Every configuration is a dataclass registered with Hydra's ConfigStore. This
module imports no JAX, so importing it never creates an array.

One entry point uses it: ``scripts/poisson_study.py`` composes
``conf/config_poisson_test.yaml`` on top of the ``_poisson_test_schema`` node
(:class:`PoissonTestConfig`). The yaml overrides a few dataclass defaults and
configures the submitit launcher for multiruns; the dataclass is the typed
schema.

Every top-level config inherits :class:`NumericsConfig`, which carries the two
numerics knobs shared by all entry points:

``precision``
    ``float64`` (default) or ``float32``. The entry point exports it as
    ``MRX_DTYPE`` before ``import mrx``; the switch itself lives in
    :mod:`mrx.precision`, and this field is its record in the run config.
``solver_tol``
    Relative residual tolerance of every iterative solve that goes through
    :class:`~mrx.derham_sequence.DeRhamSequence`. ``None`` selects
    ``sqrt(eps)`` of the working precision. This is the single tolerance
    knob; the former ``PoissonTestConfig.cg_tol`` is this field.

Usage from CLI::

    python scripts/poisson_study.py p=3
    python scripts/poisson_study.py p=3 precision=float32
    python scripts/poisson_study.py -m p=2,3 n=8,16
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class NumericsConfig:
    """Working precision and solver tolerance; base of every top-level config.

    ``precision`` is ``float64`` or ``float32`` and must match ``MRX_DTYPE``
    at import time (the entry point sets it from the ``precision=`` override
    before importing mrx). ``solver_tol`` is passed as
    ``DeRhamSequence(tol=cfg.solver_tol)``; ``None`` means ``sqrt(eps)`` of
    the working precision.
    """
    precision: str = "float64"
    solver_tol: Optional[float] = None


@dataclass
class PoissonTestConfig(NumericsConfig):
    """Application parameters for ``scripts/poisson_study.py``."""
    n: Any = field(default_factory=lambda: [8, 12, 16, 24, 32, 48, 64])
    p: int = 3
    epsilon: float = 1 / 3
    quad_order: Optional[int] = None
    quad_order_offset: int = 0
    cg_maxiter: int = 100_000
    map_batch_size_inner: int = 0      # 0 corresponds to vmap
    map_batch_size_outer: Optional[int] = None    # None means no batching
    load_frame: str = 'ref'           # 'ref' or 'phys' (see mrx.projectors.load)


def _register() -> None:
    ConfigStore.instance().store(name="_poisson_test_schema", node=PoissonTestConfig)


_register()
