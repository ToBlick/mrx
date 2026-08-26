"""Back-compatible re-export shim.

Nothing is defined here any more. The names below moved to focused modules and
are re-exported so that old import sites (scripts, deprecated tests) keep
working. Import from the owning module in new code.

Emptied 2026-08-24: ``is_running_in_github_actions``,
``evaluate_at_xq_deprecated``, ``integrate_against_deprecated``,
``run_relaxation_loop`` and ``update_config`` were unreferenced anywhere in the
repo, and the last two were already broken -- they read the undefined names
``norm_2`` and ``DEVICE_PRESETS``. A commented-out ``solve_singular_cg`` draft
went with them. All of it is in git history.
"""

# Re-export symbols that have moved to other modules.
from mrx.differential_forms import (  # noqa: F401
    curl, det33, div, double_map, grad, inv33, jacobian_determinant, l2_product,
    safe_inv33,
)
from mrx.quadrature import evaluate_at_xq, integrate_against  # noqa: F401
from mrx.solvers import get_smallest_ev_pair  # noqa: F401

# Diagonal utilities have moved to mrx.preconditioners.
from mrx.preconditioners import (  # noqa: F401
    diag_EAET, diag_EAET_matvec, diag_matvec, diag_schur_complement,
)
