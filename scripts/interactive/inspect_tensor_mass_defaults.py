"""Inspect the production tensor mass-preconditioner defaults.

Run from the project root:

    .venv/bin/python scripts/interactive/inspect_tensor_mass_defaults.py
"""

from __future__ import annotations

import inspect

from mrx.operators import assemble_tensor_mass_preconditioner
from mrx.preconditioners import (
    TensorMassPreconditioner,
    build_mass_tensor_preconditioner,
    default_mass_preconditioner,
)


def main() -> None:
    spec = default_mass_preconditioner()
    tensor_fields = TensorMassPreconditioner.__dataclass_fields__
    print("Default mass preconditioner spec")
    print(f"  kind={spec.kind}")
    print(f"  surgery_schur={spec.surgery_schur}")
    print(f"  lanczos_iterations={spec.lanczos_iterations}")
    print()

    print("Tensor smoother defaults")
    print(f"  block_chebyshev_steps={tensor_fields['block_chebyshev_steps'].default}")
    print(f"  block_lanczos_iterations={tensor_fields['block_lanczos_iterations'].default}")
    print(
        "  build_mass_tensor_preconditioner.rank="
        f"{inspect.signature(build_mass_tensor_preconditioner).parameters['rank'].default}"
    )
    print(
        "  assemble_tensor_mass_preconditioner.rank="
        f"{inspect.signature(assemble_tensor_mass_preconditioner).parameters['rank'].default}"
    )


if __name__ == "__main__":
    main()