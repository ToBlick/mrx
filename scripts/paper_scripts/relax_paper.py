#!/usr/bin/env python
"""scripts/relax.py with the options the paper compared and the default script no longer exposes.

    python -u scripts/paper_scripts/relax_paper.py --geometry ... [relax.py's flags] [the paper's flags]

The same run (``relax.main``) on an EXTENDED configuration: :class:`PaperDescent` adds to
:class:`mrx.relax_config.Descent` the velocity smoothing order and scale, the CFL cap, the auxiliary
B field, the helicity correction and the potential velocity (all still :class:`mrx.relaxation.TimeStepper`
options), with relax.py's defaults. The launchers under runs/ call this where an arm needs one of them
(gradient.sh's smoothing, helicity and Leray arms, newton_convergence.sh's gradient arm); every other
arm runs scripts/relax.py itself.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from relax import PRECISIONS, main  # noqa: E402  (scripts/relax.py)


def _config_classes():
    from mrx.relax_config import Descent, RelaxConfig

    @dataclass(frozen=True)
    class PaperDescent(Descent):
        """The descent direction and the induction step, with the paper's variants."""
        velocity_smoothing_order: int = field(default=1, metadata=dict(
            help="descent direction v = (I - scale L)^-order F; 0 is off (the gradient descent then stops "
                 "conserving helicity after ~1e4 steps)"))
        velocity_smoothing_scale: Optional[float] = field(default=None, metadata=dict(
            help="length scale of the velocity smoothing [mrx.relaxation.SMOOTHING_C h_r^2]"))
        cfl: float = field(default=0.5, metadata=dict(
            help="cap the line-search step at cfl / (largest logical CFL number of the velocity); inf disables it"))
        auxiliary_B_field: bool = field(default=False, metadata=dict(
            flag="--auxiliary-B-field", help="route the cross products through the Dirichlet 1-form H = M_1^-1 P B"))
        helicity_correction: bool = field(default=False, metadata=dict(
            help="zero the step's discrete helicity change by one scalar correction of E"))
        potential_velocity: Optional[bool] = field(default=None, metadata=dict(
            help="the projected force as curl a + c h (k=1 Hodge solve) instead of the Leray solve [on for the "
                 "gradient descent on B; Newton and the auxiliary field have their own routes]"))

        def stepper_kwargs(self):
            return dict(super().stepper_kwargs(), velocity_smoothing_order=self.velocity_smoothing_order,
                        velocity_smoothing_scale=self.velocity_smoothing_scale, cfl=self.cfl,
                        auxiliary_B_field=self.auxiliary_B_field, helicity_correction=self.helicity_correction,
                        potential_velocity=self.potential_velocity)

    @dataclass(frozen=True)
    class PaperRelaxConfig(RelaxConfig):
        descent: PaperDescent = PaperDescent()

    return PaperRelaxConfig


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--precision", default="float32", choices=tuple(PRECISIONS))
    os.environ["MRX_DTYPE"], os.environ["MRX_RESIDUAL_DTYPE"] = PRECISIONS[pre.parse_known_args()[0].precision]
    from mrx.cli import parse
    main(parse(_config_classes(), description=__doc__))
