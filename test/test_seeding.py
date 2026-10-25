"""energy_seed names the problem when the iota range holds no resonance of the chains it can seed."""
import os

import pytest

from mrx.geometry import build_sequence
from mrx.relaxation.initial_conditions import initial_field
from mrx.relaxation.seeding import energy_seed

WOUT = os.path.join(os.path.dirname(__file__), "..", "data", "wout_li383_low_res_reference.nc")


def test_no_resonance_in_the_iota_range_is_reported():
    # li383 has nfp = 3 and iota in [0.405, 0.658]. With n_theta = 8 the chains have m <= 4, and none of
    # nfp n / m = 0.75, 1, 1.5, 3 lies in that range. (4, 8, 8) is the smallest mesh that builds for this file.
    seq = build_sequence(WOUT, (4, 8, 8), 2)
    B, _ = initial_field(seq)
    with pytest.raises(ValueError, match=r"no island chain with m <= 4 .*\[0\.4054, 0\.6583\] \(nfp = 3\)"):
        energy_seed(seq, B, iotas=(1.0,), verbose=False)
