A GVEC state in GVEC's G-frame (`which_hmap = 21`), for the reader tests.

The frame and boundary (`N2-12/N2.12-v3.1-hi.nc`) are GVEC's test-CI case `axisNB_N2-12`
(gitlab.mpcdf.mpg.de/gvec-group/gvec, commit c0dc66f, MIT license, `LICENSE_GVEC.txt`). The state is a
current-free vacuum field (beta 0, `I_tor = 0`) computed with GVEC 1.5.0 from `N2-12/parameters.toml`
(`pygvec run parameters.toml` in that directory).

The frame in this file is orthonormal only to about 1e-5 and stellarator-symmetric only to about 1e-6, so
`is_stellarator_symmetric` reports the state as not symmetric.
