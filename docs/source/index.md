# MRX

MRX is a 3D MHD equilibrium code based on admissible variations. It does
not assume nested flux surfaces. It is written in Python with JAX and
discretises the de Rham complex with tensor-product B-splines on a mapped
logical cube.

The source is at [github.com/ToBlick/mrx](https://github.com/ToBlick/mrx).
If MRX is useful for your work, please cite the
[preprint](https://arxiv.org/abs/2510.26986):

```bibtex
@article{blickhan_mrx_2025,
    title = {{MRX}: {A} differentiable {3D} {MHD} equilibrium solver without nested flux surfaces},
    url = {http://arxiv.org/abs/2510.26986},
    doi = {10.48550/arXiv.2510.26986},
    publisher = {arXiv},
    author = {Blickhan, Tobias and Stratton, Julianne and Kaptanoglu, Alan A.},
    month = oct,
    year = {2025},
}
```

```{toctree}
:maxdepth: 1
:caption: Guides

getting_started
tutorials
relaxation
cluster
sharp_bits
faq
```

```{toctree}
:maxdepth: 1
:caption: Concepts

concepts/architecture
concepts/mass
concepts/polar
concepts/preconditioning
concepts/precision
concepts/relaxation
concepts/equilibrium_input
concepts/testing_strategy
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
```

`docs/research` in the repository holds the record of measurements behind the design. It is written for
coding agents working on the code, not for reading.
