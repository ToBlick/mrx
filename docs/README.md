# MRX documentation

Everything is in `docs/source/`, the Sphinx tree: the guides
(`getting_started.md`, `tutorials.md`, `poisson.md`, `relaxation.md`,
`cluster.md`), the concept pages under `concepts/` (architecture, assembly,
preconditioning, the polar axis, precision, the relaxation loop, the GVEC
interface, the manufactured solutions, production settings, the testing
strategy) and the API reference under `api/`. Every identifier named there
exists in `mrx/` or `scripts/`. Build it with

```
pip install -r docs/requirements.txt
make -C docs html        # -> docs/build/html/index.html
```

`docs/research/` is the campaign record -- handoffs, plans, measurements,
refuted approaches. Its `README.md` indexes it by topic and `OPEN.md` lists
every open item once. It is not part of the Sphinx build.
