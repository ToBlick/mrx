# MRX documentation

Everything is in `docs/source/`, the Sphinx tree: the guides
(`getting_started.md`, `tutorials.md`, `relaxation.md`, `cluster.md`,
`faq.md`), the concept pages under `concepts/` (architecture, the mass
operators, the polar axis, preconditioning, precision, the relaxation loop,
the GVEC interface, the testing strategy) and the API reference under
`api/`. Build it with

```
pip install -r docs/requirements.txt
make -C docs html        # -> docs/build/html/index.html
```

`docs/research/` is the campaign record -- handoffs, plans, measurements,
refuted approaches. Its `README.md` indexes it by topic and `OPEN.md` lists
every open item once. It is not part of the Sphinx build.
