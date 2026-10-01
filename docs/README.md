# MRX documentation

`docs/source/` is the Sphinx tree: the guides (`getting_started.md`,
`tutorials.md`, `relaxation.md`, `cluster.md`, `sharp_bits.md`, `faq.md`), the
concept pages under `concepts/` and the API reference under `api/`.

```
pip install -r docs/requirements.txt
make -C docs html        # -> docs/build/html/index.html
```

`docs/research/` is the record of measurements behind the design, written
for coding agents rather than for people (terse lists of settings and
numbers). It is not part of the Sphinx build.
