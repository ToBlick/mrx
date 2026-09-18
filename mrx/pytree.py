"""Plain classes as JAX pytrees: their device arrays flow through ``jit`` as
ARGUMENTS instead of being baked into the program as constants.

A ``jax.jit`` over a closure captures every array it reaches as a constant
of the compiled program. For a relaxation chunk that reached the whole
sequence: the metric at every quadrature point, the element weights of the
mass and projection applies, the extraction tables -- 3.5 GB of constants
at (48,96,96), 10.6 GB for the whole torus, which XLA constant-folds (ten
seconds a scatter) and holds on the host through the compile (300 GB for
the torus). A pytree argument is a device buffer the program reads.

:func:`register_arrays` makes a class a pytree whose children are its
attributes (the device arrays and every nested pytree among them) and
whose static part is the rest: host arrays (numpy), Python scalars,
strings, functions, and the attributes named ``static``. Combined with
``eqx.filter_jit`` at the boundary -- array leaves traced, everything else
static -- a function of the sequence is pure: the same code, the arrays
as inputs.

Unflattening builds a shallow copy (``object.__new__`` and the attribute
dict), so inside a trace the methods run on an object whose arrays are
tracers. A lazily filled cache on such a copy fills the copy, not the
original: caches are warmed before the jit (the setup does).

The static part compares by the IDENTITY of its values (``is``), with
scalars, strings and tuples by value: two flattenings of the same object
give equal static parts as long as its non-array attributes are not
replaced, which is what keeps ``jit``'s cache hitting across calls.
"""
from __future__ import annotations

import jax
import numpy as np

_BY_VALUE = (int, float, bool, str, bytes, tuple, frozenset, type(None))


class Static:
    """The non-array attributes of a flattened object, as one hashable
    treedef component: equal when the same attributes hold the same
    objects (by identity, scalars and tuples by value)."""

    __slots__ = ("items", "_hash")

    def __init__(self, items):
        self.items = tuple(items)          # ((name, value), ...) in a fixed order
        self._hash = hash(tuple((k, v if isinstance(v, _BY_VALUE) else id(v))
                                for k, v in self.items))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Static) or len(self.items) != len(other.items):
            return False
        for (k, v), (k2, v2) in zip(self.items, other.items):
            if k != k2:
                return False
            if v is v2:
                continue
            if isinstance(v, _BY_VALUE) and isinstance(v2, _BY_VALUE) and v == v2:
                continue
            return False
        return True


def _is_static(name, value, static_names):
    return name in static_names or isinstance(value, np.ndarray)


def register_arrays(cls, static=()):
    """Register ``cls`` as a pytree of its attributes: device arrays and
    nested pytrees are children, ``static`` names and host arrays are not.
    Non-array children (ints, functions, other objects) are pytree leaves;
    ``eqx.filter_jit`` keeps those static and traces the arrays.
    Returns ``cls``, so it can decorate."""
    static_names = frozenset(static)

    def flatten_with_keys(obj):
        d = vars(obj)
        names = sorted(d)
        dyn = [n for n in names if not _is_static(n, d[n], static_names)]
        children = [(jax.tree_util.GetAttrKey(n), d[n]) for n in dyn]
        aux = (tuple(dyn), Static((n, d[n]) for n in names if _is_static(n, d[n], static_names)))
        return children, aux

    def flatten(obj):
        children, aux = flatten_with_keys(obj)
        return [c for _, c in children], aux

    def unflatten(aux, children):
        dyn, static_part = aux
        obj = object.__new__(cls)
        d = vars(obj)
        d.update(static_part.items)
        d.update(zip(dyn, children))
        return obj

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls
