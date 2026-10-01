"""Turn plain Python classes into JAX pytrees, so that their device arrays are passed to a jitted
function as inputs instead of being baked into the compiled code as constants.

:func:`register_arrays` is the only function a user needs. On a registered class, the JAX arrays
and nested pytrees are the dynamic part. Numpy arrays and the attributes named in ``static`` are
the static part. Scalars, strings and functions become leaves that ``eqx.filter_jit`` treats as
static. The consequences for use are these.

- Under ``eqx.filter_jit``, a function of such an object sees its arrays as traced inputs, so new
  array values (for example a new geometry) reuse the compiled code.
- Changing anything in the static part triggers a recompile. Scalars, strings and tuples in it
  compare by value, every other object by identity, so replacing such an object by an equal copy
  also recompiles.
"""
from __future__ import annotations

import jax
import numpy as np

_BY_VALUE = (int, float, bool, str, bytes, tuple, frozenset, type(None))


class Static:
    """The static attributes of a flattened object, as one hashable value.

    Two instances are equal when the same attributes hold the same objects (compared by value for
    scalars, strings and tuples, by identity otherwise)."""

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
    """Register ``cls`` as a pytree of its instance attributes and return ``cls``, so this can be
    used as a class decorator.

    The attributes named in ``static`` and all numpy arrays are static. All other attributes are
    pytree children. Of those, ``eqx.filter_jit`` traces the JAX arrays and treats the rest (ints,
    functions, other objects) as static."""
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
