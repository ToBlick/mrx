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

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
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


def compiled_when_eager(fn):
    """``fn`` run as one compiled program (``eqx.filter_jit``) when called from the host with concrete arrays, and
    inlined into the trace when any argument is traced.

    Run eagerly, every operation of ``fn`` compiles on its own, once per shape: during the setup of a sequence
    these small programs are most of the time on a TPU. Inside a jitted function a nested jit saves nothing, and
    one that takes a whole sequence as input made the compile of the enclosing solve 1.5 to 2 times longer (li383
    12x20x12, 2026-10-18). Python scalars among the arguments are static, as under ``eqx.filter_jit``."""
    compiled = eqx.filter_jit(fn)

    @functools.wraps(fn)
    def call(*args, **kwargs):
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves((args, kwargs))):
            return fn(*args, **kwargs)
        return compiled(*args, **kwargs)
    return call


def to_device(x, dtype=None):
    """``jnp.asarray(x, dtype)`` that moves a NumPy array to the device without a compile.

    ``jnp.asarray`` of a NumPy array runs a compiled identity program in JAX 0.11, one per shape and dtype. The
    setup of a sequence moves hundreds of host tables of distinct shapes, so these programs add up on a TPU.
    ``jax.device_put`` copies them without one and gives the same array (dtype, weak type). Anything other than
    a NumPy array goes to ``jnp.asarray``."""
    if isinstance(x, (np.ndarray, np.generic)):
        return jax.device_put(np.asarray(x, dtype=dtype))
    return jnp.asarray(x, dtype=dtype)
