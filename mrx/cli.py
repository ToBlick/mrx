"""Command lines from dataclasses.

A script's parameters are frozen dataclasses: one class per group, the
defaults on the class, the help in the field metadata, the validation in
``__post_init__``. The command line is generated from them here, one
argparse group per nested dataclass in declaration order, so ``--help``
reads as the configuration and a mistyped flag fails at parse time. The
parsed line comes back as the dataclass, and :func:`flatten` /
:func:`unflatten` map it to the flat dict of a run record (``relax.json``
``params``): the record is the only configuration file, and it is never
written by hand.

Field conventions (``dataclasses.field(metadata=...)``):

``help``      the option's help; the default is appended in brackets
``flag``      the option string when it is not ``--<group prefix>-<name>``
              (``--seed`` for ``Seed.spec``, ``--auxiliary-B-field``)
``prefix``    on a nested dataclass field: the flag prefix of its options
              (``Newton`` with prefix ``newton``: ``--newton-tol``); no
              prefix keeps the field names
``parse``     ``str -> value`` for a structured value (``"32,64,64"`` to a
              tuple); such a field's default is shown as the string
``positional``  a positional argument (the plotters' run directory)

Types: ``int``, ``float``, ``str``; ``bool`` is ``--x`` / ``--no-x``
(``argparse.BooleanOptionalAction``); ``Optional[T]`` is ``T`` with the
default ``None``; ``Literal[...]`` gives the choices.
"""
from __future__ import annotations

import argparse
import dataclasses
import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class Leaf:
    """One option: where it lives in the config and how it is spelled."""
    path: tuple[str, ...]        # attribute path from the root config
    dest: str                    # the flat key (``newton_penalty``): argparse dest and record key
    flag: str                    # ``--newton-penalty``
    type: type                   # the leaf type, ``Optional`` removed
    optional: bool
    choices: tuple | None
    field: dataclasses.Field


def _unwrap(t):
    """``(inner type, optional, choices)`` of an annotation."""
    origin = typing.get_origin(t)
    if origin is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]
        assert len(args) == 1, t
        inner, _, choices = _unwrap(args[0])
        return inner, True, choices
    if origin is typing.Literal:
        choices = typing.get_args(t)
        return type(choices[0]), False, choices
    return t, False, None


def leaves(cls, path=(), prefix=""):
    """The options of ``cls``, nested dataclasses expanded, in declaration order."""
    hints = typing.get_type_hints(cls)
    out = []
    for f in dataclasses.fields(cls):
        t = hints[f.name]
        inner, optional, choices = _unwrap(t)
        if dataclasses.is_dataclass(inner):
            out += leaves(inner, path + (f.name,), f.metadata.get("prefix", ""))
            continue
        flag = f.metadata.get("flag") or "--" + (f"{prefix}-" if prefix else "") + f.name.replace("_", "-")
        out.append(Leaf(path + (f.name,), flag.lstrip("-").replace("-", "_"), flag, inner, optional, choices, f))
    return out


def _default_text(leaf, default):
    if default is None or default == "" or default is dataclasses.MISSING:
        return ""
    if isinstance(default, bool):
        return f" [{str(default).lower()}]"
    if isinstance(default, (tuple, list)):
        return f" [{','.join(str(v) for v in default)}]"
    return f" [{default:g}]" if isinstance(default, float) else f" [{default}]"


def add_arguments(parser, cls):
    """Add the options of ``cls`` to ``parser``, one group per nested dataclass.

    Returns the leaves, for :func:`from_namespace`."""
    hints = typing.get_type_hints(cls)
    top = {leaf.path: leaf for leaf in leaves(cls) if len(leaf.path) == 1}
    out = []
    for f in dataclasses.fields(cls):
        inner, _, _ = _unwrap(hints[f.name])
        if dataclasses.is_dataclass(inner):
            title = (inner.__doc__ or f.name).strip().splitlines()[0].rstrip(".")
            group = parser.add_argument_group(title)
            for leaf in leaves(inner, (f.name,), f.metadata.get("prefix", "")):
                _add(group, leaf)
                out.append(leaf)
        else:
            _add(parser, top[(f.name,)])
            out.append(top[(f.name,)])
    return out


def _add(target, leaf):
    f, meta = leaf.field, leaf.field.metadata
    default = f.default if f.default is not dataclasses.MISSING else (
        f.default_factory() if f.default_factory is not dataclasses.MISSING else dataclasses.MISSING)
    help_ = meta.get("help", "") + _default_text(leaf, default)
    if meta.get("positional"):
        target.add_argument(leaf.dest, help=help_)
    elif leaf.type is bool:
        target.add_argument(leaf.flag, dest=leaf.dest, action=argparse.BooleanOptionalAction,
                            default=default, help=help_)
    else:
        kw = dict(dest=leaf.dest, default=default, help=help_,
                  type=meta.get("parse", leaf.type))
        if leaf.choices:
            kw["choices"] = leaf.choices
        if default is dataclasses.MISSING:
            kw["required"] = True
            del kw["default"]
        target.add_argument(leaf.flag, **kw)


def _build(cls, values, path=(), prefix=""):
    """Instantiate ``cls`` from ``{dest: value}`` (missing keys keep the defaults)."""
    hints = typing.get_type_hints(cls)
    dests = {leaf.path: leaf.dest for leaf in leaves(cls, path, prefix)}
    kw = {}
    for f in dataclasses.fields(cls):
        inner, _, _ = _unwrap(hints[f.name])
        if dataclasses.is_dataclass(inner):
            kw[f.name] = _build(inner, values, path + (f.name,), f.metadata.get("prefix", ""))
        elif dests[path + (f.name,)] in values:
            kw[f.name] = values[dests[path + (f.name,)]]
    return cls(**kw)


def from_namespace(cls, ns):
    return _build(cls, vars(ns))


def flatten(cfg) -> dict:
    """The flat ``{dest: value}`` of a config: the record's ``params``."""
    out = {}
    for leaf in leaves(type(cfg)):
        v = cfg
        for name in leaf.path:
            v = getattr(v, name)
        out[leaf.dest] = list(v) if isinstance(v, tuple) else v
    return out


def unflatten(cls, params: dict):
    """The config of a record's ``params``; keys the class does not know
    (``geometry_path``, ``h_r_sq``, ...: facts of the run) are ignored, and a
    stored list is the tuple it came from."""
    values = {}
    for leaf in leaves(cls):
        if leaf.dest in params:
            v = params[leaf.dest]
            values[leaf.dest] = tuple(v) if isinstance(v, list) else v
    return _build(cls, values)


def parse(cls, argv=None, description=None):
    """Parse ``argv`` (the process's by default) into a ``cls``; a
    ``ValueError`` of its validation is the parser's error."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap, cls)
    ns = ap.parse_args(argv)
    try:
        return from_namespace(cls, ns)
    except ValueError as exc:
        ap.error(str(exc))
