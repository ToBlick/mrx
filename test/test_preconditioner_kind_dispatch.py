"""Every accepted preconditioner kind must be handled where it is accepted.

`mrx/operators.py` validates preconditioner kinds against per-function
accept-lists (``valid_kinds`` / ``valid_outer_kinds``) and then dispatches on
``spec.kind == '...'``. Nothing tied the two together, and they drifted: the
retired ``'tensor'`` stack was deleted on 2026-08-25 but stayed in FOUR
accept-lists, so a caller asking for it got past validation and fell through
every branch to whatever `raise` happened to end the function -- in one case a
bare ``AssertionError("unreachable")``, in two others a message blaming the
unrelated richardson/chebyshev removal.

The drift matters most during a rename. If a kind is accepted but dead, and
some LIVE kind is later renamed into that spelling, the stale accept-list entry
and the new dispatch branch meet, and a previously-rejected kind starts
silently working -- turning a loud ValueError into a wrong number. That is why
this is a test and not a one-off grep.

Functions carrying an accept-list come in two shapes, and only one of them can
have this defect:

* DISPATCHERS end in a fall-through ``raise``. An accepted-but-unbranched kind
  reaches that raise, whose message is by construction about some OTHER kind.
  These are the ones to check.
* FORWARDERS validate and hand the kind on to a dispatcher, or end in a default
  ``return`` that handles the last kind implicitly. An unbranched kind there is
  normal.

The split is spelled out below rather than sniffed, because every heuristic for
it that I tried had false positives on this file -- and a check that cries wolf
gets an exemption bolted on rather than a bug fixed. `test_..._is_classified`
keeps the list honest: a new accept-list function fails until someone decides
which shape it is.

Pure AST: no geometry, no JAX, no GPU.
"""
from __future__ import annotations

import ast
import pathlib

OPERATORS = pathlib.Path(__file__).resolve().parents[1] / "mrx" / "operators.py"
ACCEPT_LIST_NAMES = ("valid_kinds", "valid_outer_kinds")

# Ends in a fall-through raise: every accepted kind needs its own branch.
DISPATCHERS = {
    "_build_operator_preconditioner_apply",
    "_build_diffusion_preconditioner_apply",
    "_build_scalar_hodge_preconditioner_apply",
}

# Validates, then forwards or falls to a default return.
FORWARDERS = {
    # returns the coerced spec; the kind is dispatched by its consumer
    "_coerce_saddle_preconditioner_spec",
}

# Accepted by a dispatcher without an `== 'kind'` branch of its own, for a
# reason that is not a defect.
STRUCTURAL_KINDS = {
    # resolved to a concrete kind before dispatch
    "auto",
}

# Every kind name production code may legitimately construct. raw_kron was
# deleted 2026-08-25; 'tensor' before it.
LIVE_KINDS = {"none", "jacobi", "metric_lumping", "auto"}


def _accept_list_functions(tree: ast.Module) -> dict:
    """{function name: (node, accepted kinds)} for every accept-list carrier."""
    found = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        kinds = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if not names & set(ACCEPT_LIST_NAMES):
                continue
            if isinstance(node.value, (ast.Tuple, ast.List)):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        kinds.add(elt.value)
        if kinds:
            found[fn.name] = (fn, kinds)
    return found


def _dispatched_kinds(fn: ast.FunctionDef) -> set:
    """String constants this function compares against, via == or `in`."""
    kinds = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare):
            continue
        for op, cmp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(cmp, ast.Constant):
                if isinstance(cmp.value, str):
                    kinds.add(cmp.value)
            elif isinstance(op, (ast.In, ast.NotIn)) and isinstance(
                    cmp, (ast.Tuple, ast.List, ast.Set)):
                for elt in cmp.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        kinds.add(elt.value)
    return kinds


def test_every_accept_list_function_is_classified():
    """A new accept-list must be declared a dispatcher or a forwarder."""
    tree = ast.parse(OPERATORS.read_text())
    found = set(_accept_list_functions(tree))
    unclassified = found - DISPATCHERS - FORWARDERS
    assert not unclassified, (
        "these functions carry a preconditioner accept-list but are in neither "
        f"DISPATCHERS nor FORWARDERS: {sorted(unclassified)}. Decide which "
        "shape each one is -- a dispatcher ending in a fall-through raise must "
        "branch every kind it accepts.")
    stale = (DISPATCHERS | FORWARDERS) - found
    assert not stale, (
        f"classified functions that no longer carry an accept-list: "
        f"{sorted(stale)}")


def test_dispatchers_branch_every_kind_they_accept():
    tree = ast.parse(OPERATORS.read_text())
    offenders = []
    for name, (fn, accepted) in _accept_list_functions(tree).items():
        if name not in DISPATCHERS:
            continue
        for kind in sorted(accepted - _dispatched_kinds(fn) - STRUCTURAL_KINDS):
            offenders.append(f"{name} (line {fn.lineno}) accepts {kind!r}")
    assert not offenders, (
        "accept-list admits a kind with no dispatch branch; such a kind gets "
        "past validation and falls through to the raise that ends the "
        "function, whose message is about a different kind entirely:\n  "
        + "\n  ".join(offenders))


def test_the_retired_tensor_kind_is_accepted_nowhere():
    """The specific drift that motivated this file, pinned on its own.

    'tensor' named the surgery-plus-Schur stack deleted on 2026-08-25. It must
    not reappear in ANY accept-list -- dispatcher or forwarder -- because the
    production atom is a candidate for being renamed, and a recycled name is
    exactly what makes a silent revival possible.
    """
    tree = ast.parse(OPERATORS.read_text())
    offenders = [
        f"{name} (line {fn.lineno})"
        for name, (fn, kinds) in _accept_list_functions(tree).items()
        if "tensor" in kinds
    ]
    assert not offenders, (
        "the retired 'tensor' kind is accepted by: " + ", ".join(offenders))


def _constructed_kinds(path: pathlib.Path) -> dict:
    """{kind string: [lines]} for every `kind='...'` and `kind: str = '...'`."""
    tree = ast.parse(path.read_text())
    found = {}

    def note(value, lineno):
        found.setdefault(value, []).append(lineno)

    for node in ast.walk(tree):
        # MassPreconditionerSpec(kind='...') -- and ONLY spec constructors.
        # `kind=` is a common keyword elsewhere (np.argsort(kind='stable')),
        # so matching every call would cry wolf and get the check disabled.
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "id", None) or getattr(
                node.func, "attr", None) or ""
            if not fname.endswith("PreconditionerSpec"):
                continue
            for kw in node.keywords:
                if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        note(kw.value.value, node.lineno)
        # dataclass field default:  kind: str = '...'
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            if (isinstance(t, ast.Name) and t.id == "kind"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                note(node.value.value, node.lineno)
    return found


def test_no_production_code_constructs_a_kind_nothing_accepts():
    """The OTHER direction, which this file did not check until it bit.

    The tests above verify accept-list -> dispatch. Nothing verified
    spec -> accept-list, and during the 2026-08-25 metric_lumping rename that
    gap let the tree reach a state where `MassPreconditionerSpec.kind` defaulted
    to 'metric_lumping' while every accept-list still said 'block_jacobi'. Every
    test here passed. A one-directional invariant is half an invariant.
    """
    offenders = []
    for name in ("operators.py", "preconditioners.py", "nullspace.py",
                 "derham_sequence.py"):
        path = OPERATORS.parent / name
        for kind, lines in _constructed_kinds(path).items():
            if kind not in LIVE_KINDS:
                offenders.append(f"{name}:{lines[0]} constructs kind={kind!r}")
    assert not offenders, (
        "production code constructs a preconditioner kind that is not a live "
        f"kind {sorted(LIVE_KINDS)}:\n  " + "\n  ".join(offenders))
