"""Guard the call contract between preprocessing.complement and SegVessel.

``complement_exam`` calls ``SegVessel`` with keyword arguments. Nothing else in the
test suite exercises that call: the QC tooling in ``analysis/`` invokes ``SegVessel``
directly, and the pipeline stage that would hit it needs a full preprocessed exam on
disk. A keyword renamed on one side and not the other therefore passes every test,
passes ruff, and only fails when a real ``complement_case`` run reaches the call --
which is exactly what happened once (``roi=`` vs ``normalisation_roi=``).

Checking the signatures agree is cheap and catches the whole class without needing
imaging data.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from segmentation.matlab_vessel_segmentation import SegVessel

_COMPLEMENT_SOURCE = (
    Path(__file__).resolve().parents[1] / "preprocessing" / "complement.py"
)


def _segvessel_call_keywords() -> set[str]:
    """Keyword names complement.py actually passes to SegVessel(...)."""
    tree = ast.parse(_COMPLEMENT_SOURCE.read_text())
    keywords: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "SegVessel":
            continue
        keywords.update(kw.arg for kw in node.keywords if kw.arg is not None)
    return keywords


def test_complement_calls_segvessel_with_real_keywords() -> None:
    """Every keyword complement.py passes must exist on SegVessel's signature."""
    passed = _segvessel_call_keywords()
    assert passed, "expected complement.py to call SegVessel with keyword arguments"

    accepted = set(inspect.signature(SegVessel).parameters)
    unknown = passed - accepted
    assert not unknown, (
        f"complement.py passes keyword(s) SegVessel does not accept: {sorted(unknown)}; "
        f"SegVessel accepts {sorted(accepted)}"
    )


def test_segvessel_still_takes_a_normalisation_roi() -> None:
    """The ROI parameter is load-bearing: without it the mask edge sets the maxima.

    Pinned by name because it is the fix for a measured failure (32-66% of true
    skeleton voxels scoring exactly zero vesselness), not an incidental argument.
    """
    assert "normalisation_roi" in inspect.signature(SegVessel).parameters
