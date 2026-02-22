"""Sanity checks for morphometry feature values (biological plausibility).

Validates:
- tortuosity >= 1
- bifurcation angles in [0, 180] degrees
- radii > 0
- lengths > 0
- curvature >= 0 (radians)
"""

from __future__ import annotations

import json
from pathlib import Path

# Bifurcation angles in degrees
MAX_ANGLE_DEGREES = 180


def _check_tortuosity(value: float, case_id: str, context: str) -> list[dict]:
    violations = []
    if value < 1.0:
        violations.append(
            {
                "case_id": case_id,
                "feature": "tortuosity",
                "value": value,
                "violation_type": "tortuosity_lt_1",
                "rule": "tortuosity >= 1",
                "context": context,
            }
        )
    return violations


def _check_angle(value: float, case_id: str, feature: str, context: str) -> list[dict]:
    violations = []
    if not (0 <= value <= MAX_ANGLE_DEGREES):
        violations.append(
            {
                "case_id": case_id,
                "feature": feature,
                "value": value,
                "violation_type": "angle_out_of_range",
                "rule": "angle in [0, 180]",
                "context": context,
            }
        )
    return violations


def _check_positive(
    value: float, case_id: str, feature: str, rule_name: str, context: str
) -> list[dict]:
    violations = []
    if value <= 0:
        violations.append(
            {
                "case_id": case_id,
                "feature": feature,
                "value": value,
                "violation_type": rule_name,
                "rule": f"{feature} > 0",
                "context": context,
            }
        )
    return violations


def _check_curvature(value: float, case_id: str, context: str) -> list[dict]:
    violations = []
    if value < 0:
        violations.append(
            {
                "case_id": case_id,
                "feature": "curvature",
                "value": value,
                "violation_type": "curvature_negative",
                "rule": "curvature >= 0",
                "context": context,
            }
        )
    return violations


def check_morphometry_json(json_path: Path) -> list[dict]:
    """Check one morphometry JSON file for sanity violations.

    Returns list of violation dicts with case_id, feature, value, violation_type, rule, context.
    """
    case_id = "_".join(json_path.stem.split("_")[:2])
    violations: list[dict] = []

    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return [
            {
                "case_id": case_id,
                "feature": "parse_error",
                "value": None,
                "violation_type": "parse_error",
                "rule": "valid JSON",
                "context": str(json_path),
            }
        ]

    if not isinstance(data, dict):
        return violations

    for comp_key, group in data.items():
        if not isinstance(group, dict):
            continue
        for vessel_name, items in group.items():
            if not isinstance(items, list):
                continue
            ctx = f"{comp_key}/{vessel_name}"

            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                item_ctx = f"{ctx}[{idx}]"

                # Segment metrics
                if "tortuosity" in item:
                    v = item["tortuosity"]
                    if isinstance(v, int | float):
                        violations.extend(
                            _check_tortuosity(float(v), case_id, item_ctx)
                        )

                if "length" in item:
                    v = item["length"]
                    if isinstance(v, int | float):
                        violations.extend(
                            _check_positive(
                                float(v),
                                case_id,
                                "length",
                                "length_non_positive",
                                item_ctx,
                            )
                        )

                if "radius" in item and isinstance(item["radius"], dict):
                    for k, rv in item["radius"].items():
                        if isinstance(rv, int | float):
                            violations.extend(
                                _check_positive(
                                    float(rv),
                                    case_id,
                                    f"radius.{k}",
                                    "radius_non_positive",
                                    item_ctx,
                                )
                            )

                if "curvature" in item and isinstance(item["curvature"], dict):
                    for k, cv in item["curvature"].items():
                        if isinstance(cv, int | float):
                            violations.extend(
                                _check_curvature(
                                    float(cv), case_id, f"{item_ctx}/curvature.{k}"
                                )
                            )

                # Bifurcation angles
                if "angles" in item and isinstance(item["angles"], dict):
                    for angle_name, av in item["angles"].items():
                        if isinstance(av, int | float):
                            violations.extend(
                                _check_angle(
                                    float(av), case_id, f"angles.{angle_name}", item_ctx
                                )
                            )

    return violations


def check_morphometry_dir(morphometry_dir: Path) -> list[dict]:
    """Check all morphometry JSONs in a directory. Returns flat list of violations."""
    all_violations: list[dict] = []
    for p in sorted(morphometry_dir.glob("*.json")):
        all_violations.extend(check_morphometry_json(p))
    return all_violations
