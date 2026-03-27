"""Shared constants for graph-extraction processing."""

from __future__ import annotations

from pathlib import Path

from graph_extraction.core4d import NDIM_3D, NDIM_4D

DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR = Path(
    "/net/projects2/vanguard/Duke-Breast-Cancer-MRI-Supplement-v3"
)
DEFAULT_TUMOR_MASK_DIR = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
)
PROCESSING_VIZ_FLIP_SPEC = "z"
VIZ_FLIP_SPECS = ("none", "z", "y", "x", "zy", "zx", "yx", "zyx")
SEG_NRRD_MAX_LAYERS = 32
TUMOR_NEAR_MM = 5.0
TUMOR_BOUNDARY_NEAR_MM = 2.0
BIFURCATION_MIN_DEGREE = 3
DEGREE_FOUR_PLUS = 4
MIN_PATH_POINTS = 2
MIN_CURVATURE_POINTS = 3
MIN_LINEAR_FIT_POINTS = 2
MIN_KINETIC_TIMEPOINTS = 2
KINETIC_SIGNAL_EPS = 1e-6
EARLY_CYCLE_FRACTION_MAJORITY = 0.5
EARLY_CYCLE_FRACTION_ALL = 0.999
TUMOR_SHELL_SPECS: tuple[tuple[str, float, float], ...] = (
    ("inside_tumor", float("-inf"), 0.0),
    ("shell_0_2mm", 0.0, 2.0),
    ("shell_2_5mm", 2.0, 5.0),
    ("shell_5_10mm", 5.0, 10.0),
    ("shell_10_20mm", 10.0, 20.0),
    ("shell_gt20mm", 20.0, float("inf")),
)
