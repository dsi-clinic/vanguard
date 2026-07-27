"""Python port of the MATLAB SegVessel/Jerman vessel-segmentation pipeline.

Ported from ``matlab-conv-2/vessel_pipeline`` (workspace saritbose), where it was
validated against the lab's production merged HR+UFAST skeleton (``preprocessing.merge``)
across 6 UChicago exams. Used by ``preprocessing.complement`` to add real, complementary
vessel voxels on top of that merged skeleton -- see that module for the production
integration and quality-gated combine logic.
"""

from .segment_vessels import SegVessel, SegVessel_U, segment_vessels_3d

__all__ = ["SegVessel", "SegVessel_U", "segment_vessels_3d"]
