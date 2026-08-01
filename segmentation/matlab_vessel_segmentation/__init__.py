"""Python port of the MATLAB SegVessel/Jerman vessel-segmentation pipeline.

Method provenance
-----------------
This package extends the lab MATLAB vessel analysis pipeline maintained by
Zhen Ren under ``KineticAnalysisFunctions`` (SegVessel, Vessel_morph,
segmentVessels3D, and related helpers). That MATLAB work builds on the
quantitative vessel morphology / vesselness approach described in:

    Wu C, Pineda F, Hormuth DA 2nd, Karczmar GS, Yankeelov TE.
    Quantitative analysis of vascular properties derived from ultrafast
    DCE-MRI to discriminate malignant and benign breast tumors.
    Magn Reson Med. 2019 Mar;81(3):2147-2160.
    doi:10.1002/mrm.27529; PMID:30368906.
    https://pubmed.ncbi.nlm.nih.gov/30368906/

Ported into vanguard from the standalone validation workspace
``matlab-conv-2/vessel_pipeline`` (workspace saritbose), where it was validated
against the lab's production merged HR+UFAST skeleton (``preprocessing.merge``)
across 6 UChicago exams. Used by ``preprocessing.complement`` to add real,
complementary vessel voxels on top of that merged skeleton -- see that module
for the production integration and quality-gated combine logic.
"""

from .segment_vessels import SegVessel, SegVessel_U, segment_vessels_3d

__all__ = ["SegVessel", "SegVessel_U", "segment_vessels_3d"]
