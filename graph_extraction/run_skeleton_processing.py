"""Single entrypoint for tc4d skeleton extraction and morphometry generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DEFAULT_SEGMENTATION_DIR = Path("/net/projects2/vanguard/vessel_segmentations")
DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR = Path(
    "/net/projects2/vanguard/Duke-Breast-Cancer-MRI-Supplement-v3"
)
DEFAULT_TUMOR_MASK_DIR = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run tc4d skeleton extraction and morphometry in one pipeline."
    )
    parser.add_argument("--study-id", type=str, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="Directory containing `<study-id>/images/*.npz` segmentation files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory."
    )
    parser.add_argument("--force-skeleton", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument(
        "--save-exam-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write exam skeleton/support masks as output .npy files.",
    )
    parser.add_argument(
        "--save-center-manifold-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the full 4D manifold mask as an output .npy.",
    )
    parser.add_argument(
        "--render-mip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render a vessel-coverage MIP PNG for the extracted 4D exam skeleton. "
            "If matching radiologist annotations are found for DUKE/Breast_MRI ids, "
            "they are included for comparison."
        ),
    )
    parser.add_argument(
        "--mip-dpi",
        type=int,
        default=180,
        help="DPI for coverage MIP output.",
    )
    parser.add_argument(
        "--radiologist-annotations-dir",
        type=Path,
        default=DEFAULT_RADIOLOGIST_ANNOTATIONS_DIR,
        help=(
            "Base path for Duke supplemental annotations. May be either the dataset "
            "root or direct `Segmentation_Masks_NRRD`."
        ),
    )
    parser.add_argument(
        "--tumor-mask-dir",
        type=Path,
        default=DEFAULT_TUMOR_MASK_DIR,
        help=(
            "Directory for optional tumor masks used in MIP overlays "
            "(expected `<study-id>.nii.gz`)."
        ),
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    from graph_extraction.processing import process_4d_study

    start = time.perf_counter()
    result = process_4d_study(
        input_dir=args.input_dir,
        study_id=args.study_id,
        output_dir=args.output_dir,
        force_skeleton=args.force_skeleton,
        force_features=args.force_features,
        save_exam_masks=args.save_exam_masks,
        save_center_manifold_mask=args.save_center_manifold_mask,
        render_mip=args.render_mip,
        mip_dpi=args.mip_dpi,
        radiologist_annotations_dir=args.radiologist_annotations_dir,
        tumor_mask_dir=args.tumor_mask_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "processing_summary.json"
    result["elapsed_seconds"] = float(time.perf_counter() - start)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("[done] Pipeline finished")
    print(f"  mode: {result['mode']}")
    if result.get("skeleton_path") is not None:
        print(f"  skeleton: {result['skeleton_path']}")
    print(f"  morphometry: {result['morphometry_path']}")
    if result.get("support_path") is not None:
        print(f"  support: {result['support_path']}")
    if result.get("manifold_path") is not None:
        print(f"  manifold_4d: {result['manifold_path']}")
    if result.get("coverage_mip_path") is not None:
        print(f"  coverage_mip: {result['coverage_mip_path']}")
    print(f"  summary: {summary_path}")
    print(f"[done] Total time: {result['elapsed_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
