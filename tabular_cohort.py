"""Tabular cohort assembly for clinical, vessel, and radiomics feature tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clinical_features import get_clinical_features
from features import (
    ANNOTATION_COLUMNS,
    FEATURE_BLOCK_DESCRIPTIONS,
    feature_block_for_column,
    normalize_selected_features,
)
from features.clinical import SITE_COLUMNS
from features.graph import (
    add_derived_graph_features,
    extract_graph_json_features,
    extract_local_graph_features,
    load_tumor_graph_payload,
    resolve_tumor_graph_features_path,
)
from features.kinematic import extract_kinematic_json_features
from features.morph import extract_morphometry_features
from features.tumor_size import (
    build_local_tumor_context,
    extract_tumor_size_local_features,
    parse_tumor_radii,
)


def _as_optional_bool(value: Any) -> bool | None:
    """Parse optional bool values from YAML/CLI-like representations."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | np.integer):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"none", "null", ""}:
            return None
        if lower in {"true", "1", "yes", "y"}:
            return True
        if lower in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Unable to parse optional bool from value: {value!r}")


def build_features_from_feature_jsons(morphometry_dir: Path) -> pd.DataFrame:
    """Build a feature table directly from per-study morphometry JSON files."""
    rows: list[dict[str, Any]] = []
    for morphometry_path in sorted(morphometry_dir.glob("*_morphometry.json")):
        case_id = morphometry_path.name.removesuffix("_morphometry.json")
        row: dict[str, Any] = {"case_id": case_id}
        row.update(extract_morphometry_features(morphometry_path))
        rows.append(row)
    return pd.DataFrame(rows)


def build_centerline_features(config: dict[str, Any]) -> pd.DataFrame:
    """Build study-level vascular feature rows from saved centerline outputs."""
    data_paths = config.get("data_paths", {})
    toggles = config.get("feature_toggles", {})

    centerline_root = Path(data_paths.get("centerline_root", ""))
    if not centerline_root.exists():
        raise FileNotFoundError(f"Centerline root not found: {centerline_root}")

    centerline_pattern = str(
        toggles.get("centerline_file_pattern", "{case_id}_skeleton_4d_exam_mask.npy")
    )
    require_centerline_file = bool(toggles.get("require_centerline_file", True))
    include_missing_centerline_rows = bool(
        toggles.get("include_missing_centerline_rows", False)
    )
    include_morphometry = bool(toggles.get("use_morphometry", True))
    include_tumor_graph_json = bool(toggles.get("use_tumor_graph_features_json", True))
    dataset_include = toggles.get("dataset_include")
    dataset_allow: set[str] | None = None
    if dataset_include is not None:
        if isinstance(dataset_include, str):
            dataset_include = [dataset_include]
        dataset_allow = {str(v) for v in dataset_include}
    bilateral_filter = _as_optional_bool(toggles.get("bilateral_filter", None))
    bilateral_lookup: dict[str, bool | None] = {}
    if bilateral_filter is not None:
        try:
            clinical_df = get_clinical_features(config).rename(
                columns={"case_id": "case_id"}
            )
            if "bilateral" in clinical_df.columns:
                for _, rec in (
                    clinical_df[["case_id", "bilateral"]]
                    .dropna(subset=["case_id"])
                    .iterrows()
                ):
                    raw_val = rec["bilateral"]
                    if pd.isna(raw_val):
                        bilateral_lookup[str(rec["case_id"])] = None
                    else:
                        bilateral_lookup[str(rec["case_id"])] = _as_optional_bool(
                            raw_val
                        )
            else:
                logging.warning(
                    "Bilateral prefilter requested but clinical metadata has no "
                    "'bilateral' column; continuing without early bilateral prefilter."
                )
                bilateral_filter = None
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Could not apply early bilateral prefilter: %s. "
                "Continuing without it.",
                exc,
            )
            bilateral_filter = None
    include_tumor_local = bool(toggles.get("use_tumor_local_features", False))
    tumor_mask_root = Path(
        data_paths.get(
            "tumor_mask_root",
            "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert",
        )
    )
    tumor_mask_pattern = str(toggles.get("tumor_mask_file_pattern", "{case_id}.nii.gz"))
    tumor_threshold = float(toggles.get("tumor_mask_threshold", 0.5))
    tumor_radii_voxels = parse_tumor_radii(
        toggles.get("tumor_radius_voxels", [0, 2, 4, 8])
    )

    if include_tumor_local and not tumor_mask_root.exists():
        logging.warning(
            "Tumor local features requested but tumor mask root not found: %s. "
            "Disabling tumor local features.",
            tumor_mask_root,
        )
        include_tumor_local = False

    rows: list[dict[str, Any]] = []
    total_studies = 0
    centerline_exists_count = 0
    tumor_mask_exists_count = 0
    tumor_mask_loaded_count = 0
    tumor_graph_exists_count = 0
    tumor_graph_loaded_count = 0

    study_dirs = [
        (dataset_dir.name, study_dir)
        for dataset_dir in sorted(centerline_root.iterdir())
        if dataset_dir.is_dir()
        for study_dir in sorted(dataset_dir.iterdir())
        if study_dir.is_dir()
    ]

    for idx, (dataset_name, study_dir) in enumerate(study_dirs, start=1):
        if dataset_allow is not None and str(dataset_name) not in dataset_allow:
            continue

        case_id = study_dir.name

        if bilateral_filter is not None:
            case_bilateral = bilateral_lookup.get(str(case_id))
            if case_bilateral is None or case_bilateral != bilateral_filter:
                continue

        total_studies += 1

        centerline_path = study_dir / centerline_pattern.format(case_id=case_id)
        has_centerline_file = centerline_path.exists()
        if has_centerline_file:
            centerline_exists_count += 1

        if (
            require_centerline_file
            and not has_centerline_file
            and not include_missing_centerline_rows
        ):
            continue

        row: dict[str, Any] = {
            "case_id": case_id,
            "dataset": dataset_name,
            "has_centerline_file": bool(has_centerline_file),
        }

        summary_path = study_dir / "run_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Failed run_summary parse for %s: %s", summary_path, exc
                )
                summary = {}

        feature_stats = (
            summary.get("feature_stats", {}) if isinstance(summary, dict) else {}
        )
        if not isinstance(feature_stats, dict):
            feature_stats = {}
        row.update(
            {
                "graph_skeleton_voxels": summary.get("skeleton_voxels"),
                "graph_support_voxels": summary.get("support_voxels"),
                "graph_nodes": feature_stats.get("graph_nodes"),
                "graph_edges": feature_stats.get("graph_edges"),
                "graph_segment_count": feature_stats.get("segment_count"),
                "graph_component_count": feature_stats.get("component_count"),
            }
        )

        if include_morphometry:
            morphometry_path = study_dir / f"{case_id}_morphometry.json"
            if morphometry_path.exists():
                try:
                    row.update(extract_morphometry_features(morphometry_path))
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "Failed morphometry parse for %s: %s",
                        morphometry_path,
                        exc,
                    )

        if include_tumor_local and has_centerline_file:
            local_tumor_context = build_local_tumor_context(
                case_id=case_id,
                dataset_name=dataset_name,
                centerline_path=centerline_path,
                tumor_mask_root=tumor_mask_root,
                tumor_mask_pattern=tumor_mask_pattern,
                tumor_threshold=tumor_threshold,
                tumor_radii_voxels=tumor_radii_voxels,
            )
            row.update(
                extract_tumor_size_local_features(
                    local_tumor_context,
                    tumor_radii_voxels=tumor_radii_voxels,
                )
            )
            row.update(
                extract_local_graph_features(
                    local_tumor_context,
                    tumor_radii_voxels=tumor_radii_voxels,
                )
            )
            if row.get("tumor_mask_exists") == 1.0:
                tumor_mask_exists_count += 1
            if row.get("tumor_mask_loaded") == 1.0:
                tumor_mask_loaded_count += 1

        if include_tumor_graph_json:
            tumor_graph_path = resolve_tumor_graph_features_path(
                case_id=case_id,
                study_dir=study_dir,
                summary=summary,
            )
            row["tumor_graph_features_exists"] = (
                1.0 if tumor_graph_path is not None else 0.0
            )
            row["tumor_graph_features_loaded"] = 0.0

            status_from_summary = summary.get("tumor_graph_features_status")
            if status_from_summary is None:
                status_from_summary = feature_stats.get("tumor_graph_features_status")
            if status_from_summary is not None:
                row["tumor_graph_status_ok_summary"] = (
                    1.0 if str(status_from_summary).strip().lower() == "ok" else 0.0
                )

            if tumor_graph_path is not None:
                tumor_graph_exists_count += 1
                try:
                    tumor_graph_payload = load_tumor_graph_payload(tumor_graph_path)
                    row.update(extract_graph_json_features(tumor_graph_payload))
                    row.update(extract_kinematic_json_features(tumor_graph_payload))
                    if row.get("tumor_graph_features_loaded") == 1.0:
                        tumor_graph_loaded_count += 1
                except Exception as exc:  # noqa: BLE001
                    logging.warning(
                        "Failed tumor graph feature parse for %s: %s",
                        tumor_graph_path,
                        exc,
                    )

        add_derived_graph_features(row)
        rows.append(row)

        if idx % 300 == 0:
            logging.info("Parsed %d/%d centerline studies", idx, len(study_dirs))

    if dataset_allow is not None:
        logging.info(
            "Centerline build applied dataset prefilter: %s",
            sorted(dataset_allow),
        )
    if bilateral_filter is not None:
        logging.info(
            "Centerline build applied bilateral prefilter: %s", bilateral_filter
        )
    logging.info(
        "Centerline file coverage: %d/%d studies have file (%s)",
        centerline_exists_count,
        total_studies,
        centerline_pattern,
    )
    if include_tumor_local:
        logging.info(
            "Tumor mask coverage among parsed studies: exists=%d, loaded=%d, radii=%s",
            tumor_mask_exists_count,
            tumor_mask_loaded_count,
            tumor_radii_voxels,
        )
    if include_tumor_graph_json:
        logging.info(
            "Tumor-graph JSON coverage among parsed studies: exists=%d, loaded=%d",
            tumor_graph_exists_count,
            tumor_graph_loaded_count,
        )
    centerline_df = pd.DataFrame(rows)
    logging.info("Centerline feature table shape: %s", centerline_df.shape)
    return centerline_df


def build_modular_features(config: dict[str, Any]) -> pd.DataFrame:
    """Build and merge the requested feature blocks into one case-level table."""
    toggles = config.get("feature_toggles", {})

    use_vascular = bool(toggles.get("use_vascular", False))
    use_clinical = bool(toggles.get("use_clinical", False))
    include_site_features = bool(toggles.get("include_site_features", True))
    merge_how = str(toggles.get("merge_how", "inner"))
    dataset_include = toggles.get("dataset_include")
    bilateral_filter = _as_optional_bool(toggles.get("bilateral_filter", None))

    if use_vascular:
        logging.info("Loading vascular centerline features...")
        merged_df = build_centerline_features(config)
    elif use_clinical:
        clinical_df = get_clinical_features(config).rename(
            columns={"case_id": "case_id"}
        )
        merged_df = clinical_df.copy()
    else:
        raise ValueError(
            "No feature blocks enabled. Set at least one of: "
            "feature_toggles.use_vascular or feature_toggles.use_clinical"
        )

    try:
        all_clinical = get_clinical_features(config).rename(
            columns={"case_id": "case_id"}
        )
    except Exception as exc:  # noqa: BLE001
        all_clinical = pd.DataFrame(columns=["case_id"])
        logging.warning("Could not load annotation metadata: %s", exc)

    annotation_cols = [
        c
        for c in ["case_id", "dataset", "site", "bilateral", "tumor_subtype"]
        if c in all_clinical.columns
    ]
    if annotation_cols:
        annotations = all_clinical[annotation_cols].drop_duplicates(subset=["case_id"])
        missing_ann_cols = [
            c for c in annotation_cols if c != "case_id" and c not in merged_df.columns
        ]
        if missing_ann_cols:
            merged_df = merged_df.merge(
                annotations[["case_id"] + missing_ann_cols],
                on="case_id",
                how="left",
            )

    if use_clinical and not all_clinical.empty:
        clinical_features = all_clinical.copy()
        if not include_site_features:
            clinical_features = clinical_features.drop(
                columns=SITE_COLUMNS, errors="ignore"
            )

        overlap_cols = [
            c
            for c in clinical_features.columns
            if c != "case_id" and c in merged_df.columns
        ]
        if overlap_cols:
            clinical_features = clinical_features.drop(
                columns=overlap_cols, errors="ignore"
            )
        merged_df = merged_df.merge(clinical_features, on="case_id", how=merge_how)

    if dataset_include is not None and "dataset" in merged_df.columns:
        if isinstance(dataset_include, str):
            dataset_include = [dataset_include]
        dataset_include = {str(v) for v in dataset_include}
        before = len(merged_df)
        merged_df = merged_df[
            merged_df["dataset"].astype(str).isin(dataset_include)
        ].copy()
        logging.info(
            "Applied dataset filter %s: %d -> %d rows",
            sorted(dataset_include),
            before,
            len(merged_df),
        )

    if bilateral_filter is not None and "bilateral" in merged_df.columns:
        before = len(merged_df)
        bilateral_series = merged_df["bilateral"].astype("boolean")
        merged_df = merged_df[bilateral_series == bilateral_filter].copy()
        logging.info(
            "Applied bilateral filter %s: %d -> %d rows",
            bilateral_filter,
            before,
            len(merged_df),
        )

    merged_df = merged_df.dropna(subset=["case_id"]).drop_duplicates(subset=["case_id"])
    logging.info("Merged feature table shape: %s", merged_df.shape)
    return merged_df


def load_labels(path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    """Load labels from CSV or JSON and normalize to integer {0, 1}."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df_labels = pd.read_csv(path)
    else:
        if path.is_dir():
            rows = []
            for json_path in sorted(path.glob("*.json")):
                try:
                    rows.append(json.loads(json_path.read_text()))
                except Exception:  # noqa: BLE001
                    logging.debug("Skipping unreadable label JSON: %s", json_path)
                    continue
            df_labels = pd.DataFrame(rows)
        else:
            obj = json.loads(path.read_text())
            df_labels = pd.DataFrame(obj)

    if id_col not in df_labels.columns and "case_id" in df_labels.columns:
        df_labels = df_labels.rename(columns={"case_id": id_col})

    df_labels = df_labels.dropna(subset=[label_col])
    mapping = {"true": 1, "false": 0, "yes": 1, "no": 0}

    def clean_val(value: Any) -> Any:
        s = str(value).strip().lower()
        return mapping.get(s, value)

    df_labels[label_col] = pd.to_numeric(
        df_labels[label_col].map(clean_val),
        errors="coerce",
    )
    df_labels = df_labels.dropna(subset=[label_col])
    df_labels[label_col] = df_labels[label_col].astype(int)

    return df_labels[[id_col, label_col]].rename(columns={id_col: "case_id"})


def select_features(
    df: pd.DataFrame,
    *,
    selected_blocks: Any,
    label_col: str,
) -> pd.DataFrame:
    """Filter a labeled feature table down to the requested canonical blocks."""
    normalized_blocks = normalize_selected_features(selected_blocks)
    if not normalized_blocks:
        return df

    keep_columns: list[str] = [
        column
        for column in df.columns
        if column in ANNOTATION_COLUMNS or column == label_col
    ]
    selected_set = set(normalized_blocks)
    selected_feature_columns = [
        column
        for column in df.columns
        if column not in keep_columns
        and feature_block_for_column(column) in selected_set
    ]

    if not selected_feature_columns:
        raise ValueError(
            "selected_features filtered out all modeling features. "
            f"Requested blocks: {normalized_blocks}"
        )

    block_counts = {
        block: sum(
            1
            for column in selected_feature_columns
            if feature_block_for_column(column) == block
        )
        for block in normalized_blocks
    }
    block_descriptions = {
        block: FEATURE_BLOCK_DESCRIPTIONS.get(block, block)
        for block in normalized_blocks
    }
    logging.info(
        "Selected feature blocks %s -> %d columns (%s); descriptions=%s",
        normalized_blocks,
        len(selected_feature_columns),
        block_counts,
        block_descriptions,
    )
    return df[keep_columns + selected_feature_columns].copy()


def prepare_data(config: dict[str, Any], outdir: Path) -> pd.DataFrame:
    """Load feature blocks, merge labels, and write the final labeled table."""
    feats_df = build_modular_features(config)
    feats_df.to_csv(outdir / "features_raw.csv", index=False)

    labels_path = config["data_paths"]["labels_csv"]
    label_col = config["data_paths"]["label_column"]
    id_col = config["data_paths"].get("id_column", "case_id")

    labels_df = load_labels(labels_path, id_col, label_col)
    merged_df = feats_df.merge(labels_df, on="case_id", how="inner")
    merged_df = select_features(
        merged_df,
        selected_blocks=config.get("feature_toggles", {}).get(
            "selected_features", None
        ),
        label_col=label_col,
    )
    merged_df.to_csv(outdir / "features_engineered_labeled.csv", index=False)

    logging.info("Final labeled data shape: %s", merged_df.shape)
    return merged_df
