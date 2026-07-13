"""Unit tests for the cohort-adapter scaffold (Step 1).

These exercise the adapters in isolation — construction, documented base
behavior, class-attribute defaults, the factory, and split-policy resolution —
without touching any real data on disk.

Deliberately not tested here: the exact set of methods each subclass
overrides. That's an implementation-shape detail that will keep shifting as
the design evolves; pinning it in a test would make routine changes fail for no
functional reason. What's tested instead is the *behavior* those overrides are
responsible for (identity parsing, preprocessing reorientation, provided folds,
split-policy defaults, etc.).

A few tests that need a small manifest use a synthetic CSV fixture; none touch
real data on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cohorts import (
    DatasetAdapter,
    MamaMiaDataset,
    UChicagoDataset,
    build_adapter_from_config,
    resolve_folds,
    resolve_split_policy,
)
from config import ConfigNode


def _dataset_config(dataset: dict) -> ConfigNode:
    """Wrap a bare ``dataset`` block into a config node for factory tests."""
    return ConfigNode._wrap({"dataset": dataset})


# -- construction --


@pytest.mark.parametrize("cohort", ["duke", "ispy1", "ispy2", "nact"])
def test_mamamia_constructs_for_each_cohort(cohort: str) -> None:
    """Each valid cohort constructs and stores its root and cohort."""
    adapter = MamaMiaDataset(cohort=cohort, root=Path("/data/mamamia"))
    assert adapter.cohort == cohort
    assert adapter.root == Path("/data/mamamia")


def test_mamamia_rejects_unknown_cohort() -> None:
    """An unknown cohort raises ValueError."""
    with pytest.raises(ValueError, match="Unknown MAMA-MIA cohort"):
        MamaMiaDataset(cohort="tcga", root=Path("/data/mamamia"))


def test_mamamia_cohort_is_case_insensitive() -> None:
    """The cohort argument is normalized to lower-case."""
    assert MamaMiaDataset(cohort="ISPY2", root=Path("/x")).cohort == "ispy2"


def test_uchicago_constructs_with_default_manifest() -> None:
    """UChicago derives its manifest path from the root by default."""
    adapter = UChicagoDataset(root=Path("/data/uchicago"))
    assert adapter.root == Path("/data/uchicago")
    assert adapter.manifest_csv == Path(
        "/data/uchicago/dce2d_internal_ultrafast_manifest.csv"
    )


# -- class-attribute defaults --


def test_split_policy_defaults() -> None:
    """MAMA-MIA computes its splits; UChicago ships them."""
    assert DatasetAdapter.default_split_policy == "compute"
    assert MamaMiaDataset.default_split_policy == "compute"
    assert UChicagoDataset.default_split_policy == "provided"


def test_report_by_defaults() -> None:
    """Only UChicago breaks results down by sub-source."""
    assert MamaMiaDataset.report_by is None
    assert UChicagoDataset.report_by == "dataset"


# -- pure base behavior (no data needed) --


def test_case_dataset_name_prefix_parse() -> None:
    """case_dataset_name uses the case-id prefix (batch_segmentation.py:243)."""
    adapter = MamaMiaDataset(cohort="ispy2", root=Path("/x"))
    assert adapter.case_dataset_name("ISPY2_045") == "ISPY2"


def test_naming_rules_match_current_patterns() -> None:
    """Naming methods match the current config filename patterns."""
    adapter = MamaMiaDataset(cohort="duke", root=Path("/x"))
    assert adapter.tumor_mask_filename("DUKE_1") == "DUKE_1.nii.gz"
    assert adapter.centerline_filename("DUKE_1") == "DUKE_1_skeleton_4d_exam_mask.npy"
    assert adapter.morphometry_filename("DUKE_1") == "DUKE_1_morphometry.json"


def test_preprocess_is_the_mamamia_orientation_transform() -> None:
    """Base preprocess reorients exactly like batch_segmentation.py:85."""
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    adapter = MamaMiaDataset(cohort="duke", root=Path("/x"))
    expected = np.swapaxes(np.swapaxes(volume, 0, 2), 0, 1)[::-1]
    assert np.array_equal(adapter.preprocess(volume), expected)


def test_resample_is_noop_when_no_target_spacing() -> None:
    """With no target spacing (MAMA-MIA), resample returns the volume unchanged."""
    volume = np.zeros((2, 2, 2))
    adapter = MamaMiaDataset(cohort="duke", root=Path("/x"))
    assert np.array_equal(adapter.resample(volume, (1.0, 1.0, 1.0)), volume)


def test_uchicago_preprocess_flips_x_and_z_then_applies_base_transform() -> None:
    """UChicago is RAS where MAMA-MIA is LAI, so x/z are flipped before the base transform.

    SimpleITK returns (z, y, x)-ordered arrays, so undoing that sign difference
    means flipping array axes 0 and 2 and then deferring to the base MAMA-MIA
    reorientation. An identity pass-through (the original assumption) put the
    thin slice axis where the model expects an in-plane axis.
    """
    adapter = UChicagoDataset(root=Path("/data/uchicago"))
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    flipped = volume[::-1, :, ::-1]
    expected = np.swapaxes(np.swapaxes(flipped, 0, 2), 0, 1)[::-1]
    assert np.array_equal(adapter.preprocess(volume), expected)
    assert not np.array_equal(adapter.preprocess(volume), volume)


def test_base_load_folds_is_none() -> None:
    """Datasets that compute their own splits ship no folds."""
    adapter = MamaMiaDataset(cohort="duke", root=Path("/x"))
    assert adapter.load_folds() is None


# -- factory + split-policy resolution --


def test_factory_builds_mamamia() -> None:
    """The factory picks MamaMiaDataset with the configured cohort and root."""
    config = _dataset_config(
        {
            "name": "mamamia",
            "cohort": "ispy2",
            "root": "/data/mm",
            "split_policy": "auto",
        }
    )
    adapter = build_adapter_from_config(config)
    assert isinstance(adapter, MamaMiaDataset)
    assert adapter.cohort == "ispy2"
    assert adapter.root == Path("/data/mm")


def test_factory_builds_uchicago() -> None:
    """The factory picks UChicagoDataset."""
    config = _dataset_config(
        {"name": "uchicago", "cohort": None, "root": "/data/uc", "split_policy": "auto"}
    )
    assert isinstance(build_adapter_from_config(config), UChicagoDataset)


def test_factory_returns_none_when_unset() -> None:
    """No dataset configured -> None, so callers keep today's behavior."""
    config = _dataset_config(
        {"name": None, "cohort": None, "root": "", "split_policy": "auto"}
    )
    assert build_adapter_from_config(config) is None


def test_factory_rejects_unknown_dataset() -> None:
    """An unrecognized dataset name raises."""
    config = _dataset_config(
        {"name": "tcga", "cohort": None, "root": "/x", "split_policy": "auto"}
    )
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_adapter_from_config(config)


def test_split_policy_auto_uses_adapter_default() -> None:
    """'auto' defers to the adapter default (UChicago -> provided)."""
    adapter = UChicagoDataset(root=Path("/x"))
    config = _dataset_config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "auto"}
    )
    assert resolve_split_policy(config, adapter) == "provided"


def test_split_policy_explicit_override_wins() -> None:
    """An explicit policy overrides the adapter default (see cohorts/README.md — Design decisions)."""
    adapter = UChicagoDataset(root=Path("/x"))
    config = _dataset_config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "compute"}
    )
    assert resolve_split_policy(config, adapter) == "compute"


def test_split_policy_rejects_invalid_value() -> None:
    """An invalid split policy raises."""
    adapter = MamaMiaDataset(cohort="duke", root=Path("/x"))
    config = _dataset_config(
        {"name": "mamamia", "cohort": "duke", "root": "/x", "split_policy": "bogus"}
    )
    with pytest.raises(ValueError, match="Invalid split_policy"):
        resolve_split_policy(config, adapter)


def test_default_config_has_dataset_block() -> None:
    """DEFAULT_CONFIG exposes the dataset block the factory reads."""
    from config import DEFAULT_CONFIG

    dataset = DEFAULT_CONFIG["dataset"]
    assert dataset["name"] is None
    assert dataset["split_policy"] == "auto"
    assert set(dataset) == {"name", "cohort", "root", "split_policy"}


# -- provided folds (synthetic manifest fixture, no real data) --


def _write_manifest(tmp_path: Path) -> Path:
    """Write a tiny UChicago-shaped manifest CSV and return its root dir."""
    root = tmp_path / "uc"
    root.mkdir()
    csv_path = root / "dce2d_internal_ultrafast_manifest.csv"
    csv_path.write_text(
        "exam_id,dataset,patient_key,fold,pcr,phase_files\n"
        'e1,simbiosys,p1,0,1.0,"[""/a/p0.nii.gz""]"\n'
        'e2,uch_nac,p2,1,0.0,"[""/b/p0.nii.gz""]"\n'
        'e3,her2_naclike,p3,2,0.0,"[""/c/p0.nii.gz""]"\n'
    )
    return root


def test_uchicago_load_folds_from_manifest(tmp_path: Path) -> None:
    """UChicago surfaces the manifest fold column as (case_id, fold), int-typed."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    folds = adapter.load_folds()
    assert list(folds.columns) == ["case_id", "fold"]
    assert dict(zip(folds["case_id"], folds["fold"], strict=True)) == {
        "e1": 0,
        "e2": 1,
        "e3": 2,
    }
    assert folds["fold"].dtype.kind == "i"


def test_resolve_folds_provided_returns_manifest_folds(tmp_path: Path) -> None:
    """With split_policy auto (-> provided), resolve_folds returns the folds."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = _dataset_config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "auto"}
    )
    folds = resolve_folds(config, adapter)
    assert folds is not None
    assert set(folds["case_id"]) == {"e1", "e2", "e3"}


def test_resolve_folds_compute_returns_none(tmp_path: Path) -> None:
    """Forcing split_policy compute means no provided folds (compute our own)."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))
    config = _dataset_config(
        {"name": "uchicago", "cohort": None, "root": "/x", "split_policy": "compute"}
    )
    assert resolve_folds(config, adapter) is None


def test_resolve_folds_raises_when_provided_but_none_shipped() -> None:
    """'provided' against a dataset that ships no folds is a clear error."""
    adapter = MamaMiaDataset(cohort="duke", root=Path("/x"))
    config = _dataset_config(
        {"name": "mamamia", "cohort": "duke", "root": "/x", "split_policy": "provided"}
    )
    with pytest.raises(ValueError, match="ships no folds"):
        resolve_folds(config, adapter)


# -- load_timepoints root rebasing --


def _write_manifest_with_preproc_root(tmp_path: Path, preproc_root: Path) -> Path:
    """Write a manifest whose phase_files are anchored under ``preproc_root``."""
    manifest_root = tmp_path / "uc"
    manifest_root.mkdir()
    csv_path = manifest_root / "dce2d_internal_ultrafast_manifest.csv"
    phase_path = preproc_root / "simbiosys" / "e1" / "phase_0000.nii.gz"
    csv_path.write_text(
        "exam_id,dataset,patient_key,fold,pcr,phase_files,preproc_root\n"
        f'e1,simbiosys,p1,0,1.0,"[""{phase_path.as_posix()}""]",{preproc_root.as_posix()}\n'
    )
    return manifest_root


def test_uchicago_load_timepoints_rebases_onto_injected_root(tmp_path: Path) -> None:
    """Phase paths follow the injected root, not the manifest's recorded location.

    Design decision 5 says the pipeline must survive the data moving on disk
    because the root is injected at construction time, not hardcoded. Without
    rebasing, moving the manifest directory would silently leave
    load_timepoints() pointing at the old (possibly gone) location.
    """
    old_preproc_root = tmp_path / "old_location" / "images"
    manifest_root = _write_manifest_with_preproc_root(tmp_path, old_preproc_root)
    adapter = UChicagoDataset(root=manifest_root)

    timepoints = adapter.load_timepoints("e1")

    assert timepoints == [
        manifest_root / "images" / "simbiosys" / "e1" / "phase_0000.nii.gz"
    ]


def test_uchicago_load_timepoints_without_preproc_root_returns_raw_paths(
    tmp_path: Path,
) -> None:
    """A manifest with no preproc_root column falls back to the raw paths."""
    adapter = UChicagoDataset(root=_write_manifest(tmp_path))

    timepoints = adapter.load_timepoints("e1")

    assert timepoints == [Path("/a/p0.nii.gz")]


# -- coverage gaps: missing values in an otherwise-clean manifest --


def _write_manifest_with_gap(tmp_path: Path) -> Path:
    """Write a manifest where one exam has no pcr label and no fold assignment."""
    root = tmp_path / "uc"
    root.mkdir()
    csv_path = root / "dce2d_internal_ultrafast_manifest.csv"
    csv_path.write_text(
        "exam_id,dataset,patient_key,fold,pcr,phase_files\n"
        'e1,simbiosys,p1,0,1.0,"[""/a/p0.nii.gz""]"\n'
        'e2,uch_nac,p2,,,"[""/b/p0.nii.gz""]"\n'
    )
    return root


def test_uchicago_discover_cases_and_load_labels_diverge_on_unlabeled_exam(
    tmp_path: Path,
) -> None:
    """An unlabeled exam is discovered but excluded from load_labels.

    Today's real manifest has zero NaNs (181/181 labeled), so this branch is
    otherwise never exercised.
    """
    adapter = UChicagoDataset(root=_write_manifest_with_gap(tmp_path))

    cases = adapter.discover_cases()
    labels = adapter.load_labels()

    assert cases == ["e1", "e2"]
    assert list(labels["case_id"]) == ["e1"]


def test_uchicago_load_folds_drops_exams_with_no_fold_assignment(
    tmp_path: Path,
) -> None:
    """An exam with a blank fold value is excluded from load_folds."""
    adapter = UChicagoDataset(root=_write_manifest_with_gap(tmp_path))

    folds = adapter.load_folds()

    assert list(folds["case_id"]) == ["e1"]
