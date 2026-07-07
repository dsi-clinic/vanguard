"""Unit tests for the dataset-adapter scaffold (Step 1).

These exercise the adapters in isolation — construction, the design-doc claim
that ``MamaMiaDataset`` needs no method overrides (§6.5), the ``UChicagoDataset``
override set, the factory, and split-policy resolution — without touching any
real data on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import ConfigNode
from datasets import (
    DatasetAdapter,
    MamaMiaDataset,
    UChicagoDataset,
    build_adapter_from_config,
    resolve_split_policy,
)


def _overridden_methods(subclass: type, base: type) -> set[str]:
    """Return names of base methods the subclass redefines (excluding dunders)."""
    overridden: set[str] = set()
    for name in dir(base):
        if name.startswith("__"):
            continue
        base_attr = getattr(base, name)
        if not callable(base_attr):
            continue
        if getattr(subclass, name) is not base_attr:
            overridden.add(name)
    return overridden


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


# -- the §6.5 claim: MamaMiaDataset overrides no methods --


def test_mamamia_overrides_no_methods() -> None:
    """MamaMiaDataset relies entirely on base behavior (design doc §6.5)."""
    assert _overridden_methods(MamaMiaDataset, DatasetAdapter) == set()


def test_uchicago_overrides_exactly_the_expected_methods() -> None:
    """UChicago overrides only the methods identified in §6.5."""
    expected = {
        "discover_cases",
        "load_timepoints",
        "preprocess",
        "case_dataset_name",
        "group_key",
        "load_labels",
    }
    assert _overridden_methods(UChicagoDataset, DatasetAdapter) == expected


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


def test_uchicago_preprocess_is_a_documented_stub() -> None:
    """UChicago preprocessing raises until the frozen-copy port lands."""
    adapter = UChicagoDataset(root=Path("/data/uchicago"))
    with pytest.raises(NotImplementedError, match="not ported yet"):
        adapter.preprocess(np.zeros((2, 2, 2)))


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
    """An explicit policy overrides the adapter default (design doc §10 decision 3)."""
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
