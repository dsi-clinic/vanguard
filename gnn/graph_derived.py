"""Graph-derived scalar features for the GNN pCR classifier.

Unlike ``gnn.clinical`` (external clinical/demographic data, joined by
``case_id``) or ``gnn.morphometry`` (per-case JSON files), these are scalars
already computed on the ``Data`` object itself during graph building --
pulling them out is a pure attribute read, never missing, never needing
imputation or encoding.

**Confound risk, not a default.** ``num_connected_components`` (the number of
disconnected vessel-tree components in a case's graph) is the only column
here today, and it is deliberately excluded from any default feature list.
Disconnected components can arise from real vascular fragmentation, but
equally from segmentation noise, motion, or site/scanner resolution
differences -- the same confound-risk class the repo already treats
``num_nodes`` as (see ``gnn/README.md``'s framing: a model could be learning
tumor size or a site effect instead of enhancement kinetics; `graph_qc.csv`'s
``num_connected_components`` vs. ``dataset``/``pcr`` columns exist precisely
so this can be checked before trusting it as a feature, the same audit
`gnn.graph_qc_plots.plot_num_nodes_vs_group` already does for `num_nodes`).
It is available via ``gnn_graph_features`` for anyone who explicitly opts in
and checks that audit first, mirroring how ``gnn.clinical.SITE_CLINICAL_COLUMNS``
is available but never defaulted-on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch_geometric.data import Data

GRAPH_DERIVED_COLUMNS: tuple[str, ...] = ("num_connected_components",)


def build_graph_derived_feature_matrix(
    data_list: Sequence[Data], columns: Sequence[str]
) -> np.ndarray:
    """Read ``columns`` straight off each already-built ``Data`` object.

    No imputation or encoding: every column here is a scalar the graph build
    always sets (see module docstring), so a missing value is a bug in the
    build, not an expected case -- accessing a name outside
    ``GRAPH_DERIVED_COLUMNS`` or absent from a ``Data`` object raises rather
    than being silently defaulted.

    Returns a ``(len(data_list), len(columns))`` float matrix, one row per
    graph in ``data_list`` order.
    """
    unknown = [c for c in columns if c not in GRAPH_DERIVED_COLUMNS]
    if unknown:
        raise ValueError(
            f"Unknown graph-derived columns {unknown}; supported: "
            f"{sorted(GRAPH_DERIVED_COLUMNS)}"
        )
    rows: list[list[float]] = []
    for data in data_list:
        rows.append([float(getattr(data, name)) for name in columns])
    return np.asarray(rows, dtype=np.float64)
