r"""CLI to (re)plot ``graph_qc.csv`` (and optionally a ``predictions.csv``).

The build (``gnn/data_loader.py``) and training (``gnn/train.py``) pipelines
now write these plots automatically -- see ``gnn/graph_qc_plots.py`` for the
shared plotting code both call. This CLI exists for ad-hoc/manual regeneration
(e.g. against an older cache, or a predictions.csv from a run that hasn't been
retrained) without needing to rerun a full build or training job.

Usage:
    python -m gnn.plot_graph_qc --cache-dir <gnn cache dir> --out-dir <dir> \
        [--predictions-csv <path/to/predictions.csv>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn.data_loader import VanguardCenterlineDataset
from gnn.graph_qc_plots import write_build_time_plots, write_prediction_plot

_CACHE_MANIFEST_NAME = "cache_manifest.json"
_GRAPH_QC_NAME = "graph_qc.csv"


def _load_graph_qc(cache_dir: Path) -> pd.DataFrame:
    qc_path = cache_dir / "processed" / _GRAPH_QC_NAME
    if not qc_path.exists():
        raise FileNotFoundError(
            f"{qc_path} not found -- this cache was built before graph_qc.csv "
            "was added, or with no_cache=True. Rebuild it first."
        )
    return pd.read_csv(qc_path)


def _load_node_features(cache_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Reload the cached dataset (cache hit, no rebuild) for node-level values.

    Reconstructs the exact settings the cache was built with from
    ``cache_manifest.json`` so this always hits the cache rather than risking
    a manifest mismatch against some other default.
    """
    manifest = json.loads((cache_dir / "processed" / _CACHE_MANIFEST_NAME).read_text())
    dataset = VanguardCenterlineDataset(
        root=manifest["centerline_root"],
        labels_path=manifest["labels_path"],
        dce_root=manifest["dce_root"],
        cache_dir=cache_dir,
        cases=manifest["cases"],
        node_mode=manifest["node_mode"],
        node_features=manifest["node_features"],
        id_column=manifest["id_column"],
        label_column=manifest["label_column"],
    )
    node_features = list(manifest["node_features"])
    rows = []
    for data in dataset:
        x = data.x.numpy()
        for node_idx in range(x.shape[0]):
            row = {
                "case_id": data.case_id,
                "dataset": data.dataset,
                "pcr": int(data.y.item()),
            }
            row.update({name: x[node_idx, i] for i, name in enumerate(node_features)})
            rows.append(row)
    return pd.DataFrame(rows), node_features


def main() -> None:
    """Parse CLI args and write the graph_qc plots to ``--out-dir``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, required=True, help="GNN dataset cache dir."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional predictions.csv from a gnn/train.py run, for the "
        "prediction-vs-num_nodes plot.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    qc = _load_graph_qc(args.cache_dir)
    node_df, node_features = _load_node_features(args.cache_dir)
    write_build_time_plots(qc, node_df, node_features, args.out_dir)

    if args.predictions_csv is not None:
        predictions = pd.read_csv(args.predictions_csv)
        write_prediction_plot(predictions, qc, args.out_dir)

    print(f"Wrote graph QC plots to {args.out_dir}")


if __name__ == "__main__":
    main()
