"""Self-supervised contrast forecasting for UChicago vessel graphs.

Implements the voxel-as-node pilot from ``docs/design/contrast_pretraining.md``:
pretrain an encoder to forecast each node's *future* contrast enhancement from
its own and its neighbours' earlier frames, so the encoder learns how the bolus
propagates across vascular edges (the perfusion structure we hope transfers to
downstream pCR prediction).

The production contract is implemented by :mod:`gnn.pretrain.uchicago`:

- **Target** = relative enhancement against a graph-neighbour-repaired baseline,
  with original baseline validity retained as a separate channel and loss mask.
- **Temporal encoder** = per-frame GNN encoder + a per-node recurrent temporal
  head (alternative (b) in the doc): reuses the existing ``GCNConv`` machinery
  and keeps "how the graph mixes" separate from "how time is modelled".
- **Graph ablation** = a message-passing-free per-node forecaster with the same
  temporal head, to run the doc's §7.ii "does the graph matter" check.

``gnn.pretrain.duke_forecast`` remains a historical placeholder-data smoke; it
is not the UChicago production entrypoint.
"""

from __future__ import annotations

from gnn.pretrain.baselines import last_frame_forecast, temporal_mean_forecast
from gnn.pretrain.forecast import ForecastHorizon, split_forecast_window
from gnn.pretrain.loss import masked_mae
from gnn.pretrain.model import (
    ContrastForecastGNN,
    PerNodeForecaster,
    PerNodeTemporalEncoder,
    TemporalGraphEncoder,
)
from gnn.pretrain.node_series import (
    junction_node_series,
    segment_node_series,
    voxel_node_series,
)

__all__ = [
    "ContrastForecastGNN",
    "ForecastHorizon",
    "PerNodeForecaster",
    "PerNodeTemporalEncoder",
    "TemporalGraphEncoder",
    "junction_node_series",
    "last_frame_forecast",
    "masked_mae",
    "segment_node_series",
    "split_forecast_window",
    "temporal_mean_forecast",
    "voxel_node_series",
]
