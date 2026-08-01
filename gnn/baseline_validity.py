"""Per-node precontrast-baseline validity, and local repair of invalid baselines.

Relative enhancement divides by S0, the mean of a voxel's precontrast frames. A
small tail of vessel voxels has S0 ~ 0 -- an acquisition dropout before contrast
rather than absent tissue, since those same voxels enhance normally afterwards
-- and dividing by it turns (S - S0)/S0 into a divide-by-noise artifact.

This module records **which** baselines were actually observed, and repairs the
rest locally so the affected nodes still carry usable inputs:

1. ``observed_valid`` marks nodes whose own baseline was finite and above a
   per-case validity threshold. It is the record of what was measured and is
   never overwritten -- downstream it is what gates the graph readout, so an
   unobserved baseline can never influence the loss.
2. ``repair_baselines`` fills an invalid node's S0 from the median of its
   graph neighbours that already have one, spreading one hop per round until
   nothing more can be filled. The filled value gives the node a locally
   plausible scale so it can carry messages between its valid neighbours
   without claiming its baseline was observed.

Keeping the node rather than deleting it matters here: the graph is a vessel
centerline, so dropping a mid-segment voxel severs the path between its two
neighbours and fragments exactly the topology the model is meant to use.

The repair reads only the case's own neighbouring voxels, never cohort
statistics, so a case's imputed values do not depend on which cross-validation
fold it lands in. That is why this can run once at cache-build time, unlike the
clinical/morphometry imputers which must be fit per fold (see
``gnn.data_loader``).

An alternative that needs no imputation at all is to mark the invalid nodes and
mask them at the readout, leaving their features as-is. ``gnn.data_loader``
exposes both via ``baseline_repair={"off","mask","impute"}`` so the two can be
compared rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

# A node's baseline counts as observed when it exceeds this fraction of the
# case's own median node baseline. The threshold is deliberately relative to the
# case's precontrast intensity scale and nothing else: per-case median S0 spans
# ~36x across the UChicago cohort (7.8 to 278.7), so no fixed absolute value
# means "near-void" in every case, and referencing the voxel's own peak signal
# would leak post-contrast (future) information into a precontrast decision.
# Not to be tuned against pCR performance -- it is an acquisition-validity
# threshold, not a model hyperparameter.
DEFAULT_VALID_FRAC = 0.05

_MODE_OFF = "off"
_MODE_MASK = "mask"
_MODE_IMPUTE = "impute"
REPAIR_MODES = (_MODE_OFF, _MODE_MASK, _MODE_IMPUTE)


@dataclass(frozen=True)
class BaselineRepair:
    """Outcome of validity marking + imputation for one graph.

    ``observed_valid`` is the measurement record and is what gates the readout.
    ``has_baseline`` additionally includes successfully imputed nodes and is
    what decides whether a node gets usable features at all.
    """

    s0_filled: np.ndarray
    observed_valid: np.ndarray
    has_baseline: np.ndarray
    imputation_round: np.ndarray
    threshold: float

    @property
    def n_nodes(self) -> int:
        """Total nodes considered."""
        return int(self.observed_valid.size)

    @property
    def n_invalid(self) -> int:
        """Nodes whose own baseline was not observed."""
        return int((~self.observed_valid).sum())

    @property
    def n_imputed(self) -> int:
        """Invalid nodes that received a neighbour-derived baseline."""
        return int((self.has_baseline & ~self.observed_valid).sum())

    @property
    def n_unrepaired(self) -> int:
        """Invalid nodes left with no usable baseline."""
        return int((~self.has_baseline).sum())

    @property
    def max_round(self) -> int:
        """Deepest imputation round used (0 when nothing was imputed)."""
        return int(self.imputation_round.max()) if self.imputation_round.size else 0


def observed_valid_mask(
    s0: np.ndarray, *, valid_frac: float = DEFAULT_VALID_FRAC
) -> tuple[np.ndarray, float]:
    """Mark nodes with a finite baseline above the case's validity threshold.

    Returns the mask and the absolute threshold it used, so the threshold can be
    recorded per case in QC rather than inferred later.
    """
    if not 0.0 <= valid_frac < 1.0:
        raise ValueError(f"valid_frac must be in [0, 1); got {valid_frac}")
    s0 = np.asarray(s0, dtype=float)
    finite = np.isfinite(s0)
    if not finite.any():
        raise ValueError("no finite node baselines; the DCE series is unusable")
    # Median over finite baselines only -- including NaNs would drag the case's
    # own reference scale toward the very voxels being screened out.
    threshold = float(valid_frac * np.median(s0[finite]))
    return finite & (s0 > threshold), threshold


def repair_baselines(
    graph: nx.Graph,
    nodes: list,
    s0: np.ndarray,
    *,
    mode: str,
    valid_frac: float = DEFAULT_VALID_FRAC,
) -> BaselineRepair:
    """Mark baseline validity for one graph, and under ``"impute"`` also fill it.

    ``mode`` is ``"mask"`` (record validity, leave baselines untouched) or
    ``"impute"`` (additionally fill invalid baselines from valid neighbours).
    ``"off"`` is handled by the caller, which skips this module entirely.
    """
    if mode not in (_MODE_MASK, _MODE_IMPUTE):
        raise ValueError(
            f"mode must be {_MODE_MASK!r} or {_MODE_IMPUTE!r}; got {mode!r}"
        )
    observed_valid, threshold = observed_valid_mask(s0, valid_frac=valid_frac)
    if mode == _MODE_MASK:
        s0 = np.asarray(s0, dtype=float)
        return BaselineRepair(
            s0_filled=np.where(observed_valid, s0, np.nan),
            observed_valid=observed_valid,
            has_baseline=observed_valid.copy(),
            imputation_round=np.zeros(s0.size, dtype=np.int64),
            threshold=threshold,
        )
    return _impute_baselines(
        graph,
        nodes,
        s0,
        observed_valid,
        threshold=threshold,
    )


def _impute_baselines(
    graph: nx.Graph,
    nodes: list,
    s0: np.ndarray,
    observed_valid: np.ndarray,
    *,
    threshold: float,
) -> BaselineRepair:
    """Fill invalid baselines from valid graph neighbours, one hop per round.

    Each round collects every eligible node with at least one neighbour that
    already has a baseline, computes all their medians from the state at the
    START of the round, then applies them together -- so a value travels at most
    one neighbourhood step per round and the fill order cannot depend on node
    iteration order. Repeats until no node can be filled.

    ``nodes`` must be the node ordering ``s0`` and the masks are indexed by.

    Eligibility depends only on the precontrast baseline and graph adjacency.
    Postcontrast frames are deliberately not inspected: using future signal to
    decide whether a baseline is valid would leak the forecasting target into
    preprocessing.
    """
    s0 = np.asarray(s0, dtype=float)
    observed_valid = np.asarray(observed_valid, dtype=bool)
    if not (len(nodes) == s0.size == observed_valid.size):
        raise ValueError("nodes, s0, and observed_valid must align")

    index_of = {node: i for i, node in enumerate(nodes)}
    s0_filled = np.where(observed_valid, s0, np.nan)
    has_baseline = observed_valid.copy()
    imputation_round = np.zeros(s0.size, dtype=np.int64)

    round_number = 0
    while True:
        round_number += 1
        pending: dict[int, float] = {}
        for i, node in enumerate(nodes):
            if has_baseline[i]:
                continue
            usable = [
                s0_filled[index_of[nbr]]
                for nbr in graph.neighbors(node)
                if nbr in index_of and has_baseline[index_of[nbr]]
            ]
            if not usable:
                continue
            candidate = float(np.median(usable))
            pending[i] = candidate
        if not pending:
            break
        for i, value in pending.items():
            s0_filled[i] = value
            has_baseline[i] = True
            imputation_round[i] = round_number

    return BaselineRepair(
        s0_filled=s0_filled,
        observed_valid=observed_valid,
        has_baseline=has_baseline,
        imputation_round=imputation_round,
        threshold=threshold,
    )
