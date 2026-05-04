# Deep Sets inclusion rule default (issue #121)

This note records the inclusion-rule comparison framework added to `build_deepsets_dataset.py`, the default rule decision, and how to reproduce the summary table.

## Compared rule set

The builder now supports a compact comparison set:

- `local_radius_with_fallback`
- `local_radius_only`
- `nearest_64_only`

Per build run, it writes `inclusion_rule_summary.csv` to the output directory with:

- `cases_written`
- `cases_skipped`
- `fallback_fraction`
- `num_points_median`
- `num_points_range`

## Chosen default

Default: `local_radius_with_fallback`

Rationale:

- Preserves current production behavior while making alternatives measurable.
- Maintains tumor-local inclusion as the primary selection criterion.
- Avoids dropping valid cases when no points pass local-radius filtering.
- Keeps per-case point count bounded through nearest-64 fallback.

## Comparison stats (fixture run)

From `configs/deepsets_issue121_fixture.yaml` written to
`experiments/issue121_fixture_build/inclusion_rule_summary.csv`:

These are fixture-level stats from a four-case synthetic sample, not cohort-level
ISPY2 statistics.

| inclusion_rule | cases_written | cases_skipped | fallback_fraction | num_points_median | num_points_range |
|---|---:|---:|---:|---:|---|
| `local_radius_with_fallback` | 4 | 0 | 0.5 | 1.5 | `1-3` |
| `local_radius_only` | 2 | 2 | 0.0 | 1.0 | `1-1` |
| `nearest_64_only` | 4 | 0 | 0.0 | 2.0 | `1-3` |

Interpretation:

- `local_radius_only` drops cases when no points fall in the local radius.
- `nearest_64_only` never drops cases but ignores the tumor-local gate entirely.
- `local_radius_with_fallback` keeps tumor-local selection by default while avoiding
  case drops via bounded fallback.

## Reproduce comparison stats

Use the Deep Sets build stage with a config that includes:

- `model_params.deepsets_inclusion_rule: local_radius_with_fallback`
- `model_params.deepsets_compare_inclusion_rules` listing the compact rule set

Then inspect:

- `OUT_ROOT/inclusion_rule_summary.csv`

Fixture example:

```bash
PYTHONPATH=. python build_deepsets_dataset.py \
  --config configs/deepsets_issue121_fixture.yaml \
  --output-dir experiments/issue121_fixture_build \
  --num-shards 1 \
  --shard-index 0
```

Full-cohort example (submit through Slurm; do not run this build directly on a
login/head node):

```bash
MODE=build \
CONFIG=configs/deepsets_ispy2.yaml \
OUT_ROOT=experiments/deepsets_issue121_compare \
NUM_SHARDS=1 \
SLURM_ARRAY_TASK_ID=0 \
sbatch slurm/deepsets_job.slurm
```
