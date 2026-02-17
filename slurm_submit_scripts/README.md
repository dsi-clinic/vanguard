# SLURM Submit Scripts

This directory contains cluster submit scripts for the Vanguard pipeline.

## Available Scripts

### Vessel Segmentation

1. `submit_vessel_segmentation.slurm`
- Main GPU segmentation job.

2. `submit_vessel_segmentation_optimized.slurm`
- Higher-resource segmentation variant.

3. `submit_vessel_segmentation_array.slurm`
- Array segmentation job (one file index per task).

### Legacy-vs-Primary Benchmark Suite

4. `submit_legacy_vs_primary_benchmark.sh`
- Head-node orchestration helper.
- Generates study manifest, submits Slurm array benchmark job, then submits reducer with `afterok` dependency.

5. `submit_legacy_vs_primary_benchmark_array.slurm`
- Array worker script.
- Each task runs one study through `batch_processing/benchmark_legacy_vs_primary.py`.
- Produces per-study metrics + 3D visualization artifacts via the compare script output.

6. `submit_legacy_vs_primary_benchmark_reduce.slurm`
- Reducer job.
- Runs `batch_processing/reduce_legacy_vs_primary.py` after array completion.

## Recommended Benchmark Workflow (Head Node)

```bash
# Choose an output directory for one benchmark run
OUT_DIR="/net/projects2/vanguard/benchmarks/legacy_vs_primary/run_$(date +%Y%m%d_%H%M%S)"

# Optional overrides:
# export SEGMENTATION_DIR=/net/projects2/vanguard/vessel_segmentations
# export COMPARE_SCRIPT=/path/to/graph_extraction/run_compare_legacy_pipeline_debug.py
# export AUTO_TUMOR_MASK=1

slurm_submit_scripts/submit_legacy_vs_primary_benchmark.sh "${OUT_DIR}"
```

The helper prints the submitted array/reducer job IDs and writes:
- `${OUT_DIR}/study_ids.txt`
- `${OUT_DIR}/studies/<study_id>/benchmark_record.json`
- `${OUT_DIR}/benchmark_reduced_per_study.csv` (after reducer)
- `${OUT_DIR}/benchmark_reduced_summary.json` (after reducer)

## Direct Manual Submission (Advanced)

```bash
# 1) Build manifest first
micromamba run -n vanguard python batch_processing/benchmark_legacy_vs_primary.py \
  --segmentation-dir /net/projects2/vanguard/vessel_segmentations \
  --output-dir "${OUT_DIR}" \
  --manifest-only

# 2) Submit array manually
N=$(grep -cv '^\s*$' "${OUT_DIR}/study_ids.txt")
ARRAY_JOB_ID=$(sbatch --parsable --array "0-$((N-1))" \
  --export=ALL,BENCH_MANIFEST="${OUT_DIR}/study_ids.txt",BENCH_OUTPUT_DIR="${OUT_DIR}" \
  slurm_submit_scripts/submit_legacy_vs_primary_benchmark_array.slurm)

# 3) Submit reducer dependency
sbatch --dependency "afterok:${ARRAY_JOB_ID}" \
  --export=ALL,BENCH_OUTPUT_DIR="${OUT_DIR}" \
  slurm_submit_scripts/submit_legacy_vs_primary_benchmark_reduce.slurm
```

## Monitoring

```bash
squeue -u "$USER"

tail -f logs/legacy-primary-bench-<ARRAY_JOB_ID>-<TASK_ID>.out
tail -f logs/legacy-primary-reduce-<JOB_ID>.out
```

## Notes

- Benchmark scripts assume head-node orchestration for manifest generation and job submission.
- Email notifications use the `#SBATCH --mail-*` settings in each `.slurm` script.
- Paths and compare script can be overridden through environment variables in the head-node helper.
