# Slurm Site Notes — RANDI cluster

## Default account
`karczmar-lab`

## Partitions
| Partition | Timelimit | Notes |
|-----------|-----------|-------|
| `tier1q`  | 10 days   | Default for CPU jobs (148 nodes); use `PARTITION=tier1q` |
| `tier2q`  | 10 days   | Secondary CPU tier (44 nodes) |
| `tier3q`  | 10 days   | Smaller tier (7 nodes) |
| `express` | 6 hours   | Short jobs, 4 nodes |
| `gpuq`    | 10 days   | A100-PCIe-40GB GPUs |
| `sxmq`    | 10 days   | A100-SXM4-40GB GPUs |
| `ghq`     | 10 days   | GH200-96GB GPUs |
| `clinical`| 10 days   | Clinical data nodes |

## QoS
`nonpreemptible, normal, opportunistic, priority, standard`

## Preemptible / overflow
`opportunistic` QoS — use for short probes or when standard capacity is full.

## Important limits
- Login/head node: lightweight coordination only (file edits, squeue, sbatch, small reads).
- No heavy compute on head node — CLAUDE.md enforces this as a hard rule.

## Native launcher
Repo uses a three-stage Slurm array pipeline:
1. `slurm/submit_independent_signal_matrix_array.sh` — orchestrates cache → array → merge
2. Per-experiment wrappers (`slurm/submit_*.sh`) set CONFIG and OUT_ROOT, then exec the above
3. Array tasks: `slurm/submit_ablation_arm_fold_array.slurm` runs `modeling/run_arm_fold.py`
4. Merge: `modeling/merge_results.py` aggregates fold outputs

Submission pattern:
```bash
PARTITION=tier1q bash slurm/submit_<experiment>.sh
```

## Environment convention
`micromamba` env named `vanguard`:
```bash
micromamba run -n vanguard python <script>
# or inside jobs:
eval "$(micromamba shell hook -s bash)"; micromamba activate vanguard
```
Python binary: `${HOME}/micromamba/envs/vanguard/bin/python`
