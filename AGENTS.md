# Agent Instructions for Vanguard

These are shared instructions for AI coding agents working in this repo.

## Shared Rules
- Read existing code before adding new code.
- Prefer clear, small changes over large rewrites.
- Treat raw imaging data as read-only.
- Do not discard, flatten, or obscure dynamic/kinematic information in DCE-MRI data.
- Preserve how signal intensity changes across time points unless the user explicitly asks for a derived summary.
- For visual comparison across DCE-MRI time points, prefer one shared intensity window computed over the full 4D image.
- Avoid per-timepoint windowing by default because it can hide real enhancement or washout patterns.
- Use Slurm for heavy compute; do not run large jobs on the head node.
- Record important commands, configs, input paths, output paths, and job IDs.
- Keep outputs reproducible and easy to find.
- Before trusting or reporting results, sanity-check inputs, outputs, and assumptions.

## Environment
- Run commands from `~/vanguard`.
- Activate the project environment once per terminal session with:
  `micromamba activate vanguard`
- The environment lives at:
  `/ess/scratch/scratch1/aakrithiram/micromamba/envs/vanguard`
- Before running Python or project commands, verify:
  `which python`
  It should point to the Vanguard environment.
- One-off command pattern:
  `micromamba run -n vanguard python <script.py>`
- Every checked-in `.slurm` script self-activates the environment via
  `eval "$(micromamba shell hook -s bash)"; micromamba activate vanguard` -- you don't need to
  activate it separately before `sbatch`.

## HPC / Slurm
- Do not run heavy compute on the head node.
- Use Slurm batch jobs for long, memory-heavy, GPU-heavy, or many-case runs.
- Keep Slurm scripts in a clear location, preferably `slurm/` if appropriate for the repo.
- Send Slurm logs to a reproducible logs directory, such as `logs/slurm/` or another project-agreed location.
- Record Slurm job IDs.
- Standard check command:
  `squeue -u $USER`
- Standard submit pattern:
  `sbatch path/to/job.sbatch`
- If a job fails, inspect the Slurm log before rerunning.
- No `submit.py`/`submitit` entrypoint exists in this repo -- checked-in `.slurm` sbatch scripts
  (under `slurm/`, `gnn/slurm/`, `deepsets/slurm/`) are the launch convention.

### Site facts (this cluster)
Recorded from live `sinfo`/`sacctmgr` inspection. Refresh only if commands contradict this or
admins change policy.

- **Default account**: `karczmar-lab`
- **Default partition**: `tier1q` (10-day time limit, large and typically has idle capacity;
  this is what every current `gnn/slurm/*.slurm` and `deepsets/slurm/*.slurm` script uses).
  The older `--partition=general` referenced by some `slurm/*.slurm` scripts does **not** exist
  on this cluster -- don't use it.
- **Preemptible/overflow QoS**: none identified for this account (QOS shows
  `nonpreemptible,norm...`). `express` (6h limit, 128 nodes) exists as a separate partition for
  short jobs, but isn't a preemptible tier -- treat it as a short-job option, not overflow
  capacity to prefer by default.
- **Important limits**: `tier1q` allows up to 10-day jobs; plenty of idle capacity as of last
  check. No special per-job core/mem caps observed beyond standard `--cpus-per-task`/`--mem`.

## Vanguard Data Layout
These placeholders should be filled in as the team confirms paths:

- MAMA-MIA images: TODO
- Tumor masks: TODO
- Breast masks: TODO
- Centerline outputs: TODO
- pCR labels / split CSV: TODO
- Derived feature outputs: TODO
- QC outputs: TODO

Raw images, labels, and masks should be treated as read-only.

Derived outputs should go in clearly named output folders with enough provenance to reproduce them.

### Site-specific data
Concrete dataset roots, cohort counts, and raw-source paths are site-specific and are not
part of the repo's portable contract. The UChicago ultrafast cohorts and their restricted
Karczmar-lab DICOM sources are documented in [`UCHICAGO.md`](UCHICAGO.md). Those paths are
valid only on the UChicago cluster -- anywhere else, prepare an equivalent dataset or arrange
access, and point `data_paths` in your own YAML at it rather than assuming those roots exist.

## Project Conventions
- Keep scripts and outputs named clearly enough that another student can understand what they are for.
- `analysis/` is for `.ipynb` notebooks only. Do not add `.py`, `.sh`, or `.slurm` files there, including one-off diagnostics, QC helpers, and plotting scripts. Python scripts go in `scripts/` (or the owning package: `gnn/`, `deepsets/`, `graph_extraction/`, `preprocessing/`, ...); sbatch scripts go in `slurm/` or the package's own `slurm/`. If a notebook needs a helper, put the helper in `scripts/` and import it. Older `.py` files still sitting in `analysis/` are a violation to migrate, not a precedent to follow.
- Write comments and docstrings for a newcomer reading the code as it is now, in the present tense: state what the code does and why it is shaped this way. Do not narrate the change that produced it ("now we…", "no longer…", "instead of the old…", "removed X", "backward compatible with…") or reference prior approaches. That transitional rationale belongs in the commit message, lab notebook, and agent memory — in the code it bloats and goes stale. If a comment only makes sense to someone who saw the diff, it does not belong in the code.
- Prefer saved CSVs, plots, QC panels, and short README/provenance notes for important outputs.
- Document split/CV policy when training or evaluating models.
- Do not change cohort definitions, labels, or train/test splits without making the change explicit.
- Do not commit evolving or personal working docs — migration plans, scratch notes, study guides, TODO/roadmap files. These clutter branches and PRs and go stale. Keep them in a gitignored location (`docs/design/`, `agents.local.md`, or an untracked scratch path), not in `docs/` or the repo root.
- Committed docs must be durable references others rely on: READMEs, data-layout/policy docs, provenance/results write-ups (e.g. `docs/issue*.md`), and **decision records that code or configs point to** (e.g. `gnn/DESIGN_segment_graph.md` and `gnn/PLAN_advanced_modeling.md`, which `config.py` cites by name). A "plan" or "design" file earns a committed spot only as a stable decision record referenced from code — not as a work-in-progress roadmap.

## Local Preferences
If `agents.local.md` exists, read it after this file.

`agents.local.md` is for personal preferences, student-specific context, and temporary working notes. It should not be committed.

## Data Visualization Guidelines
- Use color only to encode information, not for decoration.
- If all bars represent the same type of data, use a single, consistent color.
- Reserve contrasting colors only to highlight a specific result or distinguish meaningful categories.
- Avoid unnecessary visual elements ("chartjunk") such as excessive colors, gradients, 3D effects, and heavy styling.
- Favor clean, simple figures that emphasize the data rather than the design.

## Avoiding Repeated Mistakes
After any material mistake, near-miss, wasted compute, confusing workflow, or user correction, identify the general rule that would have prevented it and propose a concise `AGENTS.md` change in the next substantive update. If an existing rule already covers the incident, explain why it was missed and propose clearer wording, placement, or triggers instead of a duplicate. Prefer reusable principles over incident-specific prohibitions; do not edit `AGENTS.md` without approval.
