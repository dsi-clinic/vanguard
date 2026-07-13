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

## Project Conventions
- Keep scripts and outputs named clearly enough that another student can understand what they are for.
- Prefer saved CSVs, plots, QC panels, and short README/provenance notes for important outputs.
- Document split/CV policy when training or evaluating models.
- Do not change cohort definitions, labels, or train/test splits without making the change explicit.

## Local Preferences
If `agents.local.md` exists, read it after this file.

`agents.local.md` is for personal preferences, student-specific context, and temporary working notes. It should not be committed.

## Data Visualization Guidelines
- Use color only to encode information, not for decoration.
- If all bars represent the same type of data, use a single, consistent color.
- Reserve contrasting colors only to highlight a specific result or distinguish meaningful categories.
- Avoid unnecessary visual elements ("chartjunk") such as excessive colors, gradients, 3D effects, and heavy styling.
- Favor clean, simple figures that emphasize the data rather than the design.
