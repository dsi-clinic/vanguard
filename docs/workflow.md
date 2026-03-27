# Workflow Guide

This is the short version of how work is expected to move through the repo.

## Basic Process

1. Create or pick up an issue.
2. Assign it to yourself when you start working.
3. Create a branch from `main`.
4. Open a pull request early, even if it is still in progress.
5. Update the same PR as you respond to review.
6. Merge after review and passing checks.

## Suggested Conventions

- Put the issue number in the branch name when possible.
- Link the issue in the PR description.
- Keep one logical change per PR when practical.
- Push fixes to the same branch instead of opening replacement PRs.

## Practical Notes For This Repo

- Heavy compute should run through Slurm, not on the headnode.
- Use the `vanguard` micromamba environment for project Python commands.
- If you change paths in a config, treat them as local run settings and document the change in the PR if it matters to reproducibility.
