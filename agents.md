# Cluster Safety Instructions for AI Agents
You are working on a shared HPC/cluster environment. Your first responsibility is to avoid disrupting the cluster, other users, or user data.

## Hard Rules
- Never run compute-heavy work on the login/head node.
- Do not run training, inference, benchmarking, large data processing, model evaluation, simulations, compilation-heavy builds, or multiprocessing jobs directly on the head node.
- Do not start background jobs on the head node with `&`, `nohup`, `tmux`, `screen`, or long-running shell loops unless explicitly instructed and the task is clearly lightweight.
- Do not run commands that are expected to use significant CPU, RAM, GPU, disk I/O, or network I/O outside a scheduler job.
- If a command might run for more than ~30 seconds, use more than ~1 GB RAM, use multiple CPU cores, touch many files, or process large data, submit it through the scheduler.
- Never use GPUs outside an allocated GPU job.
- Never launch job arrays, large sweeps, or many parallel jobs without explicit user approval.

## Head Node Usage
The login/head node may only be used for lightweight coordination tasks, such as:
- Inspecting files with `ls`, `find`, `rg`, `head`, `tail`, `sed`, `cat` on small files
- Editing code
- Checking git status or diffs
- Reading small logs
- Submitting, checking, or cancelling scheduler jobs
- Inspecting environment/module availability
- Creating small scripts or config files

Before running anything potentially expensive, assume the current shell is on a login node unless proven otherwise.

## Data and Filesystem Safety
- Do not delete, overwrite, or move large datasets unless explicitly instructed.
- Do not run broad destructive commands such as `rm -rf *`, `find ... -delete`, or mass renames without explicit confirmation.
- Prefer writing large outputs to the appropriate scratch/work directory, not home.
- Avoid generating large numbers of tiny files unless required.
- Check disk usage before large writes using commands like `df -h` or quota tools if available.
- Do not change permissions recursively with `chmod -R` or ownership recursively with `chown -R` unless explicitly instructed.
- Do not modify shared reference data, shared environments, or other users' files.

## When Unsure
If there is any doubt about whether something is safe to run on the head node, do not run it there. Create a small scheduler job or ask the user for confirmation.

Cluster stability and data safety take priority over speed.
