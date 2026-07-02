## Development Checks

Before opening or updating a PR, run the same linting checks that GitHub Actions runs:

```bash
pre-commit run --all-files
```

### What pre-commit is doing

pre-commit is just a command-line tool that reads the repo's config file:

`.pre-commit-config.yaml`


That file lists the checks to run. In this repo, it runs:

- basic cleanup checks, like trailing whitespace, final newlines, YAML validity, and large-file checks
- Ruff linting, with some auto-fixes enabled
- Ruff formatting

Ruff's detailed rule settings live in:

`pyproject.toml`

So the flow is:

pre-commit command
  -> reads .pre-commit-config.yaml
  -> runs the listed hooks
  -> Ruff uses pyproject.toml for lint/format rules

### One-time setup

After activating the environment:

```bash 
micromamba activate vanguard
pre-commit install
```

This adds a local git hook so the same checks run automatically when you commit.

### Normal use

Run all checks before pushing:

```bash
pre-commit run --all-files
```

Many issues are auto-fixed. If that happens, inspect and stage the edits:

```bash
git diff
git status
git add path/to/changed_file.py
```

Then rerun:

```bash 
pre-commit run --all-files
```

A PR should be pushed only after this passes locally.
