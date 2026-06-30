#!/usr/bin/env bash
# Issue #121 fixture smoke: build inclusion-rule summary on tiny local data.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FIXTURE_ROOT="${REPO_ROOT}/tmp/issue121_fixture"
if [[ ! -d "${FIXTURE_ROOT}/centerlines" ]]; then
  echo "Missing fixture data: ${FIXTURE_ROOT}/centerlines" >&2
  echo "Copy or create tmp/issue121_fixture before running." >&2
  exit 2
fi

OUT_ROOT="${REPO_ROOT}/experiments/issue121_fixture_build"
PYTHONPATH=. python build_deepsets_dataset.py \
  --config configs/deepsets_issue121_fixture.yaml \
  --output-dir "${OUT_ROOT}" \
  --num-shards 1 \
  --shard-index 0

SUMMARY="${OUT_ROOT}/inclusion_rule_summary.csv"
echo "Wrote ${SUMMARY}"
if [[ -f "${SUMMARY}" ]]; then
  column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
fi
