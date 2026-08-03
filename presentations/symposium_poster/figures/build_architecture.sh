#!/usr/bin/env bash
# Build the TikZ architecture figure to architecture.pdf.
#
#   ./build_architecture.sh          one build
#   ./build_architecture.sh png      build, then also write architecture.png
#
# LuaTeX-only for the same reason as the poster itself (fontspec). Same scratch
# TeX install and writable-cache dance as ../build.sh -- see that script for why.
#
# The real-data inputs under architecture_data/ are NOT rebuilt here: they come
# from a Slurm job that loads a 4D DCE series. Regenerate them with
#   sbatch gnn/slurm/submit_poster_architecture_data.slurm
# from the repo root, then rerun this.

set -euo pipefail

export PATH=/ess/scratch/scratch1/t-9svena/texlive/2026/bin/x86_64-linux:$PATH

FIG_TEX_CACHE="${TMPDIR:-/tmp}/vanguard-poster-texcache-${USER}"
mkdir -p "${FIG_TEX_CACHE}"
export TEXMFCACHE="${FIG_TEX_CACHE}"
export TEXMFVAR="${FIG_TEX_CACHE}"
export TEXMFCONFIG="${FIG_TEX_CACHE}"
export XDG_CACHE_HOME="${FIG_TEX_CACHE}"

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f architecture_data/graph.tex ]]; then
  echo "architecture_data/graph.tex missing -- run the Slurm extractor first" >&2
  exit 1
fi

lualatex -interaction=nonstopmode -halt-on-error architecture.tex

if [[ "${1:-}" == "png" ]]; then
  /ess/scratch/scratch1/t-9svena/envs/tex/bin/pdftoppm \
    -r 60 -png -singlefile architecture.pdf architecture
  echo "wrote architecture.png"
fi
