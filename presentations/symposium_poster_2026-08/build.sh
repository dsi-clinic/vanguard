#!/usr/bin/env bash
# Build the symposium poster.
#
#   ./build.sh          one build (latexmk runs pdflatex as many times as needed)
#   ./build.sh watch    rebuild on every save, until Ctrl-C
#   ./build.sh png      build, then also write preview.png (VS Code previews PNGs
#                       natively, so this is the no-extension way to see the poster)
#
# There is no system TeX on this cluster; both paths below are the scratch install
# described in README.md. The conda-forge perl comes first because latexmk is a Perl
# script and the system perl is missing List::Util and Scalar::Util.

set -euo pipefail

export PATH=/ess/scratch/scratch1/t-9svena/envs/perl/bin:/ess/scratch/scratch1/t-9svena/texlive/2026/bin/x86_64-linux:$PATH

cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:-build}" in
  build)
    latexmk -pdf -synctex=1 -interaction=nonstopmode poster.tex
    ;;
  watch)
    # -pvc polls for changes and rebuilds; -view=none because the cluster has no
    # PDF viewer and no X11 display.
    latexmk -pdf -synctex=1 -interaction=nonstopmode -pvc -view=none poster.tex
    ;;
  png)
    latexmk -pdf -synctex=1 -interaction=nonstopmode poster.tex
    /ess/scratch/scratch1/t-9svena/envs/tex/bin/pdftoppm -r 60 -png -singlefile poster.pdf preview
    echo "wrote preview.png"
    ;;
  *)
    echo "usage: $0 [build|watch|png]" >&2
    exit 1
    ;;
esac
