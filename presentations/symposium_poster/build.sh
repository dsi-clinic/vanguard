#!/usr/bin/env bash
# Build the gemini/beamerposter version of the symposium poster.
#
#   ./build.sh          one build
#   ./build.sh watch    rebuild on every save, until Ctrl-C
#   ./build.sh png      build, then also write preview.png (VS Code previews
#                       PNGs natively, so this is the no-extension way to look
#                       at the poster)
#
# Same interface as ../symposium_poster_2026-08/build.sh, but this poster is
# LuaTeX-only: the gemini theme loads Raleway and Lato through fontspec, which
# aborts under pdflatex. Plain `latexmk` and `make` default to pdflatex and will
# fail with "fontspec requires either XeTeX or LuaTeX".
#
# There is no system TeX on this cluster; the paths below are the scratch
# install described in README.md. The conda-forge perl comes first because
# latexmk is a Perl script and the system perl's List::Util/Scalar::Util are
# unreadable.

set -euo pipefail

export PATH=/ess/scratch/scratch1/t-9svena/envs/perl/bin:/ess/scratch/scratch1/t-9svena/texlive/2026/bin/x86_64-linux:$PATH

# LuaTeX needs a writable font cache. Login-node environments do not always
# expose one through kpathsea, so keep a poster-specific cache in the local
# temporary filesystem rather than failing after producing an unusable PDF.
POSTER_TEX_CACHE="${TMPDIR:-/tmp}/vanguard-poster-texcache-${USER}"
mkdir -p "${POSTER_TEX_CACHE}"
export TEXMFCACHE="${POSTER_TEX_CACHE}"
export TEXMFVAR="${POSTER_TEX_CACHE}"
export TEXMFCONFIG="${POSTER_TEX_CACHE}"
export XDG_CACHE_HOME="${POSTER_TEX_CACHE}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# -g forces a full reprocess. Without it latexmk remembers a failed run in
# poster.fdb_latexmk and refuses to retry with "gave an error in previous
# invocation" even after the cause is fixed -- which happens every time a stray
# pdflatex run touches this directory.
LATEXMK=(latexmk -g -pdf -synctex=1 -pdflatex='lualatex -interaction=nonstopmode -synctex=1')

case "${1:-build}" in
  build)
    "${LATEXMK[@]}" poster.tex
    ;;
  watch)
    # -pvc polls for changes and rebuilds; -view=none because the cluster has no
    # PDF viewer and no X11 display.
    "${LATEXMK[@]}" -pvc -view=none poster.tex
    ;;
  png)
    "${LATEXMK[@]}" poster.tex
    /ess/scratch/scratch1/t-9svena/envs/tex/bin/pdftoppm -r 60 -png -singlefile poster.pdf preview
    echo "wrote preview.png"
    ;;
  *)
    echo "usage: $0 [build|watch|png]" >&2
    exit 1
    ;;
esac
