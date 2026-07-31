$bibtex_use = 2;
$clean_ext = "nav snm";

# This poster is LuaTeX-only: the gemini theme loads Raleway and Lato through
# fontspec, which aborts under pdflatex with "fontspec requires either XeTeX or
# LuaTeX". latexmk defaults to pdflatex, so anything that invokes it without
# arguments -- a bare `latexmk`, `make`, or an editor's build-on-save -- would
# fail. Setting the engine here means every caller gets LuaTeX, not just
# build.sh.
$pdf_mode = 4;
$lualatex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';

# ...and `latexmk -pdf` sets pdf_mode=1 on the command line, which overrides the
# line above and selects pdflatex again. Editor extensions commonly pass -pdf in
# their default recipe, so point the pdflatex slot at lualatex too. The engine
# for this directory is LuaTeX no matter how latexmk is invoked.
$pdflatex = 'lualatex -interaction=nonstopmode -synctex=1 %O %S';
