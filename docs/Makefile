# Minimal Sphinx Makefile.
#
# Typical use:
#   make html        — build into _build/html
#   make clean       — remove _build
#   make livehtml    — auto-rebuild + serve at http://localhost:8000
#   make linkcheck   — validate every external URL

SPHINXBUILD   ?= sphinx-build
SPHINXOPTS    ?= -W --keep-going -n
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml linkcheck

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
