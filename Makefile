WRAP_DIR = wrapper
DOC_DIR = docs
NB_DIR = notebooks

SPHINX_DIR = sphinx_doc
SPHINX_GEN = $(SPHINX_DIR)/source/generated
SPHINX_BUILDGEN = $(SPHINX_DIR)/build/html/generated

#########################################
.PHONY: default all setup_wrapper doc

default: setup_wrapper
all: setup_wrapper doc

setup_wrapper:
	cd $(NB_DIR); ln -sf .. glow
	cd tests; ln -sf .. glow
	make -C $(WRAP_DIR)

#########################################
.PHONY: doc_only notebooks

doc: notebooks doc_only

doc_only: | $(DOC_DIR) $(SPHINX_GEN) $(SPHINX_BUILDGEN)
	cd $(SPHINX_DIR); ln -s .. glow; python3 store_defaults.py;
	python3 $(SPHINX_DIR)/include_notebooks.py
	make -C $(SPHINX_DIR) html
	cd $(DOC_DIR); ln -fs ../$(SPHINX_DIR)/build/html/index.html glow_doc.html

notebooks: | $(SPHINX_BUILDGEN)
	files=$$(ls $(NB_DIR) | grep -E "^examples.*\.ipynb") &&     \
	for f in $$files; do                                     \
		jupyter nbconvert --to html --execute --allow-errors $(NB_DIR)/$$f; \
	done
	mv $(NB_DIR)/*.html $(SPHINX_BUILDGEN)

$(DOC_DIR):
	mkdir -p $(DOC_DIR)

$(SPHINX_GEN):
	mkdir -p $(SPHINX_GEN)

$(SPHINX_BUILDGEN):
	mkdir -p $(SPHINX_BUILDGEN)

#########################################
.PHONY: clean clean_doc clean_nb

clean_doc:
	rm -f $(SPHINX_DIR)/glow
	make -C $(SPHINX_DIR) clean
	rm -rf $(SPHINX_DIR)/source/generated
	rm -f $(DOC_DIR)/glow_doc.html

clean_nb:
	@find notebooks -maxdepth 1 -name "examples*.ipynb" -exec\
		jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace --to notebook {} +

clean: clean_doc
	rm -rf *.egg-info
	rm -rf .eggs
	rm -rf build
	rm -rf dist
	rm -rf __pycache__
	rm -f $(NB_DIR)/glow
	rm -f tests/glow
	rm -f configure.log wrapper/.tmp_log
	-mv $(WRAP_DIR)/Makefile.bak $(WRAP_DIR)/Makefile
	make -C $(WRAP_DIR) clean
