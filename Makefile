WRAP_DIR = wrapper
SPHINX_DIR = sphinx_doc
DOC_DIR = docs

#########################################

default: setup_wrapper
all: setup_wrapper doc

.PHONY: setup_wrapper
setup_wrapper:
	make -C $(WRAP_DIR)
.PHONY: doc

doc: | $(DOC_DIR) $(SPHINX_DIR)/source/generated
	cd $(SPHINX_DIR); python3 store_defaults.py
	make -C $(SPHINX_DIR) html
	cd $(DOC_DIR); ln -fs ../$(SPHINX_DIR)/build/html/index.html glow_doc.html
	pwd

$(DOC_DIR):
	mkdir -p $(DOC_DIR)

$(SPHINX_DIR)/source/generated:
	mkdir -p $(SPHINX_DIR)/source/generated


#########################################

.PHONY: clean
clean:
	rm -rf __pycache__
	make -C $(WRAP_DIR) clean
	make -C $(SPHINX_DIR) clean
	mkdir -p $(SPHINX_DIR)/build/html
	rm -f $(SPHINX_DIR)/source/generated/*
	rm -f $(DOC_DIR)/glow_doc.html
