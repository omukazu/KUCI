SRC_DIR := <abs_path_to_cbeps>
TGT_DIR := <abs_path_to_candidates>

USERNAME := $(shell whoami)
TMP_DIR := /tmp/$(USERNAME)/cbep2candidate

SRC_EXT := .cbep.tsv
TGT_EXT := .candidate.npy
SOURCES := $(shell find $(SRC_DIR)/ -name "*$(SRC_EXT)")
TARGETS := $(patsubst $(SRC_DIR)/%$(SRC_EXT),$(TGT_DIR)/%$(TGT_EXT),$(SOURCES))

PREFIX := nice -n 10
PYTHON := <abs_path_to_venv>/bin/python
CONFIG := <abs_path_to_config>/config.json
RUNNER := $(PREFIX) $(PYTHON) <abs_path_to_script>/select_candidates.py $(CONFIG)

all: $(TARGETS)

$(TARGETS): $(TGT_DIR)/%$(TGT_EXT): $(SRC_DIR)/%$(SRC_EXT)
	mkdir -p $(TMP_DIR) && \
	cp $< $(TMP_DIR)/$(notdir $<) && \
	$(RUNNER) $(TMP_DIR)/$(notdir $<) $(TMP_DIR)/$(basename $(basename $(notdir $<)))$(TGT_EXT) && \
	gzip $(TMP_DIR)/$(basename $(basename $(notdir $<)))$(TGT_EXT) && \
	mv $(TMP_DIR)/$(basename $(basename $(notdir $<)))$(TGT_EXT).gz $@.gz && \
	rm -f $(TMP_DIR)/$(notdir $<)
