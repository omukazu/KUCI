SRC_DIR := <abs_path_to_event_pairs>
TGT_DIR := <abs_path_to_filtered>

USERNAME := $(shell whoami)
TMP_DIR := /tmp/$(USERNAME)/event_pair2filtered

SRC_EXT := .event_pair.gz
TGT_EXT := .filtered.gz
SOURCES := $(shell find $(SRC_DIR)/ -name "*$(SRC_EXT)")
TARGETS := $(patsubst $(SRC_DIR)/%$(SRC_EXT),$(TGT_DIR)/%$(TGT_EXT),$(SOURCES))

PREFIX := nice -n 10
PYTHON := <abs_path_to_venv>/bin/python
RUNNER := $(PREFIX) $(PYTHON) <abs_path_to_script>/filter_by_conditions.py

all: $(TARGETS)

$(TARGETS): $(TGT_DIR)/%$(TGT_EXT): $(SRC_DIR)/%$(SRC_EXT)
	mkdir -p $(TMP_DIR) && mkdir -p $(dir $@) && \
	cp $< $(TMP_DIR)/$(notdir $<) && \
	$(RUNNER) $(TMP_DIR)/$(notdir $<) $(TMP_DIR)/$(basename $(basename $(notdir $<))).filtered && \
	gzip $(TMP_DIR)/$(basename $(basename $(notdir $<))).filtered && \
	mv $(TMP_DIR)/$(basename $(basename $(notdir $<)))$(TGT_EXT) $@ && \
    mv $(TMP_DIR)/$(basename $(basename $(notdir $<))).filtered.count $(patsubst %$(TGT_EXT),%.filtered.count,$@) && \
	rm -f $(TMP_DIR)/$(notdir $<)
