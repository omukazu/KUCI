SRC_DIR := <abs_path_to_knp_parsing_results>
TGT_DIR := <abs_path_to_event_pairs>

USERNAME := $(shell whoami)
TMP_DIR := /tmp/$(USERNAME)/knp2event_pair

SRC_EXT := .knp.gz
TGT_EXT := .event_pair.gz
SOURCES := $(shell find $(SRC_DIR)/ -name "*$(SRC_EXT)")
TARGETS := $(patsubst $(SRC_DIR)/%$(SRC_EXT),$(TGT_DIR)/%$(TGT_EXT),$(SOURCES))

PREFIX := nice -n 10
PYTHON := <abs_path_to_venv>/bin/python
CORE_EVENTS := <abs_path_to_core_events>/core_events.txt
RUNNER := $(PREFIX) $(PYTHON) <abs_path_to_script>/extract_event_pairs.py --core-events $(CORE_EVENTS) --silent

all: $(TARGETS)

# $(basename $(basename $(notdir $<)))
# = $(basename $(basename $(notdir e.g. <path_to_knp_parsing_results>/0.knp.gz)))
# = $(basename $(basename 0.knp.gz))
# = $(basename 0.knp)
# = 0

$(TARGETS): $(TGT_DIR)/%$(TGT_EXT): $(SRC_DIR)/%$(SRC_EXT)
	mkdir -p $(TMP_DIR) && mkdir -p $(dir $@) && \
	cp $< $(TMP_DIR)/$(notdir $<) && \
	$(RUNNER) $(TMP_DIR)/$(notdir $<) $(TMP_DIR)/$(basename $(basename $(notdir $<))).event_pair && \
	gzip $(TMP_DIR)/$(basename $(basename $(notdir $<))).event_pair && \
	mv $(TMP_DIR)/$(basename $(basename $(notdir $<)))$(TGT_EXT) $@ && \
	rm -f $(TMP_DIR)/$(notdir $<)
