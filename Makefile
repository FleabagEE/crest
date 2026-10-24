# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
.PHONY: help install test integration-test test-all start-gpu start-hil smoke env-create arduino-setup stm32-setup prepare-dataset prepare-audio-dataset rp2040-export rp2040-compile rp2040-bench rp2040-upload-stress clean

PYTHON ?= python
ENV ?= crest
OXIOD_ZIP ?= OxIOD.zip
URBANSOUND8K_ARGS ?=
PORT ?= /dev/ttyACM0
RP2040_CANDIDATE ?= models/rp2040_bench/dscnn_us8k_min_int8
RP2040_BENCH_ARGS ?=
RP2040_STRESS_ARGS ?=
PYTHON_LIB_DIR := $(shell $(PYTHON) -c 'import sys; print(sys.prefix + "/lib")')
TEST_ENV := LD_LIBRARY_PATH="$(PYTHON_LIB_DIR):$(LD_LIBRARY_PATH)"

help:
	@echo "Targets:"
	@echo "  install        Install editable package without deps (use conda for deps)"
	@echo "  test           Run the fast/default pytest suite in test/"
	@echo "  integration-test  Run slow integration tests in test/integration/"
	@echo "  test-all       Run both the fast suite and integration tests"
	@echo "  start-gpu      Run NAS client (GPU box)"
	@echo "  start-hil      Run HIL server (device host)"
	@echo "  smoke          Run a short NAS smoke test (override ARGS)"
	@echo "  env-create     Create conda env from environment.yml"
	@echo "  arduino-setup  Run Arduino CLI bootstrapper"
	@echo "  stm32-setup    Validate STM32CubeCLT paths and bootstrap STM32CubeN6 firmware"
	@echo "  prepare-dataset  Prepare OxIOD dataset (override OXIOD_ZIP=/path/to/OxIOD.zip)"
	@echo "  prepare-audio-dataset  Prepare UrbanSound8K log-mel cache (use URBANSOUND8K_ARGS=\"--fold-rotation\" for Phase 9 reports)"
	@echo "  rp2040-export  Export the smallest int8 DS-CNN as an RP2040 candidate (RP2040_CANDIDATE=...)"
	@echo "  rp2040-compile Compile-only RP2040 memory check (no board needed)"
	@echo "  rp2040-bench   Upload + time the RP2040 candidate (PORT=/dev/ttyACM0, RP2040_BENCH_ARGS=...)"
	@echo "  rp2040-upload-stress  Upload the RP2040 candidate 10 times and log each attempt (RP2040_STRESS_ARGS=...)"
	@echo "  clean          Remove Python cache/build artifacts"

install:
	pip install -e . --no-deps

test:
	$(TEST_ENV) $(PYTHON) -m pytest test/

integration-test:
	$(TEST_ENV) RUN_INTEGRATION_TESTS=1 $(PYTHON) -m pytest test/integration/

test-all:
	$(TEST_ENV) RUN_INTEGRATION_TESTS=1 $(PYTHON) -m pytest test/

start-gpu:
	$(PYTHON) src/nas_model_client.py $(ARGS)

start-hil:
	$(PYTHON) src/hil_server.py

smoke:
	$(PYTHON) src/nas_model_client.py --smoke-test 1 $(ARGS)

env-create:
	conda env create -f environment.yml -n $(ENV)

arduino-setup:
	bash ./setup_arduino.sh

stm32-setup:
	bash ./setup_stm32.sh

prepare-dataset:
	$(PYTHON) data/dataset_download_and_splits/oxiod/prepare_oxiod.py --zip-path $(OXIOD_ZIP)

prepare-audio-dataset:
	$(PYTHON) data/dataset_download_and_splits/urbansound8k/prepare_urbansound8k.py $(URBANSOUND8K_ARGS)

rp2040-export:
	$(PYTHON) analysis_scripts/rp2040_latency_bench/export_smallest_dscnn.py --output-dir $(RP2040_CANDIDATE)

rp2040-compile:
	$(PYTHON) analysis_scripts/rp2040_latency_bench/run_rp2040_latency_bench.py --candidate-dir $(RP2040_CANDIDATE) --compile-only $(RP2040_BENCH_ARGS)

rp2040-bench:
	$(PYTHON) analysis_scripts/rp2040_latency_bench/run_rp2040_latency_bench.py --candidate-dir $(RP2040_CANDIDATE) --port $(PORT) $(RP2040_BENCH_ARGS)

rp2040-upload-stress:
	$(PYTHON) analysis_scripts/rp2040_latency_bench/rp2040_upload_stress.py --candidate-dir $(RP2040_CANDIDATE) --port $(PORT) $(RP2040_STRESS_ARGS)

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
