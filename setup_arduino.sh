#!/usr/bin/env bash
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
# Sets up a project-local Arduino CLI plus Conda activation hooks.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
echo "Project root: $PROJECT_ROOT"
TOOLS_DIR="$PROJECT_ROOT/tools"
BIN_DIR="$TOOLS_DIR/bin"
CONFIG_FILE="$TOOLS_DIR/arduino-cli.yaml"
ARDUINO_DATA_DIR="$TOOLS_DIR/arduino-data"
ARDUINO_DOWNLOADS_DIR="$TOOLS_DIR/arduino-downloads"
ARDUINO_USER_DIR="$TOOLS_DIR/arduino-user"
LIB_SUBMODULE="$PROJECT_ROOT/tools/arduino-user/libraries/Arduino_TensorFlowLite"

mkdir -p "$BIN_DIR"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to download Arduino CLI" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please 'conda activate' the target environment before running this script." >&2
  exit 1
fi

echo "Downloading Arduino CLI into $BIN_DIR ..."
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh \
  | BINDIR="$BIN_DIR" sh

mkdir -p "$TOOLS_DIR" "$ARDUINO_DATA_DIR" "$ARDUINO_DOWNLOADS_DIR" "$ARDUINO_USER_DIR"

echo "Writing Arduino CLI config at $CONFIG_FILE ..."
cat >"$CONFIG_FILE" <<EOF
board_manager:
  additional_urls: []
directories:
  data: "$ARDUINO_DATA_DIR"
  downloads: "$ARDUINO_DOWNLOADS_DIR"
  user: "$ARDUINO_USER_DIR"
EOF

echo "Updating core index and installing Arduino mbed cores (nano + portenta) ..."
"$BIN_DIR/arduino-cli" core update-index --config-file "$CONFIG_FILE"
"$BIN_DIR/arduino-cli" core install arduino:mbed_nano --config-file "$CONFIG_FILE"
"$BIN_DIR/arduino-cli" core install arduino:mbed_portenta --config-file "$CONFIG_FILE"
# Optional: esp32 core for the XIAO ESP32-S3 Sense backend (large download).
# Its prebuilt esp-tflite-micro + ESP-NN are used instead of Chirale on ESP32.
if [[ "${CREST_INSTALL_ESP32:-0}" == "1" ]]; then
  ESP32_INDEX_URL="https://espressif.github.io/arduino-esp32/package_esp32_index.json"
  "$BIN_DIR/arduino-cli" core update-index --config-file "$CONFIG_FILE" --additional-urls "$ESP32_INDEX_URL"
  "$BIN_DIR/arduino-cli" core install esp32:esp32@3.3.12 --config-file "$CONFIG_FILE" --additional-urls "$ESP32_INDEX_URL"
fi
echo "Installing required Arduino libraries ..."
"$BIN_DIR/arduino-cli" lib install "Adafruit INA228 Library" --config-file "$CONFIG_FILE"

echo "Copying Conda activation hooks into $CONDA_PREFIX ..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
escaped_root="${PROJECT_ROOT//\//\\/}"
sed "s#__PROJECT_ROOT__#$escaped_root#g" "$PROJECT_ROOT/env_setup/conda_activate_arduino.sh" \
  > "$CONDA_PREFIX/etc/conda/activate.d/arduino.sh"
sed "s#__PROJECT_ROOT__#$escaped_root#g" "$PROJECT_ROOT/env_setup/conda_deactivate_arduino.sh" \
  > "$CONDA_PREFIX/etc/conda/deactivate.d/arduino.sh"
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/arduino.sh" "$CONDA_PREFIX/etc/conda/deactivate.d/arduino.sh"

cat <<EOF
Arduino CLI bootstrap complete.
- Binary location: $BIN_DIR/arduino-cli
- Config file: $CONFIG_FILE
- Conda hooks installed under $CONDA_PREFIX/etc/conda
- Arduino library root: $PROJECT_ROOT/tools/arduino-user/libraries
  $(if [[ -d "$LIB_SUBMODULE" ]]; then echo "Chirale TensorFlow Lite library detected."; else echo "Chirale TensorFlow Lite repo missing; run 'git submodule update --init --recursive' to fetch tools/arduino-user/libraries/Arduino_TensorFlowLite."; fi)
Remember to add tools directories to .gitignore to keep binaries out of Git.
EOF
