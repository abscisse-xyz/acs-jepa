#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ROOT_CONFIG_DIR="${SCRIPT_DIR}/configs/tuning"
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/default.yaml"
DATASET_DIR="${SCRIPT_DIR}/dataset"
DEVICE="cuda"
SEED="1"
CONFIG_NAME=""
BASE_OUTPUT_DIR=""

usage() {
    echo "Usage: $0 --config-name NAME [options]"
    echo
    echo "Options:"
    echo "  --config-name NAME       Required tuning config directory name."
    echo "  --dataset-dir PATH       Dataset directory. Default: ${SCRIPT_DIR}/dataset"
    echo "  --device DEVICE          Training device. Default: cpu"
    echo "  --seed SEED              Training seed. Default: 1"
    echo "  --output-dir PATH        Base output directory. Default: ${SCRIPT_DIR}/out/NAME"
    echo "  --default-config PATH    Base config file. Default: ${SCRIPT_DIR}/configs/default.yaml"
    echo "  --root-config-dir PATH   Tuning config root. Default: ${SCRIPT_DIR}/configs/tuning"
    echo "  -h, --help               Show this help."
}

require_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "Missing value for $1"
        exit 2
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config-name)
            require_value "$1" "${2:-}"
            CONFIG_NAME="$2"
            shift 2
            ;;
        --config-name=*)
            CONFIG_NAME="${1#*=}"
            shift
            ;;
        --dataset-dir)
            require_value "$1" "${2:-}"
            DATASET_DIR="$2"
            shift 2
            ;;
        --dataset-dir=*)
            DATASET_DIR="${1#*=}"
            shift
            ;;
        --device)
            require_value "$1" "${2:-}"
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --seed)
            require_value "$1" "${2:-}"
            SEED="$2"
            shift 2
            ;;
        --seed=*)
            SEED="${1#*=}"
            shift
            ;;
        --output-dir)
            require_value "$1" "${2:-}"
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            BASE_OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --default-config)
            require_value "$1" "${2:-}"
            DEFAULT_CONFIG="$2"
            shift 2
            ;;
        --default-config=*)
            DEFAULT_CONFIG="${1#*=}"
            shift
            ;;
        --root-config-dir)
            require_value "$1" "${2:-}"
            ROOT_CONFIG_DIR="$2"
            shift 2
            ;;
        --root-config-dir=*)
            ROOT_CONFIG_DIR="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

if [ -z "$CONFIG_NAME" ]; then
    echo "Missing required argument: --config-name"
    usage
    exit 2
fi

CONFIG_DIR="${ROOT_CONFIG_DIR}/${CONFIG_NAME}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${SCRIPT_DIR}/out/${CONFIG_NAME}}"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Config directory does not exist: $CONFIG_DIR"
    exit 1
fi

# Iterate over all YAML config files
for CONFIG_PATH in "$CONFIG_DIR"/*.yaml; do
    # Skip if no files match
    [ -e "$CONFIG_PATH" ] || continue

    # Extract config filename without extension
    RUN_CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)

    # Create distinct output directory per config
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$RUN_CONFIG_NAME"

    echo "Running config: $CONFIG_PATH"
    echo "Output dir: $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"

    uv run --package acs-jepa-cli acs-jepa train \
        "$DATASET_DIR" \
        --output "$OUTPUT_DIR" \
        --config "$DEFAULT_CONFIG" "$CONFIG_PATH" \
        --device "$DEVICE" \
        --seed "$SEED"

    echo "Finished: $RUN_CONFIG_NAME"
    echo "-----------------------------------"
done
