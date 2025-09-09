#!/bin/bash

cd "$(dirname "$0")"

DX_RT_PATH="$PWD/SDK/dx_rt"

REQUIRED_VERSION="3.11"
PYTHON_TO_USE=""

#1. Check if the 'python3' command is available and meets the version requirement.
if command -v python3 &> /dev/null; then
  PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n 1) == "$REQUIRED_VERSION" ]]; then
    PYTHON_TO_USE="python3"
  fi
fi

# 2. If 'python3' is not suitable, check for a dedicated 'python3.11' executable.
if [[ -z "$PYTHON_TO_USE" ]]; then
  if command -v python3.11 &> /dev/null; then
    PYTHON_TO_USE="python3.11"
  fi
fi

# 3. If no suitable Python executable is found, provide installation instructions and exit.
if [[ -z "$PYTHON_TO_USE" ]]; then
  if [ -z "$PYTHON_VERSION" ]; then
    echo "Error: python3 is not found. Please install python3." >&2
  else
    echo "Error: The installed python3 version is $PYTHON_VERSION, but this script requires version 3.11 or higher." >&2
  fi
  echo "Please install python3.11 and its venv package. For example:" >&2
  echo "" >&2
  echo "  sudo apt-get install python3.11 python3.11-venv" >&2
  echo "" >&2
  exit 1
fi

# Validate that the dx-runtime submodule exists.
# The submodule must be cloned and initialized to proceed.
if [ ! -d "$DX_RT_PATH" ]; then
  echo "Error: The directory '$DX_RT_PATH' does not exist." >&2
  echo "It seems that submodules were not cloned. Please run 'git submodule update --init --recursive' to get the submodules." >&2
  exit 1
fi

# Set up a new virtual environment using the determined Python executable.
# This ensures that project dependencies are isolated and managed correctly.
echo "Using $PYTHON_TO_USE to set up the virtual environment."
$PYTHON_TO_USE -m venv venv
source venv/bin/activate

# Install dependencies from the requirements file into the virtual environment.
python -m pip install -r ./requirements.txt

CONFIG_FILE="$DX_RT_PATH/cmake/dxrt.cfg.cmake"
if [ -f "$CONFIG_FILE" ]; then
  # Check if the first line contains the option set to OFF
  if head -n 1 "$CONFIG_FILE" | grep -q 'option(USE_ORT "Use ONNX Runtime" OFF)'; then
    echo "Found USE_ORT option set to OFF. Changing to ON..."
    # Use sed to replace OFF with ON only on the first line
    sed -i '1s/OFF/ON/' "$CONFIG_FILE"
    echo "Successfully enabled USE_ORT option in $CONFIG_FILE"
  else
    echo "USE_ORT option is already ON or not found in the expected format."
  fi
else
  echo "Warning: Configuration file $CONFIG_FILE not found. Skipping USE_ORT check."
fi

# Proceed with the build process for the dx-runtime component.
cd "$DX_RT_PATH"
./build.sh --clean
