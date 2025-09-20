#!/bin/bash

# =============================================================================
# DXNN-OCR Benchmark Environment Setup and Execution Script
# Mirroring PP-OCRv5-Cpp-Baseline methodology for compatibility
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/output/json" "$PROJECT_ROOT/output/vis"
mkdir -p "$PROJECT_ROOT/images"

# Logging function - matches PP-OCRv5 format
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/dxnn_benchmark_${TIMESTAMP}.log"
}

# Error handling - exit on critical errors only
set -e
trap 'log "ERROR: Script failed at line $LINENO"' ERR

log "=== Starting DXNN-OCR Benchmark Environment Setup ==="
log "Project Root: $PROJECT_ROOT"
log "Log Directory: $LOG_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Basic environment checks
log "=== Environment Prerequisites Check ==="

# Check Python
if ! command_exists python3; then
    log "ERROR: python3 is not installed. Please install python3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
if [[ $(echo -e "$PYTHON_VERSION\n3.11" | sort -V | head -n 1) != "3.11" ]]; then
    log "WARNING: Python version is $PYTHON_VERSION, recommended 3.11+"
fi
log "✓ Python: $(python3 --version)"

# Check conda
if ! command_exists conda; then
    log "ERROR: conda is not installed. Please install conda first."
    exit 1
fi
log "✓ Conda: $(conda --version)"

# Check essential tools
for tool in wget unzip; do
    if ! command_exists $tool; then
        log "ERROR: $tool is not installed. Please install: sudo apt install $tool"
        exit 1
    fi
done
log "✓ Essential tools: wget, unzip"

# =============================================================================
# Conda Environment Setup
# =============================================================================

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    log "ERROR: Conda is not installed or not in PATH"
    exit 1
fi
log "✓ Conda found: $(which conda)"

# Initialize conda for bash
log "Initializing conda for current shell..."
eval "$(conda shell.bash hook)"

# Define environment name following PP-OCRv5 pattern
ENV_NAME="PaddleOCR-deepx"
log "Target environment: $ENV_NAME"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    log "✓ Environment '$ENV_NAME' already exists"
else
    log "Creating new conda environment: $ENV_NAME"
    
    # Create environment with Python 3.11 (matching DXNN requirements)
    conda create -n "$ENV_NAME" python=3.11 -y
    
    if [[ $? -eq 0 ]]; then
        log "✓ Environment '$ENV_NAME' created successfully"
    else
        log "ERROR: Failed to create environment '$ENV_NAME'"
        exit 1
    fi
fi

# Activate the environment
log "Activating conda environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    log "ERROR: Failed to activate $ENV_NAME environment"
    log "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi
log "✓ Conda environment activated: $CONDA_DEFAULT_ENV"

# Set library paths for runtime (following PP-OCRv5 pattern)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
log "✓ Library paths configured"

# =============================================================================
# Dependencies Installation
# =============================================================================

log "=== Installing Dependencies ==="

# Check if requirements.txt exists
if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
    log "ERROR: requirements.txt not found in project root"
    exit 1
fi

# Install Python dependencies
log "Installing Python packages from requirements.txt..."
pip install -r "$PROJECT_ROOT/requirements.txt" 2>&1 | tee -a "$LOG_DIR/pip_install_${TIMESTAMP}.log"

if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    log "✓ Python dependencies installed successfully"
else
    log "ERROR: Failed to install Python dependencies"
    exit 1
fi

# Verify key packages are installed
REQUIRED_PACKAGES=("paddleocr" "opencv-python" "numpy" "Pillow")
log "Verifying required packages..."
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import ${pkg//-/_}" 2>/dev/null; then
        log "✓ Package verified: $pkg"
    else
        log "⚠ Warning: Package not found or not importable: $pkg"
    fi
done

# =============================================================================
# Dataset Setup (following PP-OCRv5 methodology)
# =============================================================================

log "=== Setting up XFUND Dataset ==="

setup_xfund_dataset() {
    local dataset_dir="$PROJECT_ROOT/images/xfund"
    
    mkdir -p "$dataset_dir"
    cd "$dataset_dir"
    
    # Download XFUND validation dataset (matching PP-OCRv5 approach)
    local xfund_base_url="https://github.com/doc-analysis/XFUND/releases/download/v1.0"
    
    if [[ ! -f "zh.val.zip" ]]; then
        log "Downloading XFUND validation dataset..."
        wget -q --show-progress "$xfund_base_url/zh.val.zip" || {
            log "ERROR: Failed to download zh.val.zip"
            return 1
        }
        log "✓ Downloaded zh.val.zip"
    else
        log "✓ zh.val.zip already exists"
    fi
    
    if [[ ! -f "zh.val.json" ]]; then
        log "Downloading XFUND validation annotations..."
        wget -q --show-progress "$xfund_base_url/zh.val.json" || {
            log "ERROR: Failed to download zh.val.json"
            return 1
        }
        log "✓ Downloaded zh.val.json"
    else
        log "✓ zh.val.json already exists"
    fi
    
    # Extract images if not already extracted
    if [[ ! -d "images" ]] || [[ $(find images -name "*.jpg" | wc -l) -lt 50 ]]; then
        log "Extracting XFUND validation images..."
        unzip -q zh.val.zip
        if [[ -d "zh_val" ]]; then
            mv zh_val images
        fi
        log "✓ XFUND images extracted"
    else
        log "✓ XFUND images already extracted"
    fi
    
    # Verify dataset
    local image_count=$(find images -name "*.jpg" | wc -l)
    log "✓ XFUND dataset ready: $image_count images found"
    
    cd "$PROJECT_ROOT"
}

# Setup dataset
if ! setup_xfund_dataset; then
    log "ERROR: Failed to setup XFUND dataset"
    exit 1
fi

# =============================================================================
# Environment Verification
# =============================================================================

log "=== Environment Verification ==="

# Verify Python installation
PYTHON_VERSION=$(python --version 2>&1)
log "✓ Python version: $PYTHON_VERSION"

# =============================================================================
# Benchmark Execution
# =============================================================================

log "=== Starting DXNN-OCR Benchmark ==="

# Function to run benchmark (following PP-OCRv5 pattern)
run_benchmark() {
    log "Executing DXNN benchmark with XFUND dataset..."
    
    # Prepare benchmark command
    local benchmark_cmd="python -u scripts/dxnn_benchmark.py"
    benchmark_cmd+=" --directory images/xfund"
    benchmark_cmd+=" --ground-truth images/xfund/zh.val.json"
    benchmark_cmd+=" --output output"
    benchmark_cmd+=" --runs 3"
    benchmark_cmd+=" --save-individual"
    
    log "Benchmark command: $benchmark_cmd"
    
    # Run benchmark with logging
    $benchmark_cmd 2>&1 | tee -a "$LOG_DIR/benchmark_execution_${TIMESTAMP}.log"
    
    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        log "✓ Benchmark completed successfully"
        
        # Display results summary
        if [[ -f "output/benchmark_summary.json" ]]; then
            log "=== Benchmark Results Summary ==="
            python -c "
import json
try:
    with open('output/benchmark_summary.json', 'r') as f:
        results = json.load(f)
    print(f\"Images processed: {results.get('total_images', 'N/A')}\")
    print(f\"Average inference time: {results.get('average_inference_ms', 'N/A'):.2f} ms\")
    print(f\"Average FPS: {results.get('average_fps', 'N/A'):.2f}\")
    if 'accuracy_metrics' in results:
        acc = results['accuracy_metrics']
        print(f\"Character accuracy: {acc.get('character_accuracy', 'N/A'):.2%}\")
except Exception as e:
    print(f'Could not display results summary: {e}')
            "
        fi
        
        # List generated reports
        log "Generated reports:"
        find output -name "*.md" -o -name "*.json" -o -name "*.csv" | while read file; do
            log "  - $(basename "$file")"
        done
        
    else
        log "ERROR: Benchmark execution failed with exit code $exit_code"
        return 1
    fi
}

# Execute benchmark
if ! run_benchmark; then
    log "ERROR: Benchmark execution failed"
    exit 1
fi

# =============================================================================
# Cleanup and Summary
# =============================================================================

log "=== Benchmark Completion Summary ==="
log "✓ Environment: $CONDA_DEFAULT_ENV"
log "✓ Dataset: XFUND validation set"
log "✓ Results: output/"
log "✓ Logs: $LOG_DIR/"

END_TIME=$(date)
log "Benchmark completed at: $END_TIME"
log "Total execution log: $LOG_DIR/dxnn_benchmark_${TIMESTAMP}.log"

echo ""
echo "=================================================="
echo "DXNN-OCR Benchmark Setup and Execution Complete!"
echo "=================================================="
echo ""
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Results directory: output/"
echo "Log files: $LOG_DIR/"
echo ""
echo "To re-run benchmark only:"
echo "  conda activate $ENV_NAME"
echo "  python -u scripts/dxnn_benchmark.py --dataset_path images/xfund --ground_truth images/xfund/zh.val.json"
echo ""