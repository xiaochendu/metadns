#!/bin/bash
# Intelligent conda environment installation script for MDNS
# Uses system modules to avoid compiling common libraries from scratch

set -e  # Exit on error

echo "=========================================="
echo "MDNS Environment Installation Script"
echo "=========================================="
echo ""

# Get environment name from environment.yml
ENV_NAME=$(grep -m1 '^name:' environment.yml | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Could not find environment name in environment.yml"
    exit 1
fi

# Set environment path
if [ -z "$HOME2" ]; then
    ENV_PATH="$HOME/conda_envs/$ENV_NAME"
    echo "HOME2 not set, using: $ENV_PATH"
else
    ENV_PATH="$HOME2/conda_envs/$ENV_NAME"
    echo "Using HOME2: $ENV_PATH"
fi

echo "Environment name: $ENV_NAME"
echo "Environment path: $ENV_PATH"
echo ""

# ============================================================================
# Step 1: Check and load conda module
# ============================================================================
echo "Step 1: Loading conda module..."
if ! command -v module &> /dev/null; then
    echo "WARNING: module command not found. Trying to source module initialization..."
    # Try common module initialization paths
    if [ -f /usr/share/modules/init/bash ]; then
        source /usr/share/modules/init/bash
    elif [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    else
        echo "ERROR: Could not initialize module system"
        exit 1
    fi
fi

# Load conda module
if module avail conda &> /dev/null; then
    module load conda
    echo "✓ Conda module loaded"
else
    echo "WARNING: conda module not available, assuming conda/mamba is in PATH"
fi

# ============================================================================
# Step 1.5: Set conda package cache to avoid home directory quota issues
# ============================================================================
echo ""
echo "Step 1.5: Configuring conda package cache location..."

# Determine best location for package cache (prioritize SCRATCH for speed)
if [ -n "$SCRATCH" ]; then
    CONDA_PKGS_DIR="$SCRATCH/.conda/pkgs"
    echo "  Using SCRATCH for package cache (faster): $CONDA_PKGS_DIR"
elif [ -d "/pscratch" ]; then
    # Perlmutter-style scratch
    first_letter=$(echo $USER | cut -c 1)
    CONDA_PKGS_DIR="/pscratch/sd/${first_letter}/${USER}/.conda/pkgs"
    echo "  Using pscratch for package cache (faster): $CONDA_PKGS_DIR"
elif [ -n "$HOME2" ]; then
    CONDA_PKGS_DIR="$HOME2/.conda/pkgs"
    echo "  Using HOME2 for package cache: $CONDA_PKGS_DIR"
else
    # Fallback to a temp location in /tmp (will be cleaned on reboot)
    CONDA_PKGS_DIR="/tmp/${USER}_conda_pkgs"
    echo "  WARNING: Using /tmp for package cache (temporary): $CONDA_PKGS_DIR"
fi

# Create directory if it doesn't exist
mkdir -p "$CONDA_PKGS_DIR"
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIR"
echo "  Conda package cache set to: $CONDA_PKGS_DIR"

# Clean up corrupted package if it exists
if [ -f "/global/homes/d/dux/.conda/pkgs/pytorch-2.5.1-py3.10_cuda12.4_cudnn9.1.0_0.tar.bz2" ]; then
    echo "  Removing corrupted package from old location..."
    rm -f "/global/homes/d/dux/.conda/pkgs/pytorch-2.5.1-py3.10_cuda12.4_cudnn9.1.0_0.tar.bz2" 2>/dev/null || true
fi

# ============================================================================
# Step 2: Check for CUDA modules (for PyTorch GPU support)
# ============================================================================
echo ""
echo "Step 2: Checking for CUDA modules..."
CUDA_MODULE=""
CUDA_VERSION=""

# Check for common CUDA module names
for cuda_name in cuda cudatoolkit cuda-toolkit; do
    if module avail "$cuda_name" &> /dev/null; then
        # Try to get a version (look for common versions)
        for version in 12.4 12.3 12.2 12.1 12.0 11.8 11.7; do
            if module avail "${cuda_name}/${version}" &> /dev/null; then
                CUDA_MODULE="${cuda_name}/${version}"
                CUDA_VERSION="$version"
                break
            fi
        done
        # If no version found, try without version
        if [ -z "$CUDA_MODULE" ]; then
            CUDA_MODULE="$cuda_name"
        fi
        break
    fi
done

if [ -n "$CUDA_MODULE" ]; then
    echo "✓ Found CUDA module: $CUDA_MODULE"
    module load "$CUDA_MODULE"
    if [ -n "$CUDA_PATH" ]; then
        export CUDA_HOME="$CUDA_PATH"
        echo "  CUDA_HOME set to: $CUDA_HOME"
    fi
else
    echo "⚠ No CUDA module found - PyTorch will install CPU-only or use conda's CUDA"
fi

# ============================================================================
# Step 3: Check for compiler modules
# ============================================================================
echo ""
echo "Step 3: Checking for compiler modules..."
COMPILER_MODULES=()

# Check for GCC
if module avail gcc &> /dev/null; then
    # Try to find a recent version
    for version in 12 11 10 9; do
        if module avail "gcc/${version}" &> /dev/null; then
            COMPILER_MODULES+=("gcc/${version}")
            echo "✓ Found GCC module: gcc/${version}"
            break
        fi
    done
    if [ ${#COMPILER_MODULES[@]} -eq 0 ] && module avail gcc/default &> /dev/null; then
        COMPILER_MODULES+=("gcc/default")
        echo "✓ Found GCC module: gcc/default"
    fi
fi

# Check for other compilers
for compiler in intel llvm clang; do
    if module avail "$compiler" &> /dev/null; then
        echo "✓ Found $compiler module (not loading by default)"
    fi
done

# Load compiler modules if found
for mod in "${COMPILER_MODULES[@]}"; do
    module load "$mod"
done

# ============================================================================
# Step 4: Set environment variables for build
# ============================================================================
echo ""
echo "Step 4: Setting build environment variables..."

# Set CUDA_HOME if CUDA module is loaded
if [ -z "$CUDA_HOME" ] && [ -n "$CUDA_PATH" ]; then
    export CUDA_HOME="$CUDA_PATH"
fi

# Set compiler flags if GCC is loaded
if [ -n "$GCC_HOME" ]; then
    export CC="$GCC_HOME/bin/gcc"
    export CXX="$GCC_HOME/bin/g++"
    echo "  CC set to: $CC"
    echo "  CXX set to: $CXX"
fi

# Disable build isolation for packages that might benefit from system libraries
export PIP_NO_BUILD_ISOLATION=1

echo "  Build environment configured"

# ============================================================================
# Step 5: Create conda environment
# ============================================================================
echo ""
echo "Step 5: Creating conda environment..."
echo "  This may take several minutes..."

# Check if mamba is available (faster than conda)
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "  Using mamba (faster)"
else
    CONDA_CMD="conda"
    echo "  Using conda"
fi

# Configure conda to use the new package cache location permanently
echo "  Configuring conda to use package cache: $CONDA_PKGS_DIR"
$CONDA_CMD config --set pkgs_dirs "$CONDA_PKGS_DIR" 2>/dev/null || echo "  (Could not set conda config, but CONDA_PKGS_DIRS env var is set)"

# Create environment
$CONDA_CMD env create \
    -p "$ENV_PATH" \
    -f environment.yml

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create conda environment"
    exit 1
fi

echo "✓ Environment created successfully"

# ============================================================================
# Step 6: Install PyTorch with system CUDA if available
# ============================================================================
echo ""
echo "Step 6: Verifying PyTorch installation..."

# Activate environment
source "$(dirname $($CONDA_CMD info --base))/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# Check if PyTorch was installed
if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch is installed"
    python -c "import torch; print(f'  PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        python -c "import torch; print(f'  CUDA version: {torch.version.cuda}')"
    fi
else
    echo "⚠ PyTorch not found in environment"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Environment: $ENV_NAME"
echo "Path: $ENV_PATH"
echo "Package cache: $CONDA_PKGS_DIR"
echo ""
echo "To activate the environment:"
echo "  module load conda"
if [ -n "$CUDA_MODULE" ]; then
    echo "  module load $CUDA_MODULE"
fi
echo "  conda activate $ENV_PATH"
echo ""
echo "Note: Conda package cache is set to avoid home directory quota issues."
echo "      If you want to clean up the old cache in ~/.conda/pkgs, you can run:"
echo "      rm -rf ~/.conda/pkgs/*"
echo ""
