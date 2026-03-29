#!/bin/bash
"""
Mac Championship Setup Script

One-command setup for the Championship DRL Training Pipeline on Mac M3 Pro.
Creates virtual environment, installs dependencies, verifies MPS GPU availability,
and provides example commands to run the training pipeline.

Usage:
    chmod +x scripts/mac_championship_setup.sh
    ./scripts/mac_championship_setup.sh
"""

set -e  # Exit on any error

echo "=================================================="
echo "  Championship DRL Training Pipeline - Mac Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is designed for macOS only${NC}"
    exit 1
fi

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    echo "Please install Python 3.8+ from https://www.python.org/ or using Homebrew:"
    echo "  brew install python@3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}Found Python ${PYTHON_VERSION}${NC}"

PYTHON_OK=$(python3 -c "import sys; print(1 if sys.version_info >= (3, 8) else 0)")
if [[ "$PYTHON_OK" != "1" ]]; then
    echo -e "${RED}Error: Python 3.8+ required (found ${PYTHON_VERSION})${NC}"
    exit 1
fi

# Check system resources
echo -e "${BLUE}Checking system resources...${NC}"
TOTAL_RAM_GB=$(sysctl -n hw.memsize | awk '{print int($0/1024/1024/1024)}')
echo -e "${GREEN}Total RAM: ${TOTAL_RAM_GB}GB${NC}"

if [[ ${TOTAL_RAM_GB} -lt 16 ]]; then
    echo -e "${YELLOW}Warning: Less than 16GB RAM detected. Training may be slower or fail.${NC}"
    echo -e "${YELLOW}Recommended: Use smaller batch sizes or train on fewer folds.${NC}"
fi

# Check for Apple Silicon
ARCH=$(uname -m)
echo -e "${GREEN}Architecture: ${ARCH}${NC}"

if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Not running on Apple Silicon. MPS acceleration unavailable.${NC}"
fi

# Create virtual environment
VENV_DIR="venv_championship"
echo -e "${BLUE}Creating virtual environment: ${VENV_DIR}${NC}"

if [[ -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo -e "${GREEN}Virtual environment created and activated${NC}"

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install PyTorch with MPS support
echo -e "${BLUE}Installing PyTorch with MPS support...${NC}"
if [[ "$ARCH" == "arm64" ]]; then
    # Install PyTorch with MPS support for Apple Silicon
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    # Install CPU-only PyTorch for Intel Macs
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install stable-baselines3 ecosystem
echo -e "${BLUE}Installing stable-baselines3 ecosystem...${NC}"
pip install stable-baselines3[extra]
pip install sb3-contrib

# Install data science dependencies
echo -e "${BLUE}Installing data science dependencies...${NC}"
pip install pandas numpy matplotlib seaborn
pip install gymnasium[all]

# Install TA-Lib (if available)
echo -e "${BLUE}Installing TA-Lib...${NC}"
if command -v brew &> /dev/null; then
    echo "Installing TA-Lib via Homebrew..."
    brew install ta-lib || echo -e "${YELLOW}Warning: Could not install TA-Lib via brew${NC}"
    pip install TA-Lib || echo -e "${YELLOW}Warning: Could not install TA-Lib Python wrapper${NC}"
else
    echo -e "${YELLOW}Warning: Homebrew not found. Skipping TA-Lib installation.${NC}"
    echo "To install TA-Lib manually:"
    echo "  1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  2. Install TA-Lib: brew install ta-lib"
    echo "  3. Install Python wrapper: pip install TA-Lib"
fi

# Install additional utilities
echo -e "${BLUE}Installing additional utilities...${NC}"
pip install psutil tqdm colorama pyyaml

# Install Jupyter for analysis
echo -e "${BLUE}Installing Jupyter for analysis...${NC}"
pip install jupyter ipykernel
python -m ipykernel install --user --name="$VENV_DIR" --display-name "Championship DRL"

# Create requirements.txt
echo -e "${BLUE}Creating requirements.txt...${NC}"
pip freeze > requirements.txt
echo -e "${GREEN}Requirements saved to requirements.txt${NC}"

# Test PyTorch MPS availability
echo -e "${BLUE}Testing PyTorch MPS availability...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✓ MPS GPU acceleration is available!')
    device = torch.device('mps')
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = torch.matmul(x, y)
    print(f'✓ MPS GPU test successful: {z.shape} tensor computed')
else:
    print('⚠ MPS GPU acceleration is not available (using CPU)')
"

# Test stable-baselines3 import
echo -e "${BLUE}Testing stable-baselines3 import...${NC}"
python3 -c "
import stable_baselines3 as sb3
from sb3_contrib import QRDQN
print(f'✓ stable-baselines3 version: {sb3.__version__}')
print('✓ QRDQN import successful')
print('✓ All dependencies installed correctly')
"

# Check available disk space
echo -e "${BLUE}Checking disk space...${NC}"
AVAILABLE_GB=$(df -H . | awk 'NR==2 {print $4}' | sed 's/G//')
echo -e "${GREEN}Available disk space: ${AVAILABLE_GB}${NC}"

if [[ ${AVAILABLE_GB%.*} -lt 50 ]]; then
    echo -e "${YELLOW}Warning: Less than 50GB free space. Training data and models require significant storage.${NC}"
fi

# Create data directories
echo -e "${BLUE}Creating data directories...${NC}"
mkdir -p data/historical
mkdir -p data/models/championship
mkdir -p logs/tensorboard
mkdir -p logs/eval

echo -e "${GREEN}Directory structure created${NC}"

# Display completion message and next steps
echo ""
echo "=================================================="
echo -e "${GREEN}  Setup Complete! 🎉${NC}"
echo "=================================================="
echo ""
echo -e "${BLUE}Virtual Environment:${NC} $VENV_DIR"
echo -e "${BLUE}Python Path:${NC} $(which python)"
echo -e "${BLUE}Pip Packages:${NC} $(pip list | wc -l) installed"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Download historical data:"
echo "   python download_historical_data.py --symbol BTCUSDT --interval 15m"
echo ""
echo "2. Test the environment:"
echo "   python -c \"from src.env.htf_env import HTFTradingEnv; print('✓ HTF Environment loads correctly')\""
echo ""
echo "3. Run a quick test training:"
echo "   python train_championship.py --phase1-steps 1000 --phase2-steps 500 --phase3-steps 500"
echo ""
echo "4. Run full championship training:"
echo "   python train_championship.py --symbol BTCUSDT --data-path data/historical/"
echo ""
echo "5. Monitor training with Tensorboard:"
echo "   tensorboard --logdir logs/tensorboard/"
echo ""
echo -e "${BLUE}Memory Recommendations for Mac M3 Pro (12GB available):${NC}"
echo "   - Batch size: 256 (default)"
echo "   - Reduce steps if memory issues: --phase1-steps 100000 --phase2-steps 75000"
echo "   - Monitor memory: Activity Monitor or htop"
echo ""
echo -e "${BLUE}Training Time Estimates:${NC}"
echo "   - Full pipeline: ~24-48 hours"
echo "   - Phase 1: ~8-12 hours"
echo "   - Phase 2: ~6-10 hours"
echo "   - Phase 3: ~4-8 hours"
echo ""
echo -e "${YELLOW}Pro Tips:${NC}"
echo "   - Use --resume to continue interrupted training"
echo "   - Start with fewer folds for testing"
echo "   - Keep Activity Monitor open to watch memory usage"
echo "   - Training will automatically use MPS GPU if available"
echo ""
echo "To activate this environment in the future:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo -e "${GREEN}Happy training! 🚀${NC}"
echo "=================================================="