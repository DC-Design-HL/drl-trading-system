#!/bin/bash
# ============================================================
# Whale Behavior LSTM — Mac Training Setup
# For Apple Silicon (M1/M2/M3) with MPS acceleration
# ============================================================

set -e

echo "🐋 Whale Behavior Training — Mac Setup"
echo "========================================"

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "❌ This script is for macOS only"
    exit 1
fi

# Check Python 3.10+
PYTHON=${PYTHON:-python3}
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
if [[ "$PY_VERSION" < "3.10" ]]; then
    echo "❌ Python 3.10+ required (found $PY_VERSION)"
    echo "   Install: brew install python@3.12"
    exit 1
fi
echo "✅ Python $PY_VERSION"

# Create venv if not exists
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies (minimal — only what training needs)
echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet \
    torch \
    numpy \
    pandas

# Verify MPS (Metal Performance Shaders) is available
$PYTHON -c "
import torch
print(f'PyTorch {torch.__version__}')
if torch.backends.mps.is_available():
    print('✅ MPS (Apple GPU) available — training will use GPU')
    # Quick smoke test
    x = torch.randn(2, 2, device='mps')
    print(f'   MPS tensor test: OK ({x.device})')
else:
    print('⚠️  MPS not available — will use CPU (still fast with 32GB RAM)')
"

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo ""
echo "To train:"
echo "  source .venv/bin/activate"
echo "  python train_whale_behavior.py --batch-size 256 --epochs 100"
echo ""
echo "Recommended experiments:"
echo "  # Default (fast, good baseline)"
echo "  python train_whale_behavior.py --batch-size 256 --epochs 100"
echo ""
echo "  # Longer sequences (if model supports it)"
echo "  python train_whale_behavior.py --batch-size 128 --epochs 100 --seq-length 50"
echo ""
echo "  # Lower learning rate for stability"
echo "  python train_whale_behavior.py --batch-size 256 --epochs 150 --lr 0.0003"
echo ""
echo "After training:"
echo "  1. Model saved to: data/whale_behavior/models/whale_behavior_lstm.pt"
echo "  2. Results in: data/whale_behavior/models/training_results.json"
echo "  3. Push to dev: git add -A && git commit -m 'trained model' && git push origin dev"
echo "  4. On server: git pull && systemctl restart drl-trade-alerter"
echo "========================================"
