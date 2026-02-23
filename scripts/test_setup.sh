#!/bin/bash
# =============================================================================
# Quick sanity check for TextME setup
#
# Usage:
#   ./test_setup.sh
#
# Tests:
#   1. Python imports
#   2. Encoder instantiation
#   3. Projector creation
#   4. Script syntax validation
# =============================================================================

set -e

echo "=============================================="
echo "TextME Setup Verification"
echo "=============================================="

# Test 1: Python imports
echo ""
echo "[1/4] Testing Python imports..."
python -c "
from textme import (
    ProjectionHead,
    HardNegativeContrastiveLoss,
    build_encoder,
    process_embeddings,
    ENCODER_DIM
)
print('  ✓ Core imports successful')
print(f'  ✓ Supported encoders: {list(ENCODER_DIM.keys())}')
"

# Test 2: Encoder instantiation (without loading weights)
echo ""
echo "[2/4] Testing encoder factory..."
python -c "
from textme.models.encoders import ENCODER_DIM, generate_offset_config

# Test offset config generation
for encoder in ENCODER_DIM.keys():
    config = generate_offset_config(encoder, 'test_dataset', 5000, './offsets')
    assert 'text' in config, f'Missing text offset for {encoder}'
    print(f'  ✓ {encoder}: offset config OK (dim={ENCODER_DIM[encoder]})')
"

# Test 3: Projector creation
echo ""
echo "[3/4] Testing projector creation..."
python -c "
import torch
from textme import ProjectionHead, ENCODER_DIM

for encoder, dim in ENCODER_DIM.items():
    proj = ProjectionHead(in_dim=dim, proj_dim=dim*2, out_dim=2560)
    dummy_input = torch.randn(2, dim)
    output = proj(dummy_input)
    assert output.shape == (2, 2560), f'Wrong output shape for {encoder}'
    print(f'  ✓ {encoder}: ProjectionHead({dim} → 2560) OK')
"

# Test 4: Shell script syntax
echo ""
echo "[4/4] Validating shell scripts..."
SCRIPT_DIR="$(dirname "$0")"
for script in compute_offsets.sh train.sh evaluate.sh eval_retrieval.sh eval_classification.sh; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        bash -n "$SCRIPT_DIR/$script" && echo "  ✓ $script: syntax OK"
    fi
done

echo ""
echo "=============================================="
echo "All checks passed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Set environment variables (DATA_ROOT, RAW_DATA_ROOT, EMBED_DIR)"
echo "  2. Run: ./scripts/compute_offsets.sh clip"
echo "  3. Run: ./scripts/train.sh clip"
echo ""
