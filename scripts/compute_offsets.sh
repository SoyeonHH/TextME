#!/bin/bash
# =============================================================================
# Compute offsets for TextME encoders
#
# Usage:
#   ./compute_offsets.sh [ENCODER]    # Compute offset for specific encoder
#   ./compute_offsets.sh all          # Compute offsets for all encoders
#   ./compute_offsets.sh              # Same as 'all'
#
# Examples:
#   ./compute_offsets.sh clip
#   ./compute_offsets.sh uni3d
#   ./compute_offsets.sh moleculestm
#   ./compute_offsets.sh all
#
# Supported encoders:
#   clip, clap, uni3d, cxr_clip, moleculestm, remoteclip, languagebind
# =============================================================================

set -e

ENCODER=${1:-"all"}

# Configuration
DATA_ROOT=${DATA_ROOT:-"/path/to/pretraining_captions"}
RAW_DATA_ROOT=${RAW_DATA_ROOT:-"/path/to/raw_data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./offsets"}
NUM_SAMPLES=${NUM_SAMPLES:-5000}
BATCH_SIZE=${BATCH_SIZE:-64}
DEVICE=${DEVICE:-"cuda"}

# Encoder configurations: encoder_name -> "dataset dim"
declare -A ENCODER_CONFIG=(
    ["clip"]="coco 1024"
    ["clap"]="audiocaps 512"
    ["uni3d"]="objaverse 1024"
    ["cxr_clip"]="chestxray 512"
    ["moleculestm"]="pubchem 256"
    ["remoteclip"]="remoteclip_ret3 768"
    ["languagebind"]="coco 768"
)

compute_offset() {
    local encoder=$1
    local config=${ENCODER_CONFIG[$encoder]}

    if [ -z "$config" ]; then
        echo "Error: Unknown encoder '$encoder'"
        echo "Available: ${!ENCODER_CONFIG[@]}"
        exit 1
    fi

    local dataset=$(echo $config | cut -d' ' -f1)
    local dim=$(echo $config | cut -d' ' -f2)

    echo ""
    echo "Computing offset for $encoder ($dataset, dim=$dim)..."
    python compute_offset.py \
        --offset_model $encoder \
        --dataset_name $dataset \
        --data_root $DATA_ROOT \
        --raw_data_root $RAW_DATA_ROOT/$dataset \
        --saving_path $OUTPUT_DIR/$NUM_SAMPLES/${encoder}_${dataset} \
        --offset_num $NUM_SAMPLES \
        --dim $dim \
        --batch_size $BATCH_SIZE \
        --device $DEVICE
}

echo "=============================================="
echo "TextME Offset Computation"
echo "=============================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Num samples: $NUM_SAMPLES"
echo "=============================================="

if [ "$ENCODER" = "all" ]; then
    echo "Computing offsets for all encoders..."
    for enc in clip clap uni3d cxr_clip moleculestm remoteclip languagebind; do
        compute_offset $enc
    done
else
    compute_offset $ENCODER
fi

echo ""
echo "=============================================="
echo "Offset computation complete!"
echo "=============================================="
