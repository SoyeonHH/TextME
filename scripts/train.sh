#!/bin/bash
# =============================================================================
# Train TextME projections
#
# Usage:
#   ./train.sh [ENCODER]    # Train projection for specific encoder
#   ./train.sh all          # Train projections for all encoders
#   ./train.sh              # Same as 'all'
#
# Examples:
#   ./train.sh clip
#   ./train.sh uni3d
#   ./train.sh moleculestm
#   ./train.sh all
#
# Supported encoders:
#   clip, clap, uni3d, cxr_clip, moleculestm, remoteclip, languagebind
# =============================================================================

set -e

ENCODER=${1:-"all"}

# Configuration
DATA_ROOT=${DATA_ROOT:-"/path/to/pretraining_captions"}
EMBED_DIR=${EMBED_DIR:-"/path/to/qwen3_4B_embeds"}
OFFSET_DIR=${OFFSET_DIR:-"./offsets"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints"}
OFFSET_NUM=${OFFSET_NUM:-5000}

# Training parameters (Table 11 in paper)
BATCH_SIZE=${BATCH_SIZE:-512}
EPOCHS=${EPOCHS:-50}
LR=${LR:-5e-4}
TEMPERATURE=${TEMPERATURE:-0.07}
TOP_MARGIN=${TOP_MARGIN:-0.9}
BOTTOM_MARGIN=${BOTTOM_MARGIN:-0.1}
OUT_DIM=${OUT_DIM:-2560}
PIVOT_MODEL=${PIVOT_MODEL:-"qwen3_embed_4b"}

# Encoder configurations: encoder_name -> "dataset"
declare -A ENCODER_CONFIG=(
    ["clip"]="coco"
    ["clap"]="audiocaps"
    ["uni3d"]="objaverse"
    ["cxr_clip"]="chestxray"
    ["moleculestm"]="pubchem"
    ["remoteclip"]="remoteclip_ret3"
    ["languagebind"]="coco"
)

# Special configurations for specific encoders
declare -A ENCODER_LR=(
    ["cxr_clip"]="5e-3"
)
declare -A ENCODER_EPOCHS=(
    ["cxr_clip"]="100"
)
declare -A ENCODER_SKIP_SCHEDULER=(
    ["cxr_clip"]="true"
)

train_encoder() {
    local encoder=$1
    local dataset=${ENCODER_CONFIG[$encoder]}

    if [ -z "$dataset" ]; then
        echo "Error: Unknown encoder '$encoder'"
        echo "Available: ${!ENCODER_CONFIG[@]}"
        exit 1
    fi

    # Get encoder-specific parameters or use defaults
    local lr=${ENCODER_LR[$encoder]:-$LR}
    local epochs=${ENCODER_EPOCHS[$encoder]:-$EPOCHS}
    local skip_scheduler=${ENCODER_SKIP_SCHEDULER[$encoder]:-"false"}

    local scheduler_arg=""
    if [ "$skip_scheduler" = "true" ]; then
        scheduler_arg="--skip_scheduler"
    fi

    echo ""
    echo "Training $encoder projection ($dataset → $PIVOT_MODEL)..."
    python train.py \
        --model_name $encoder \
        --pivot_model_name $PIVOT_MODEL \
        --dataset_name $dataset \
        --data_root $DATA_ROOT \
        --embed_dir ${EMBED_DIR}_${dataset} \
        --use_offset \
        --use_projection \
        --offset_dir $OFFSET_DIR \
        --offset_num $OFFSET_NUM \
        --out_dim $OUT_DIM \
        --batch_size $BATCH_SIZE \
        --epochs $epochs \
        --lr $lr \
        --temperature $TEMPERATURE \
        --top_perc_margin $TOP_MARGIN \
        --bottom_perc_margin $BOTTOM_MARGIN \
        --checkpoint_dir $OUTPUT_DIR \
        --save_logs \
        --name "${encoder}_${PIVOT_MODEL}_${dataset}" \
        $scheduler_arg
}

echo "=============================================="
echo "TextME Projection Training"
echo "=============================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Pivot model: $PIVOT_MODEL (dim=$OUT_DIM)"
echo "=============================================="

if [ "$ENCODER" = "all" ]; then
    echo "Training projections for all encoders..."
    for enc in clip clap uni3d cxr_clip moleculestm remoteclip languagebind; do
        train_encoder $enc
    done
else
    train_encoder $ENCODER
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
