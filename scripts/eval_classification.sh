#!/bin/bash
# =============================================================================
# Evaluate TextME on classification tasks
#
# Usage:
#   ./eval_classification.sh [MODALITY] [DATASET]
#
# Examples:
#   ./eval_classification.sh 3d modelnet40
#   ./eval_classification.sh audio esc50
#   ./eval_classification.sh xray rsna
# =============================================================================

MODALITY=${1:-"3d"}
DATASET=${2:-"modelnet40"}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
OFFSET_DIR=${OFFSET_DIR:-"./offsets"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}
PIVOT_MODEL=${PIVOT_MODEL:-"qwen3_embed_4b"}
OUT_DIM=${OUT_DIM:-2560}

case $MODALITY in
    "3d"|"point")
        python evaluate.py \
            --val_p_cls_data $DATASET \
            --source_model_name languagebind \
            --target_model_name uni3d \
            --pivot_model_name $PIVOT_MODEL \
            --out_dim $OUT_DIM \
            --use_offset \
            --use_projection \
            --load_projection_checkpoint \
            --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
            --uni3d_proj_checkpoint $CHECKPOINT_DIR/uni3d_${PIVOT_MODEL}_objaverse/best.pt \
            --logs $OUTPUT_DIR \
            --run_name eval_p_cls_$DATASET
        ;;
    "audio")
        python evaluate.py \
            --val_a_cls_data $DATASET \
            --source_model_name languagebind \
            --target_model_name clap \
            --pivot_model_name $PIVOT_MODEL \
            --out_dim $OUT_DIM \
            --use_offset \
            --use_projection \
            --load_projection_checkpoint \
            --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
            --clap_proj_checkpoint $CHECKPOINT_DIR/clap_${PIVOT_MODEL}_audiocaps/best.pt \
            --logs $OUTPUT_DIR \
            --run_name eval_a_cls_$DATASET
        ;;
    "xray"|"xray")
        python evaluate.py \
            --val_x_cls_data $DATASET \
            --source_model_name languagebind \
            --target_model_name cxr_clip \
            --pivot_model_name $PIVOT_MODEL \
            --out_dim $OUT_DIM \
            --use_offset \
            --use_projection \
            --load_projection_checkpoint \
            --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
            --cxr_clip_proj_checkpoint $CHECKPOINT_DIR/cxr_clip_${PIVOT_MODEL}_chestxray/best.pt \
            --batch_size_val 4 \
            --logs $OUTPUT_DIR \
            --run_name eval_x_cls_$DATASET
        ;;
    "image")
        python evaluate.py \
            --val_i_cls_data $DATASET \
            --source_model_name languagebind \
            --target_model_name clip \
            --pivot_model_name $PIVOT_MODEL \
            --out_dim $OUT_DIM \
            --use_offset \
            --use_projection \
            --load_projection_checkpoint \
            --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
            --clip_proj_checkpoint $CHECKPOINT_DIR/clip_${PIVOT_MODEL}_coco/best.pt \
            --logs $OUTPUT_DIR \
            --run_name eval_i_cls_$DATASET
        ;;
    *)
        echo "Unknown modality: $MODALITY"
        echo "Available: 3d, audio, xray, image"
        exit 1
        ;;
esac
