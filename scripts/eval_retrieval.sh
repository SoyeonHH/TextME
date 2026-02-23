#!/bin/bash
# =============================================================================
# Evaluate TextME on retrieval tasks
#
# Usage:
#   ./eval_retrieval.sh [MODALITY] [DATASET]
#
# Examples:
#   ./eval_retrieval.sh image coco
#   ./eval_retrieval.sh audio audiocaps
#   ./eval_retrieval.sh molecule drugbank
#   ./eval_retrieval.sh remote rsicd
# =============================================================================

MODALITY=${1:-"image"}
DATASET=${2:-"coco"}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
OFFSET_DIR=${OFFSET_DIR:-"./offsets"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}
PIVOT_MODEL=${PIVOT_MODEL:-"qwen3_embed_4b"}
OUT_DIM=${OUT_DIM:-2560}

case $MODALITY in
    "image")
        python evaluate.py \
            --val_il_ret_data $DATASET \
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
            --run_name eval_il_ret_$DATASET
        ;;
    "audio")
        python evaluate.py \
            --val_al_ret_data $DATASET \
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
            --run_name eval_al_ret_$DATASET
        ;;
    "molecule")
        python evaluate.py \
            --val_ml_ret_data drugbank_pharmacodynamics_removed_PubChem \
            --source_model_name languagebind \
            --target_model_name moleculestm \
            --pivot_model_name $PIVOT_MODEL \
            --out_dim $OUT_DIM \
            --use_offset \
            --use_projection \
            --load_projection_checkpoint \
            --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
            --moleculestm_proj_checkpoint $CHECKPOINT_DIR/moleculestm_${PIVOT_MODEL}_pubchem/best.pt \
            --molecule_type SMILES \
            --test_mode given_molecule \
            --logs $OUTPUT_DIR \
            --run_name eval_ml_ret_$DATASET
        ;;
    "remote")
        python evaluate.py \
            --val_remote_ret_data $DATASET \
            --source_model_name languagebind \
            --target_model_name remoteclip \
            --pivot_model_name $PIVOT_MODEL \
            --out_dim $OUT_DIM \
            --use_offset \
            --use_projection \
            --load_projection_checkpoint \
            --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
            --remoteclip_proj_checkpoint $CHECKPOINT_DIR/remoteclip_${PIVOT_MODEL}_remoteclip/best.pt \
            --multi_sentence True \
            --logs $OUTPUT_DIR \
            --run_name eval_remote_ret_$DATASET
        ;;
    *)
        echo "Unknown modality: $MODALITY"
        echo "Available: image, audio, molecule, remote"
        exit 1
        ;;
esac
