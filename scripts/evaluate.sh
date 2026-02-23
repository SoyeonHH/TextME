#!/bin/bash
# =============================================================================
# Evaluate TextME on all benchmarks
#
# Extracted from: EfficientBind/scripts/evaluation/*.sh
#
# Evaluation tasks:
# - Text→Image Retrieval (COCO, Flickr30K)
# - Text→Audio Retrieval (AudioCaps, Clotho)
# - Text→3D Retrieval (Objaverse-LVIS)
# - Text→Molecule Retrieval (DrugBank)
# - Text→Remote Sensing Retrieval (RSICD, RSITMD, UCM)
# - Zero-shot 3D Classification (ModelNet40, ScanObjectNN)
# - Zero-shot Audio Classification (ESC-50, AudioSet)
# - Zero-shot X-ray Classification (RSNA, SIIM)
# =============================================================================

set -e

# Configuration
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
OFFSET_DIR=${OFFSET_DIR:-"./offsets"}
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_DIR=${OUTPUT_DIR:-"./results"}
OFFSET_NUM=${OFFSET_NUM:-5000}
OUT_DIM=${OUT_DIM:-2560}
PIVOT_MODEL=${PIVOT_MODEL:-"qwen3_embed_4b"}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "=============================================="
echo "TextME Evaluation Suite"
echo "=============================================="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Offset dir: $OFFSET_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# =============================================================================
# TEXT→IMAGE RETRIEVAL
# =============================================================================
echo ""
echo "=========================================="
echo "TEXT→IMAGE RETRIEVAL"
echo "=========================================="

# COCO
echo "[Image] Evaluating COCO..."
python evaluate.py \
    --val_il_ret_data coco \
    --source_model_name languagebind \
    --target_model_name clip \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --clip_proj_checkpoint $CHECKPOINT_DIR/clip_${PIVOT_MODEL}_coco/best.pt \
    --batch_size_val $BATCH_SIZE \
    --logs $OUTPUT_DIR \
    --run_name eval_il_ret_coco

# Flickr30K
echo "[Image] Evaluating Flickr30K..."
python evaluate.py \
    --val_il_ret_data flickr \
    --source_model_name languagebind \
    --target_model_name clip \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --clip_proj_checkpoint $CHECKPOINT_DIR/clip_${PIVOT_MODEL}_coco/best.pt \
    --batch_size_val $BATCH_SIZE \
    --multi_sentence True \
    --logs $OUTPUT_DIR \
    --run_name eval_il_ret_flickr

# =============================================================================
# TEXT→AUDIO RETRIEVAL
# =============================================================================
echo ""
echo "=========================================="
echo "TEXT→AUDIO RETRIEVAL"
echo "=========================================="

# AudioCaps
echo "[Audio] Evaluating AudioCaps..."
python evaluate.py \
    --val_al_ret_data AudioCaps \
    --source_model_name languagebind \
    --target_model_name clap \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --clap_proj_checkpoint $CHECKPOINT_DIR/clap_${PIVOT_MODEL}_audiocaps/best.pt \
    --batch_size_val $BATCH_SIZE \
    --logs $OUTPUT_DIR \
    --run_name eval_al_ret_audiocaps

# Clotho
echo "[Audio] Evaluating Clotho..."
python evaluate.py \
    --val_al_ret_data Clotho \
    --source_model_name languagebind \
    --target_model_name clap \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --clap_proj_checkpoint $CHECKPOINT_DIR/clap_${PIVOT_MODEL}_audiocaps/best.pt \
    --batch_size_val $BATCH_SIZE \
    --multi_sentence True \
    --logs $OUTPUT_DIR \
    --run_name eval_al_ret_clotho

# =============================================================================
# TEXT→MOLECULE RETRIEVAL
# =============================================================================
echo ""
echo "=========================================="
echo "TEXT→MOLECULE RETRIEVAL"
echo "=========================================="

# DrugBank (Pharmacodynamics)
echo "[Molecule] Evaluating DrugBank..."
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
    --batch_size_val 4 \
    --logs $OUTPUT_DIR \
    --run_name eval_ml_ret_drugbank

# =============================================================================
# TEXT→REMOTE SENSING RETRIEVAL
# =============================================================================
echo ""
echo "=========================================="
echo "TEXT→REMOTE SENSING RETRIEVAL"
echo "=========================================="

# RSICD
echo "[Remote] Evaluating RSICD..."
python evaluate.py \
    --val_remote_ret_data rsicd \
    --source_model_name languagebind \
    --target_model_name remoteclip \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --remoteclip_proj_checkpoint $CHECKPOINT_DIR/remoteclip_${PIVOT_MODEL}_remoteclip/best.pt \
    --batch_size_val $BATCH_SIZE \
    --multi_sentence True \
    --logs $OUTPUT_DIR \
    --run_name eval_remote_ret_rsicd

# RSITMD
echo "[Remote] Evaluating RSITMD..."
python evaluate.py \
    --val_remote_ret_data rsitmd \
    --source_model_name languagebind \
    --target_model_name remoteclip \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --remoteclip_proj_checkpoint $CHECKPOINT_DIR/remoteclip_${PIVOT_MODEL}_remoteclip/best.pt \
    --batch_size_val $BATCH_SIZE \
    --multi_sentence True \
    --logs $OUTPUT_DIR \
    --run_name eval_remote_ret_rsitmd

# UCM
echo "[Remote] Evaluating UCM..."
python evaluate.py \
    --val_remote_ret_data ucm \
    --source_model_name languagebind \
    --target_model_name remoteclip \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --remoteclip_proj_checkpoint $CHECKPOINT_DIR/remoteclip_${PIVOT_MODEL}_remoteclip/best.pt \
    --batch_size_val $BATCH_SIZE \
    --multi_sentence True \
    --logs $OUTPUT_DIR \
    --run_name eval_remote_ret_ucm

# =============================================================================
# ZERO-SHOT 3D CLASSIFICATION
# =============================================================================
echo ""
echo "=========================================="
echo "ZERO-SHOT 3D CLASSIFICATION"
echo "=========================================="

# ModelNet40
echo "[3D] Evaluating ModelNet40..."
python evaluate.py \
    --val_p_cls_data modelnet40 \
    --source_model_name languagebind \
    --target_model_name uni3d \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --uni3d_proj_checkpoint $CHECKPOINT_DIR/uni3d_${PIVOT_MODEL}_objaverse/best.pt \
    --batch_size_val $BATCH_SIZE \
    --logs $OUTPUT_DIR \
    --run_name eval_p_cls_modelnet40

# ScanObjectNN
echo "[3D] Evaluating ScanObjectNN..."
python evaluate.py \
    --val_p_cls_data scanobjnn \
    --source_model_name languagebind \
    --target_model_name uni3d \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --uni3d_proj_checkpoint $CHECKPOINT_DIR/uni3d_${PIVOT_MODEL}_objaverse/best.pt \
    --batch_size_val $BATCH_SIZE \
    --logs $OUTPUT_DIR \
    --run_name eval_p_cls_scanobjnn

# =============================================================================
# ZERO-SHOT AUDIO CLASSIFICATION
# =============================================================================
echo ""
echo "=========================================="
echo "ZERO-SHOT AUDIO CLASSIFICATION"
echo "=========================================="

# ESC-50
echo "[Audio] Evaluating ESC-50..."
python evaluate.py \
    --val_a_cls_data esc50 \
    --source_model_name languagebind \
    --target_model_name clap \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --clap_proj_checkpoint $CHECKPOINT_DIR/clap_${PIVOT_MODEL}_audiocaps/best.pt \
    --batch_size_val $BATCH_SIZE \
    --logs $OUTPUT_DIR \
    --run_name eval_a_cls_esc50

# AudioSet
echo "[Audio] Evaluating AudioSet..."
python evaluate.py \
    --val_a_cls_data AudioSet \
    --source_model_name languagebind \
    --target_model_name clap \
    --pivot_model_name $PIVOT_MODEL \
    --out_dim $OUT_DIM \
    --use_offset \
    --use_projection \
    --load_projection_checkpoint \
    --languagebind_proj_checkpoint $CHECKPOINT_DIR/languagebind_${PIVOT_MODEL}_coco/best.pt \
    --clap_proj_checkpoint $CHECKPOINT_DIR/clap_${PIVOT_MODEL}_audiocaps/best.pt \
    --batch_size_val $BATCH_SIZE \
    --logs $OUTPUT_DIR \
    --run_name eval_a_cls_audioset

# =============================================================================
# ZERO-SHOT X-RAY CLASSIFICATION
# =============================================================================
echo ""
echo "=========================================="
echo "ZERO-SHOT X-RAY CLASSIFICATION"
echo "=========================================="

# RSNA Pneumonia
echo "[X-ray] Evaluating RSNA Pneumonia..."
python evaluate.py \
    --val_x_cls_data rsna \
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
    --run_name eval_x_cls_rsna

# SIIM Pneumothorax
echo "[X-ray] Evaluating SIIM Pneumothorax..."
python evaluate.py \
    --val_x_cls_data siim \
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
    --run_name eval_x_cls_siim

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================================="
