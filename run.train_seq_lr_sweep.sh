#!/bin/bash
#BSUB -J trace_lr_sweep
#BSUB -n 3
#BSUB -q gpuA
#BSUB -gpu "num=3:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o trace_lr_sweep.%J.out
#BSUB -e trace_lr_sweep.%J.err

set -euo pipefail

PROJECT_DIR="/public-supool/home/annie/translation_model/TRACE"
TRAIN_LOG_DIR="${PROJECT_DIR}/../log/train"
EVALUATION_LOG="${PROJECT_DIR}/run.evaluate_hyperparameter_ablation.log"

mkdir -p "${TRAIN_LOG_DIR}"
cd "${PROJECT_DIR}"

# These settings define a controlled learning-rate ablation. LSF assigns the
# requested GPUs, so CUDA_VISIBLE_DEVICES should not be overwritten manually.
export TRACE_DATASET_PRESET="5c"
export TRACE_DATASET_NAME="hs_5c_6k_depth0.1_cov0.1_rpm1_e25_a1_b0_exp_aug_i03_m15_lr_sweep"
export TRACE_EPOCH_NUM="25"
export TRACE_ALPHA_START="0.1"
export TRACE_ALPHA_FINAL="1.0"
export TRACE_RANKING_LOSS_WEIGHT="0"
export TRACE_RESUME="true"
export TRACE_DISABLE_EARLY_STOPPING="true"
unset TRACE_MODEL_CONFIG_PATH

for learning_rate in 0.005 0.001 0.0001; do
    export TRACE_LEARNING_RATE="${learning_rate}"
    train_log="${TRAIN_LOG_DIR}/run.train_seq.hs_5c.alpha1_beta0.lr${learning_rate}.log"
    echo "Starting TRACE learning-rate experiment: ${learning_rate}"
    torchrun --standalone --nproc-per-node=3 "${PROJECT_DIR}/run.train_seq.py" \
        > "${train_log}" 2>&1
    echo "Finished TRACE learning-rate experiment: ${learning_rate}"
done

export TRACE_LEARNING_RATE="0.001"
export TRACE_DATASET_NAME="hs_5c_6k_depth0.1_cov0.1_rpm1_e25_a1_b0_exp_aug_i03_m15"

export TRACE_MODEL_CONFIG_PATH="${PROJECT_DIR}/src/config/base_model_256d_16h_12l_64env_16ad_bs.yaml"
train_log="${TRAIN_LOG_DIR}/run.train_seq.hs_5c.256d_12l.alpha1_beta0.lr0.001.log"
echo "Starting TRACE architecture experiment: 256d, 12 layers"
torchrun --standalone --nproc-per-node=3 "${PROJECT_DIR}/run.train_seq.py" \
    > "${train_log}" 2>&1
echo "Finished TRACE architecture experiment: 256d, 12 layers"

export TRACE_MODEL_CONFIG_PATH="${PROJECT_DIR}/src/config/base_model_256d_8h_12l_64env_16ad_bs.yaml"
train_log="${TRAIN_LOG_DIR}/run.train_seq.hs_5c.256d_8h_12l.alpha1_beta0.lr0.001.log"
echo "Starting TRACE attention-head experiment: 256d, 8 heads, 12 layers"
torchrun --standalone --nproc-per-node=3 "${PROJECT_DIR}/run.train_seq.py" \
    > "${train_log}" 2>&1
echo "Finished TRACE attention-head experiment: 256d, 8 heads, 12 layers"

export TRACE_MODEL_CONFIG_PATH="${PROJECT_DIR}/src/config/base_model_256d_16h_6l_64env_16ad_bs.yaml"
train_log="${TRAIN_LOG_DIR}/run.train_seq.hs_5c.256d_6l.alpha1_beta0.lr0.001.log"
echo "Starting TRACE architecture experiment: 256d, 6 layers"
torchrun --standalone --nproc-per-node=3 "${PROJECT_DIR}/run.train_seq.py" \
    > "${train_log}" 2>&1
echo "Finished TRACE architecture experiment: 256d, 6 layers"

echo "Starting zero-shot hyperparameter evaluation"
python "${PROJECT_DIR}/run.evaluate_hyperparameter_ablation.py" \
    > "${EVALUATION_LOG}" 2>&1
echo "Finished zero-shot hyperparameter evaluation"
