#!/bin/bash
#SBATCH --job-name=critic_pretraining
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=federico.carrara@fht.org
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/group/jug/federico/critic_net_training/2604/hpc_outputs/training/training_%j.log
#SBATCH --mem=64GB
#SBATCH --array=0-7

ARGS1=("Nucleus" "Mitochondria" "Endoplasmic reticulum" "Microtubules")
ARGS2=(64 256)

COMBOS=()
for V1 in "${ARGS1[@]}"; do
    for V2 in "${ARGS2[@]}"; do
        COMBOS+=("$V1 $V2")
    done
done
read VAL1 VAL2 <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

python scripts/train_binary.py \
    --dataset CellAtlas --target $VAL1 \
    --bg-cache /group/jug/federico/data/CellAtlas/precomputed_bg_CellAtlas_NucMitEndMic_crop256_stride64.json \
    --crop_size $VAL2 --num-crops 32 --aug geometric \
    --neg-families trivial \
    --batch-size 32 --epochs 50 --num-workers 6 \
    --calibrate --log \