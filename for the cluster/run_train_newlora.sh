#!/bin/sh
### Queue Name (Restricted/Locked to gpul40s as requested)
#BSUB -q gpul40s

### Job Name
#BSUB -J thesis_llm

### Output and Error Logs (%J is the Job ID)
#BSUB -o logs/output_%J.txt
#BSUB -e logs/error_%J.txt

### Walltime (4 hours)
#BSUB -W 04:00

### Resources
#BSUB -n 4                       # Request 4 CPU cores
#BSUB -R "span[hosts=1]"         # Keep all cores on the same node
#BSUB -R "rusage[mem=48GB]"      # Request 48GB System RAM
#BSUB -gpu "num=1:mode=exclusive_process"  # Request 1 GPU in exclusive mode

echo "=========================================================="
echo "Starting Thesis Job on DTU HPC (LSF Scheduler)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================================="

# 1. Load Modules
# DTU LSF specific modules
module purge
module load python3/3.10.13
module load cuda/12.1.1
module load cudnn/v8.9.7.29-prod-cuda-12.X

# 2. Activate Environment
source venv/bin/activate

# Check GPU
echo "GPU Allocated:"
nvidia-smi

# 3. RUN TRAINING
echo "----------------------------------------------------------"
echo "STEP 1: Starting Fine-Tuning (Train)"
echo "----------------------------------------------------------"
# -u forces unbuffered output so you see logs in real-time
python3 -u scripts/train.py

# 4. RUN INFERENCE
# Only run if training succeeds (Exit code 0)
if [ $? -eq 0 ]; then
    echo "----------------------------------------------------------"
    echo "STEP 2: Starting Bulk Inference (Challenge)"
    echo "----------------------------------------------------------"
    python3 -u scripts/inference.py
else
    echo "Training failed! Skipping inference."
    exit 1
fi

echo "=========================================================="
echo "Job Complete"
echo "=========================================================="