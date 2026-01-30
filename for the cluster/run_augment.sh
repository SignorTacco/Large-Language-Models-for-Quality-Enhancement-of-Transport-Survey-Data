#!/bin/sh
### Queue Name
#BSUB -q gpua100

### Job Name
#BSUB -J rag_torch_l40s

### Logs
#BSUB -o logs/torch_out_%J.txt
#BSUB -e logs/torch_err_%J.txt

### Walltime (It will be fast, 2 hours is safe)
#BSUB -W 02:00

### Resources
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

echo "=========================================================="
echo "Starting PyTorch Native Augmentation (A100)"
echo "=========================================================="

# 1. Load Modules
module purge
module load python3/3.10.13
module load cuda/12.1.1
module load cudnn/v8.9.7.29-prod-cuda-12.X

# 2. Activate Environment
source venv/bin/activate

# 3. Ensure NumPy is clean
pip install "numpy<2"

# 4. Run the NEW Script
python3 -u scripts/augment_with_rag.py

echo "Job Complete"