#!/bin/sh
### Queue Name
#BSUB -q gpul40s

### Job Name
#BSUB -J build_rag_index

### Logs
#BSUB -o logs/index_out_%J.txt
#BSUB -e logs/index_err_%J.txt

### Walltime 
#BSUB -W 03:00

### Resources
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

echo "=========================================================="
echo "Starting Index Build Job"
echo "Node: $(hostname)"
echo "=========================================================="

# 1. Load Modules (Same as before)
module purge
module load python3/3.10.13
module load cuda/12.1.1
module load cudnn/v8.9.7.29-prod-cuda-12.X

# 2. Activate Environment
source venv/bin/activate

# 3. FIX: FORCE NUMPY DOWNGRADE
# This is the critical line to fix your error
pip install "numpy<2"

# 4. Install other dependencies (if missing)
pip install faiss-gpu sentence-transformers

# 5. Run Script
python3 -u scripts/build_rag_index.py

echo "=========================================================="
echo "Job Complete"
echo "=========================================================="