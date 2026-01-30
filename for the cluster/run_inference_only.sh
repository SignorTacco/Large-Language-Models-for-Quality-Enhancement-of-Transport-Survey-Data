#!/bin/sh
### Queue Name
#BSUB -q gpua100

### Job Name
#BSUB -J thesis_inference

### Output and Error Logs
#BSUB -o logs/inf_output_%J.txt
#BSUB -e logs/inf_error_%J.txt

### Walltime
#BSUB -W 04:00

### Resources
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=40GB]"
#BSUB -gpu "num=1:mode=exclusive_process"

echo "=========================================================="
echo "Starting Inference Only Job"
echo "Node: $(hostname)"
echo "=========================================================="

# 1. Load Modules
module purge
module load python3/3.10.13
module load cuda/12.1.1
module load cudnn/v8.9.7.29-prod-cuda-12.X

# 2. Activate Environment
source venv/bin/activate

# 3. VERIFY VLLM (Sanity Check)
# This prints the version so we know it's actually there this time
pip show vllm

# 4. RUN INFERENCE ONLY
echo "----------------------------------------------------------"
echo "STEP 2: Starting Bulk Inference (Challenge)"
echo "----------------------------------------------------------"
python3 -u scripts/inference.py

echo "=========================================================="
echo "Job Complete"
echo "=========================================================="