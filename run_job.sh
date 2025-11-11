#!/bin/bash
#SBATCH --job-name="entropy"
#SBATCH --nodes=1
#SBATCH --partition=base
#SBATCH --output="slurms/%x.o%j"
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --time=60:00:00

cd /home/drd92/pcfg-entropy

module load cuda/12.5

module load gcc/11.4.0
 
source venv/bin/activate

python3 -m uv pip install -r requirements.txt
python3 --version

# 0 = ngrams, 1 = pfsas, 2 = pcfgs
python3 experiments.py -j 2