#!/bin/bash
#SBATCH --job-name=land_cover_deploy
#SBATCH --account=def-ramankut-ab
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00

module load StdEnv/2020
module load python/3.6
module load gcc
module load gdal

source ~/projects/def-ramankut-ab/alantkt/env/agland/bin/activate
python -u generate_land_cover_counts.py --output_dir ./pred_input_map

echo "End"