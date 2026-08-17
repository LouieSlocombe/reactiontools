#!/bin/bash
#SBATCH --job-name=install
#SBATCH -N 1
#SBATCH -p htc
#SBATCH -c 128
#SBATCH -q public
#SBATCH --time=0-04:00:00
#SBATCH --mem=0
#SBATCH -o run.out
#SBATCH -e run.out
#SBATCH --export=NONE

# Submit from build_tools/:
#   sbatch sub_sol_install.sh
#
# custom_install_sol.sh purges and loads its own modules, so none are loaded here.
# No GPU is requested: the build and reactiontools itself are CPU-only.

"${SLURM_SUBMIT_DIR:-.}/custom_install_sol.sh" >> bash.out 2>&1
