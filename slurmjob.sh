#!/bin/bash

#SBATCH --job-name=hmc
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=192G
#SBATCH --mail-user=ksharma2@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dmz_spk_sharma2026

cd $SLURM_SUBMIT_DIR
python run_mcmc.py --Pk_model BCEmu7_high_z --dir_path data/frb_results --sample frb_sample --N_params 5 --eta_prior Chandra --sample_extended Yes --host_z_evol No
