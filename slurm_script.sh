#!/bin/bash
#SBATCH --ntasks=1           ### How many CPU cores do you need?
#SBATCH --mem=14G            ### How much RAM memory do you need?
#SBATCH -p long         ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1         ### How many GPUs do you need?
#SBATCH -t 0-48:00:00        ### The time limit in D-hh:mm:ss format
#SBATCH -o /trinity/home/r104791/Projet/output/out_%j.log       ### Where to store the console output (%j is the job number)
#SBATCH -e /trinity/home/r104791/Projet/error/error_%j.log      ### Where to store the error output
#SBATCH --job-name=3DF0 ### Name your job so you can distinguish between jobs

# ----- Load the modules -----
module purge
module load Python/3.9.5-GCCcore-10.3.0
# replace with required python version! (check with module avail which versions are available)

# If you need to read/write many files quickly in tmp directory use:
# source "/tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env"

# ----- Activate virtual environment -----
# Do this after loading python module
source /trinity/home/r104791/Projet/EnvProjet/bin/activate
# replace the above path with your own virtualenv!

# ----- Your tasks -----
#nnUNetv2_plan_and_preprocess -d 5 --verify_dataset_integrity
#nnUNet_keep_files_open=True nnUNet_compile=True nnUNet_n_proc_DA=0 nnUNetv2_train 5 2d 4 -device cuda --npz 
nnUNet_keep_files_open=True nnUNet_compile=True nnUNet_n_proc_DA=0 nnUNetv2_train 2 3d_fullres 4 -device cuda --npz 
