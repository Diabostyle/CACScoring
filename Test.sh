#!/bin/bash
#SBATCH --ntasks=1           ### How many CPU cores do you need?
#SBATCH --mem=14G            ### How much RAM memory do you need?
#SBATCH -p express           ### The queue to submit to: express, short, long, interactive
#SBATCH --gres=gpu:1         ### How many GPUs do you need?
#SBATCH -t 0-01:00:00        ### The time limit in D-hh:mm:ss format
#SBATCH -o /trinity/home/r104791/Projet/output/out_%j.log       ### Where to store the console output (%j is the job number)
#SBATCH -e /trinity/home/r104791/Projet/error/error_%j.log      ### Where to store the error output
#SBATCH --job-name=Test ### Name your job so you can distinguish between jobs

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
#nnUNetv2_find_best_configuration 1 -c 2d 
#nnUNetv2_predict -d Dataset001_COCA -i /trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_raw/Dataset001_COCA/imagesTs/ -o /trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_raw/Dataset001_COCA/nnUNet_predict -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans
#nnUNetv2_apply_postprocessing -i /trinity/home/r104791/Projet/Prediction/Input -o /trinity/home/r104791/Projet/Prediction/Output -pp_pkl_file /trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_results/Dataset001_COCA/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 6 -plans_json /trinity/home/r104791/Projet/COCADataset/cocacoronarycalciumandchestcts-2/nn-Unetdataset/nnUNet_results/Dataset001_COCA/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json
#python3 /trinity/home/r104791/Projet/EnvProjet/lib/python3.9/site-packages/nnunetv2/evaluation/evaluate_predictions.py
python3 data_process_Ju.py