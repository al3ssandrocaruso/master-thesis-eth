#!/bin/bash

#SBATCH --job-name=embeddings_clf_10cv
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:4

# Models to run
models=("lstm" "cnn" "rnn" "transformer" "lstm_vae" "cnn_vae" "rnn_vae" "transformer_vae")

# Base paths
PYTHONPATH="/cluster/customapps/smslab/cgallego_git/master-thesis/"
SCRIPT_PATH="/cluster/customapps/smslab/cgallego_git/master-thesis/experiments/embeddings_noCV/embeddings_noCV.py"
VENV_PATH="/cluster/customapps/smslab/cgallego_venvs/venv/bin/activate"

for model in "${models[@]}"; do
    JOB_SCRIPT="run_${model}.sh"
    
    # Create a separate script for each model
    cat <<EOT > $JOB_SCRIPT
#!/bin/bash

#SBATCH --job-name=embeddings_${model}_10cv
#SBATCH --output=output_${model}.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:4

export PYTHONPATH=\$PYTHONPATH:$PYTHONPATH
source $VENV_PATH
python $SCRIPT_PATH --model $model
EOT

    # Submit the job
    sbatch $JOB_SCRIPT
done
