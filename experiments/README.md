Experiment run un the HPC

# Jobscript example
```bash
#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J VAE_2
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"

### -- set the email address --
#BSUB <email_address>

##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o VAE_2%J.out
#BSUB -e VAE_2%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load python/3.10.7
module load cuda/12.0

source /zhome/51/c/155356/Desktop/armortized-inference-for-spectroscopy/venv/bin/activate
python /zhome/51/c/155356/Desktop/armortized-inference-for-spectroscopy/experiments/2_VAE.py
```

# Experiment results
Losses are logged to Weights and Biases and can be found here: https://wandb.ai/andreaslf/amortized-inference-for-spectroscopy/

The trained models of the experiments are stored in a results folder.
