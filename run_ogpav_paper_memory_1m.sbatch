#!/bin/bash --login
#SBATCH -A mendoza_q
#SBATCH --exclude=amr-163,amr-178,amr-179,acm-028,acm-034
#SBATCH --job-name=ogpav-mem-1m
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --constraint=amd20

# Total CPU memory
# All available memory per GPU is allocated by default.
# Specify "M" or "G" for MB and GB respectively.
#SBATCH --mem=30G

# Wall time
# Format: "minutes", "hours:minutes:seconds", "days-hours", or
# "days-hours:minutes".
#SBATCH --time=08:00:00

# Mail type
# Events that trigger email notifications.
#SBATCH --mail-type=ALL

# Mail address
#SBATCH --mail-user=susanalopezmoreno@gmail.com

# Standard output and error to one file
# %x: job name, %j: job ID
#SBATCH --output=%x-%j.SLURMout

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the OGPAV directory}"

module purge
module load Python/3.11.3

ENVIRONMENT="${SLURM_SUBMIT_DIR}/ogpav_env/bin/activate"
if [[ ! -f "${ENVIRONMENT}" ]]; then
    echo "Missing Python environment activation file: ${ENVIRONMENT}"
    exit 1
fi
source "${ENVIRONMENT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export TMPDIR="${SLURM_TMPDIR}"
else
    export TMPDIR="${SLURM_SUBMIT_DIR}/tmp/${SLURM_JOB_ID}"
    mkdir -p "${TMPDIR}"
fi

MODELS=(linear_low linear_mix linear_high)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"

echo "Compute node: $(hostname)"
echo "Python executable: $(command -v python)"
python --version
python -c 'import networkx, numpy, psutil; print(f"numpy={numpy.__version__}, networkx={networkx.__version__}, psutil={psutil.__version__}")'
echo "Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Temporary directory: ${TMPDIR}"

srun python -u experiment_ogpav_scale.py \
    --model "${MODEL}" \
    --sizes 1000000 \
    --variant default \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "Experiment completed successfully."
scontrol show job "${SLURM_JOB_ID}"
sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName%24,State,Elapsed,ReqMem,MaxRSS,MaxVMSize,ExitCode || true
