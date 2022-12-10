#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --job-name=cv
#SBATCH --gres=gpu:rtx8000:1 
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user=ds5749@nyu.edu

module purge

singularity exec --nv \
    --overlay /scratch/ds5749/NLQ/overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c \
    "source /ext3/miniconda3/etc/profile.d/conda.sh; conda activate vslnet; 
    python siamese_train.py --model_name TransformerNet"

    # SiameseConvNet, TransformerNet