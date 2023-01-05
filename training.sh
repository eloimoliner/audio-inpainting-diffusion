#!/bin/bash
#SBATCH  --time=2-23:59:59
##SBATCH  --time=03:59:59
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=filter_score_model
#SBATCH  --gres=gpu:a100:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/%a/train_%j.out

#SBATCH --array=[50]

module load anaconda
source activate /scratch/work/molinee2/conda_envs/cqtdiff
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES
#export HYDRA_FULL_ERROR=1
#export CUDA_LAUNCH_BLOCKING=1
n=$SLURM_ARRAY_TASK_ID

n=cqtdiff+_MAESTRO #original CQTDiff (with fast implementation) (22kHz)

if [[ $n -eq CQTdiff+_MAESTRO ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/audio-inpainting-diffusion/experiments/cqtdiff+_MAESTRO/22k_8s-750000.pt"
    exp=maestro22k_8s
    network=paper_1912_unet_cqt_oct_attention_adaLN_2
    dset=maestro_allyears
    tester=inpainting_tester
    CQT=True
fi

PATH_EXPERIMENT=experiments/$n
mkdir $PATH_EXPERIMENT

#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python train.py model_dir="$PATH_EXPERIMENT" \
               dset=$dset \
               exp=$exp \
               network=$network \
               tester=$tester \
               tester.checkpoint=$ckpt \
              tester.filter_out_cqt_DC_Nyq=$CQT \
                logging=huge_model_logging \
                exp.batch=1 \
                exp.resume=False
