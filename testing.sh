#!/bin/bash
#SBATCH  --time=00:59:59
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=freq_convs_everywhere
##SBATCH  --gres=gpu:a100:1
#SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/%a/train_%j.out


module load anaconda
source activate /scratch/work/molinee2/conda_envs/cqtdiff
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES
#export HYDRA_FULL_ERROR=1
#export CUDA_LAUNCH_BLOCKING=1
n=$SLURM_ARRAY_TASK_ID

#maestro

#n=cqtdiff+_MAESTRO #trained on maestro fs=22.05 kHz, audio_len=8s
n=cqtdiff+_MUSICNET #trained on musixnet fs=44.1kHz, audio_len=4s

if [[ $n -eq CQTdiff+_MUSICNET ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/audio-inpainting-diffusion/experiments/cqtdiff+_MUSICNET/musicnet_44k_4s_560000.pt"
    exp=musicnet44k_4s
    network=paper_1912_unet_cqt_oct_attention_44k_2
    dset=maestro_allyears
    tester=inpainting_tester
elif [[ $n -eq CQTdiff+_MAESTRO ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/audio-inpainting-diffusion/experiments/cqtdiff+_MAESTRO/maestro_22k_8s-750000.pt"
    exp=maestro22k_8s
    network=paper_1912_unet_cqt_oct_attention_adaLN_2
    dset=maestro_allyears
    tester=inpainting_tester


PATH_EXPERIMENT=experiments/$n
#PATH_EXPERIMENT=experiments/cqtdiff_original
mkdir $PATH_EXPERIMENT


#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python test.py model_dir="$PATH_EXPERIMENT" \
               dset=$dset \
               exp=$exp \
               network=$network \
               tester=$tester \
               tester.checkpoint=$ckpt \
                
