#!/bin/bash
#SBATCH  --time=2:29:59
#SBATCH --mem=30G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sgirtgaos
##SBATCH  --gres=gpu:a100:1
#SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/inpainting_test_shorgapts_%j.out


module load anaconda
source activate /scratch/work/molinee2/conda_envs/cqtdiff
module load gcc/8.4.0
export TORCH_USE_RTLD_GLOBAL=YES
#export HYDRA_FULL_ERROR=1
#export CUDA_LAUNCH_BLOCKING=1
n=$SLURM_ARRAY_TASK_ID

#maestro

#n=3 #original CQTDiff (with fast implementation) (22kHz)
#n=49 #dance diffusion (16 kHz)
#n=44 #cqtdiff+ no attention maestro 8s
#n=45 #cqtdiff+ attention maestro 8s
#n=50 #cqtdiff+ attention maestro 8s (alt version)

#n=54 #cqtdiff+ maestro 8s (alt version)
#n=50 #cqtdiff+ attention maestro 8s (alt version)

#n=56 #ADP

n=51 #musicnet

if [[ $n -eq 54 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/54/22k_8s-850000.pt"
    exp=maestro22k_8s
    network=paper_1912_unet_cqt_oct_noattention_adaln
    tester=inpainting_tester
    dset=maestro_allyears
    CQT=True
elif [[ $n -eq 3 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/3/weights-489999.pt"
    exp=test_cqtdiff_22k
    network=unet_cqtdiff_original
    dset=maestro_allyears
    CQT=False

elif [[ $n -eq 56 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/56/22k_8s-510000.pt"
    exp=maestro22k_131072
    network=ADP_raw_patching
    tester=inpainting_tester
    dset=maestro_allyears
    CQT=False
elif [[ $n -eq 50 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/50/22k_8s-750000.pt"
    exp=maestro22k_8s
    network=paper_1912_unet_cqt_oct_attention_adaLN_2
    dset=maestro_allyears
    tester=inpainting_tester
    CQT=True

elif [[ $n -eq 51 ]] 
then
    ckpt="/scratch/work/molinee2/projects/ddpm/diffusion_autumn_2022/A-diffusion/experiments/51/44k_4s-560000.pt"
    exp=musicnet44k_4s
    network=paper_1912_unet_cqt_oct_attention_44k_2
    dset=inpainting_musicnet_50
    #dset=inpainting_musicnet
    tester=inpainting_tester_shortgaps
    CQT=True
fi


PATH_EXPERIMENT=experiments/inpainting_tests/$n
#PATH_EXPERIMENT=experiments/cqtdiff_original
mkdir $PATH_EXPERIMENT


#python train_w_cqt.py path_experiment="$PATH_EXPERIMENT"  $iteration
python test.py model_dir="$PATH_EXPERIMENT" \
               dset=$dset \
               exp=$exp \
               network=$network \
               tester=$tester \
               tester.checkpoint=$ckpt \
              tester.filter_out_cqt_DC_Nyq=$CQT
                
