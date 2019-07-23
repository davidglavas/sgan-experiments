#!/bin/bash
#SBATCH --job-name="SocialGAN_rand_predLength8_eth_72h"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:8G
#SBATCH --mem=8GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

##srun ../Social-GAN/venv/bin/python ../Social-GAN/scripts/trainCPU.py --delim tab --noise_dim=0 --use_gpu=0
##srun Social-GAN/venv/bin/python Social-GAN/scripts/HelloWorld.py


# before runs:  change checkpoint_name (eth8_RetrainedMax_24h), change slurm job name, change slurm timeout time
# these are exactly the same parameters as used by the paper's pretrained models (extracted params with printargs)

# eth8
srun python3 scripts/trainCPU.py --encoder_h_dim_d=48 --neighborhood_size=2.0 --clipping_threshold_d=0 --clipping_threshold_g=2.0 --delim=tab --print_every=100 --pred_len=8 --loader_num_workers=4 --d_steps=1 --batch_size=64 --num_epochs=200 --num_layers=1 --best_k=20 --obs_len=8 --skip=1 --g_steps=1 --g_learning_rate=0.0001 --l2_loss_weight=1.0 --grid_size=8 --bottleneck_dim=8 --checkpoint_name=eth8_retrainedRand72h --gpu_num=0 --restore_from_checkpoint=1 --dropout=0.0 --checkpoint_every=300 --noise_mix_type=global --decoder_h_dim_g=32 --pooling_type=pool_net --use_gpu=1 --num_iterations=9012 --batch_norm=0 --noise_type=gaussian --pool_every_timestep=0 --encoder_h_dim_g=32 --timing=0 --d_learning_rate=0.001 --checkpoint_start_from=None --embedding_dim=16 --mlp_dim=64 --num_samples_check=5000 --d_type=global --noise_dim=8 --dataset_name=eth

# hotel8
#srun python3 scripts/trainCPU.py --encoder_h_dim_d=48 --neighborhood_size=2.0 --pool_every_timestep=0 --clipping_threshold_g=2.0 --delim=tab --dataset_name=hotel --print_every=100 --skip=1 --loader_num_workers=4 --d_steps=1 --batch_size=64 --num_epochs=200 --num_layers=1 --best_k=20 --obs_len=8 --pred_len=8 --g_steps=1 --g_learning_rate=0.0001 --l2_loss_weight=1.0 --grid_size=8 --bottleneck_dim=8 --checkpoint_name=hotel8_retrainedRand_72h --gpu_num=0 --restore_from_checkpoint=1 --dropout=0.0 --noise_mix_type=global --decoder_h_dim_g=32 --pooling_type=pool_net --use_gpu=1 --num_iterations=8512 --batch_norm=0 --noise_type=gaussian --clipping_threshold_d=0 --encoder_h_dim_g=32 --checkpoint_every=300 --d_learning_rate=0.001 --checkpoint_start_from=None --timing=0 --mlp_dim=64 --num_samples_check=5000 --d_type=global --noise_dim=8 --embedding_dim=16

# univ8
#srun python3 scripts/trainCPU.py --encoder_h_dim_d=48 --num_layers=1 --neighborhood_size=2.0 --pool_every_timestep=0 --clipping_threshold_g=2.0 --delim=tab --print_every=100 --pred_len=8 --loader_num_workers=4 --d_steps=1 --encoder_h_dim_g=32 --batch_size=64 --num_epochs=200 --best_k=20 --obs_len=8 --skip=1 --g_steps=1 --g_learning_rate=0.0001 --l2_loss_weight=1.0 --grid_size=8 --bottleneck_dim=8 --checkpoint_name=univ8_retrainedRand_72h --gpu_num=0 --restore_from_checkpoint=1 --dropout=0.0 --checkpoint_every=300 --noise_mix_type=global --decoder_h_dim_g=32 --pooling_type=pool_net --use_gpu=1 --batch_norm=0 --noise_type=gaussian --clipping_threshold_d=0 --timing=0 --d_learning_rate=0.001 --checkpoint_start_from=None --embedding_dim=16 --mlp_dim=64 --num_samples_check=5000 --d_type=global --num_iterations=6937 --noise_dim=8 --dataset_name=univ

# zara1
#srun python3 scripts/trainCPU.py --encoder_h_dim_d=48 --neighborhood_size=2.0 --pool_every_timestep=0 --clipping_threshold_g=2.0 --delim=tab --print_every=100 --pred_len=8 --loader_num_workers=4 --d_steps=1 --encoder_h_dim_g=32 --batch_size=64 --num_epochs=200 --num_layers=1 --best_k=20 --obs_len=8 --skip=1 --g_steps=1 --g_learning_rate=0.0001 --l2_loss_weight=1.0 --grid_size=8 --bottleneck_dim=8 --checkpoint_name=zara1_retrainedRand_72h --gpu_num=0 --restore_from_checkpoint=1 --dropout=0.0 --checkpoint_every=300 --noise_mix_type=global --decoder_h_dim_g=32 --pooling_type=pool_net --use_gpu=1 --num_iterations=7537 --batch_norm=0 --noise_type=gaussian --clipping_threshold_d=0 --d_learning_rate=0.001 --checkpoint_start_from=None --timing=0 --mlp_dim=64 --num_samples_check=5000 --d_type=global --noise_dim=8 --dataset_name=zara1 --embedding_dim=16

# zara2
#srun python3 scripts/trainCPU.py --encoder_h_dim_d=48 --neighborhood_size=2.0 --clipping_threshold_d=0 --clipping_threshold_g=2.0 --delim=tab --dataset_name=zara2 --print_every=100 --pred_len=8 --loader_num_workers=4 --obs_len=8 --encoder_h_dim_g=32 --batch_size=64 --num_epochs=200 --num_layers=1 --best_k=20 --d_steps=1 --skip=1 --g_steps=1 --g_learning_rate=0.0001 --l2_loss_weight=1.0 --grid_size=8 --bottleneck_dim=8 --checkpoint_name=zara2_retrainedRand_72h --gpu_num=0 --restore_from_checkpoint=1 --dropout=0.0 --checkpoint_every=300 --noise_mix_type=global --decoder_h_dim_g=32 --pooling_type=pool_net --use_gpu=1 --batch_norm=0 --noise_type=gaussian --pool_every_timestep=0 --d_learning_rate=0.001 --checkpoint_start_from=None --timing=0 --mlp_dim=64 --num_samples_check=5000 --d_type=global --num_iterations=7234 --noise_dim=8 --embedding_dim=16
