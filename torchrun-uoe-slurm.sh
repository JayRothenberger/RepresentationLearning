#!/bin/bash

#SBATCH --partition=disc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
# Thread count:
#SBATCH --cpus-per-task=16
# memory in MB
#SBATCH --mem=100000
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/home/pkage/ou-collab-simclr/torch_clr_%04a_stdout.txt
#SBATCH --error=/home/pkage/ou-collab-simclr/torch_clr_%04a_stderr.txt
#SBATCH --time=10:00:00
#SBATCH --job-name=clr_torch
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/pkage/ou-collab-simclr/
#
#################################################
# $SLURM_ARRAY_TASK_ID
# Used to use --exclusive and --nodelist=c732

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")
# incorrect lscratch dir
export LSCRATCH=/lscratch/15937969
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=16

echo Node IP: $head_node_ip


srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $SLURM_NTASKS \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
mp.py



