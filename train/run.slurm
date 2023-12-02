#!/bin/bash
#SBATCH --job-name=train         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2021.11
conda activate /tigress/yifanl/usr/licensed/anaconda3/2021.11/envs/dp-v2.2.7

( /bin/sh -c '{ if [ ! -f model.ckpt.index ]; then dp train input.json; else dp train input.json --restart model.ckpt; fi }'&&dp freeze&&dp compress ) 1>>train.log 2>>train.log
