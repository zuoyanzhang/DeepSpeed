#!/bin/bash
#SBATCH -J ds_chunk_demo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1   
#SBATCH --output=logs/ds_chunk_demo.out.log
#SBATCH --error=logs/ds_chunk_demo.err.log

python ./chunk_gemm_test.py
