#!/bin/bash
#SBATCH --job-name=X1.scf
#SBATCH --output=/srv/app/runs/2026-09-10_115034/jobs/crystal/X1.scf.slurm.out
#SBATCH --partition=pzero
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00

set -e

module load medsci
module load quantum-espresso/7.1

infile=$(basename "X1.scf.in" .in)
indir="/srv/app/runs/2026-09-10_115034/inputs/crystal"
workdir=/hpfs/userws/$USER/workdir/$SLURM_JOBID

mkdir -p "$workdir" && cd "$workdir" && echo "$workdir"
cp -r "$indir/$infile.in" "$workdir"

srun pw.x -i "$infile.in" > "$infile.out"
cp "$infile.out" "$indir"
