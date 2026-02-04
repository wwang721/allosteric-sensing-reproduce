#!/bin/bash -l
#SBATCH --job-name=adapt1
#SBATCH --time=1-0:0:0
#SBATCH --account=bcamley1
#SBATCH --mail-type=end
#SBATCH --mail-user=wwang138@jh.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p shared


### exit on error
set -e

### module load python
python -u ./adaptation_time1.py
### python -u ./adaptation_time2.py
### python -u ./adaptation_time3.py

wait
date
