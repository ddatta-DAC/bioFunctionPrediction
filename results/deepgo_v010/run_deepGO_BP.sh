#!/bin/bash
#SBATCH -J PFP
#SBATCH -p normal_q

## Comment this next line in huckleberry
##SBATCH -p gpu_q

#SBATCH -n 1

## Next line needed only in VBI servers
##SBATCH -A fungcat

## Check sinfo before setting this
#SBATCH --nodelist hu006

#SBATCH -t 360:00
#SBATCH --mem=30G

## Uncomment for huckleberry
#SBATCH --gres=gpu:pascal:1

## comment if not huckleberry
##SBATCH --gres=gpu:1


## ---  Modules for huckleberry, uncomment accordingly --- ##
#module load anaconda2

module load cuda
module load nccl

## User specific anaconda virtual environment
source activate venv
#source activate pytorch

## Modules for discovery
#module load TensorFlow/1.6.0-foss-2018a-Python-3.6.4-CUDA-9.1.85
#source ~/.start_discovery.sh


#LOCAL="workspace"
LOCAL="Code/fungcat"
#### Code for running python
RESULTDIR="${HOME}/${LOCAL}/bioFunctionPrediction/results/deepgo_v010/"
SCRIPT_ROOT="${HOME}/${LOCAL}/bioFunctionPrediction/src/"
cd $SCRIPT_ROOT
DATA="${HOME}/${LOCAL}/bioFunctionPrediction/resources/data"
FUNCTION="bp"
OUTDIR="${RESULTDIR}/deepgo_v010_${FUNCTION}_$( date -I)"
mkdir -p $OUTDIR

BATCHSIZE=16

python ${SCRIPT_ROOT}/deepGO_v.1.0.py --resources ${SCRIPT_ROOT}/../resources --function bp  --outputdir testout --trainsize $(( 36380 /  $BATCHSIZE )) --testsize  $(( 6822 /  $BATCHSIZE )) --validationsize  $(( 2274 /  $BATCHSIZE )) --inputfile ../resources/data/data_BP --batchsize $BATCHSIZE --num_epochs 10 --featuretype ngrams

cd -
source deactivate
