#!/bin/bash
#SBATCH -J PFP
#SBATCH -p normal_q

## Comment this next line in huckleberry
##SBATCH -p gpu_q

#SBATCH -n 1

## Next line needed only in VBI servers
##SBATCH -A fungcat

## Check sinfo before setting this
#SBATCH --nodelist hu010

#SBATCH -t 1250:00
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
RESULTDIR="${HOME}/${LOCAL}/bioFunctionPrediction/results/label_emb_1"
SCRIPT_ROOT="${HOME}/${LOCAL}/bioFunctionPrediction/src/models/label_emb_1/"
cd $SCRIPT_ROOT
DATA="${HOME}/${LOCAL}/bioFunctionPrediction/resources/data"
FUNCTION="mf"
OUTDIR="${RESULTDIR}/label_emb_1_${FUNCTION}_$( date -I)"
mkdir -p $OUTDIR

BATCHSIZE=16

python ${SCRIPT_ROOT}/label_emb_model_1.py --resources ${SCRIPT_ROOT}/../../../resources --function bp  --outputdir ./../../../results/label_emb_1/ --trainsize $(( 27056 /  $BATCHSIZE )) --testsize  $(( 8444 /  $BATCHSIZE )) --validationsize  $(( 6765 /  $BATCHSIZE )) --inputfile ../../../resources/data/data_BP --batchsize $BATCHSIZE --num_epochs 12 --featuretype ngrams --maxseqlen 2002 --predict False

cd -
source deactivate
