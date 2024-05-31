#!/bin/bash
IMG_DIR="/media/xavier/Storage/model_evaluation/alignment/crops_l/*"
NETWORK="/home/xavier/PycharmProjects/training-runs/stylegan2/00000-stylegan2-trainingset2-gpus4-batch112-gamma10/network-snapshot-001512.pkl"
OUTPUT_DIR="/media/xavier/Storage/model_evaluation/alignment/crops_bp"
degree=5
i=1
for FILE in $IMG_DIR;
do
  IMG_FILE=${FILE##*/};
  IMG_NAME=${IMG_FILE%%.*};
  echo $FILE;
  echo $IMG_NAME;
  python projector.py --outdir=$OUTPUT_DIR/$IMG_NAME --network=$NETWORK --target=$FILE --save-video=False
# nohup  python projector.py --outdir=/home/xavier/Documents/project/stylegan3/training-runs/00004-stylegan2-myxo-selected-gpus1-batch32-gamma10/backproject/$IMG_NAME --network=$NETWORK --target=$FILE --save-video=False >/dev/null 2>&1 &
#  [ `expr $i % $degree` -eq 0 ] && wait
  i=$[$i+1]
done