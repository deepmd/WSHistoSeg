#!/bin/bash

TRIAL_NO=2007
NUM_ROUNDS=5

export CUBLAS_WORKSPACE_CONFIG=:4096:8

for ((r=1; r<=$NUM_ROUNDS; r++))
do
    python main_contrastive_selftraining.py \
      --data_root=datasets/GlaS \
      --task=stdcl \
      --encoder_name=resnet50 \
      --proj_dim=128 \
      --num_classes=2 \
      --pretrained=weights \
      --use_aspp \
      --optimizer=SGD \
      --lr_policy=lambda_poly \
      --learning_rate=0.001 \
      --lr_heads_ratio=10 \
      --weight_decay=0.0001 \
      --momentum=0.9 \
      --power=0.9 \
      --loss_weight=0.1 \
      --temperature=0.1 \
      --base_temperature=0.07 \
      --sample_ratio_cl=0.04 \
      --sample_ratio_ce=0.2 \
      --batch_size=32 \
      --num_workers=8 \
      --num_epochs=1000 \
      --num_rounds=$NUM_ROUNDS \
      --resize_size=256 \
      --crop_size=224 \
      --metadata_root=datasets/folds/GLAS/fold-0 \
      --print_freq=1 \
      --eval_freq=30 \
      --round=$r \
      --trial=$TRIAL_NO
      #--debug
done

# moving generated pseudo-masks and prepare for a new trial
CAMS_PATH=datasets/GlaS/Warwick_QU_Dataset_\(Released_2016_07_08\)/CAMs
mv $CAMS_PATH/Layer4 $CAMS_PATH/Layer4_round$NUM_ROUNDS
mv $CAMS_PATH/Layer4_round1 $CAMS_PATH/Layer4
mkdir $CAMS_PATH/$TRIAL_NO
for ((r=2; r<=$NUM_ROUNDS; r++)); do mv $CAMS_PATH/Layer4_round$r $CAMS_PATH/$TRIAL_NO/; done