#!/bin/bash

TRIAL_NO=3024
NUM_ROUNDS=4

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
      --learning_rate=0.01 \
      --lr_heads_ratio=1 \
      --weight_decay=0.0001 \
      --momentum=0.9 \
      --power=0.9 \
      --loss_weight=0.1 \
      --temperature=0.1 \
      --base_temperature=0.07 \
      --labeled_sample_ratio_cl=0.5 \
      --sample_ratio_cl=0.013 \
      --sample_ratio_ce=0.2 \
      --batch_size=32 \
      --labeled_batch_ratio=0.5 \
      --num_workers=8 \
      --num_epochs=500 \
      --num_rounds=$NUM_ROUNDS \
      --resize_size=256 \
      --crop_size=224 \
      --metadata_root=datasets/folds/GLAS/fold-0 \
      --labeled_suffix=labeled_7 \
      --unlabeled_suffix=unlabeled_60 \
      --print_freq=1 \
      --eval_freq=30 \
      --round=$r \
      --trial=$TRIAL_NO
      #--debug
done

# moving generated pseudo-masks and prepare for a new trial
CAMS_PATH=datasets/GlaS/Warwick_QU_Dataset_\(Released_2016_07_08\)/CAMs
mv $CAMS_PATH/training_cams $CAMS_PATH/cams_round$NUM_ROUNDS
mv $CAMS_PATH/cams_round0 $CAMS_PATH/training_cams
mkdir $CAMS_PATH/$TRIAL_NO
for ((r=1; r<=$NUM_ROUNDS; r++)); do mv $CAMS_PATH/cams_round$r $CAMS_PATH/$TRIAL_NO/; done
