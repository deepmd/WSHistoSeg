#!/bin/bash

for r in {1..5}
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
      --sample_ratio_cl=0.03 \
      --sample_ratio_ce=0.2 \
      --batch_size=32 \
      --labeled_batch_ratio=0.1 \
      --num_workers=8 \
      --num_epochs=1000 \
      --num_rounds=5 \
      --pseudo_labeling_step=200 \
      --resize_size=256 \
      --crop_size=224 \
      --metadata_root=datasets/folds/GLAS/fold-0 \
      --labeled_suffix=labeled_7 \
      --unlabeled_suffix=unlabeled_60 \
      --print_freq=1 \
      --eval_freq=30 \
      --round=$r \
      --trial=2001
      #--debug
done