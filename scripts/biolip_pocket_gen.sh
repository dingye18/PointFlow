#! /bin/bash

dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=16
lr=2e-3
epochs=4000
log_name="gen/biolip-pocket-seqback"
ds="BioLipPocketPointCloud"

python train.py \
    --log_name ${log_name} \
    --lr ${lr} \
    --train_T False \
    --dataset_type ${ds} \
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --save_freq 50 \
    --viz_freq 1 \
    --log_freq 1 \
    --val_freq 10 \
    --input_node_features_dim 17 \
    --input_edge_features_dim 15 \
    --use_dist_in_layers \
    --use_latent_flow

echo "Done"
exit 0
