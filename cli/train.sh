#!/bin/bash
set -ex

fairseq-train \
    /home/tasaki/Machine_Translation_Proto/data-bin \
    --fp16 \
    --save-interval 10 \
    --log-interval 1 \
    --log-format simple \
    --max-epoch 120 \
    --update-freq 1 \
    --max-update 30000 \
    --max-tokens 4000 \
    --arch bart_base \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 4096 \
    --encoder-attention-heads 8 \
    --encoder-layers 8 \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 4096 \
    --decoder-attention-heads 8 \
    --decoder-layers 8 \
    --share-all-embeddings \
    --dropout 0.3 \
    --attention-dropout 0.0 \
    --activation-dropout 0.0 \
    --activation-fn gelu \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --lr 0.0015 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --clip-norm 1.0 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.3 \
    | tee /home/tasaki/Machine_Translation_Proto/train.log
