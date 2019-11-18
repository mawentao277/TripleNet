#!bin/bash
date
export CUDA_VISIBLE_DEVICES=2,3
python main.py \
    --evaluate_ubuntu \
    --use_CuDNNRNN \
    --use_word_embeddings \
    --use_char_embeddings \
    --max_utterance_len  50 \
    --model_weight_file  './model/model_ubuntu.hdf5' \
    --word_embeddings_dim 100 \
    --hidden_dim 100 \
    --word_vocab_size 144954 \
    --test_file './data/ubuntu/test.txt'
date
