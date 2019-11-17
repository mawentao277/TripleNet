#!bin/bash
date
source ~/.bashrc
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=3
python main.py \
    --use_CuDNNRNN \
    --evaluate_douban \
    --use_word_embeddings \
    --max_utterance_len  30 \
    --max_utterance_num  10 \
    --model_weight_file  './model/model_douban.hdf5' \
    --word_embeddings_dim 200 \
    --word_embeddings_file './data/douban/glove_douban.txt' \
    --hidden_dim 200 \
    --word_vocab_size 297989 \
    --test_file './data/douban/test.txt' \
    --word_vocab_file './data/douban/douban_vocab' \
    --stopword_file 'stopwords.txt'
date
