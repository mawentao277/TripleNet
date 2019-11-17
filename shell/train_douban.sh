#!bin/bash
date
export CUDA_VISIBLE_DEVICES=2,3
python main.py \
    --use_CuDNNRNN \
    --train_douban \
    --use_word_embeddings \
    --max_utterance_len  30 \
    --train_batch_size 200 \
    --max_utterance_num  10 \
    --word_embeddings_dim 200 \
    --word_embeddings_file './data/douban/glove_douban.txt' \
    --hidden_dim 200 \
    --word_vocab_size 297989 \
    --test_file './data/douban/test.txt' \
    --word_vocab_file './data/douban/douban_vocab' \
    --stopword_file './data/douban/stopwords.txt' \
    --train_file './data/douban/train_shuf.txt'
date
