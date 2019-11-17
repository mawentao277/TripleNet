import codecs
import json
from keras.utils import multi_gpu_model
from triplenet_model import TripleNetModel
from callback import SaveModelCallback
from evaluate import evaluate_ubuntu, evaluate_douban
from preprocess import data_generator
import numpy as np
import random
import tensorflow as tf
import argparse
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='./data/ubuntu/train_shuf.txt', type=str, help='train file')
    parser.add_argument("--test_file", default='./data/ubuntu/test.txt', type=str, help='valid or test file')
    parser.add_argument('--output_dir', default='./model/', type=str, help='a path to output')
    parser.add_argument('--config_file', default=None, type=str, help='config file')
    parser.add_argument('--model_weight_file', default='./model/model_epoch7_seed19565_prec0.79006.hdf5', type=str)
    parser.add_argument('--stopword_file', default='./data/ubuntu/stopwords.txt', type=str)

    parser.add_argument('--train_ubuntu', action='store_true',
                        help='Whether to run training on ubuntu dataset')
    parser.add_argument('--evaluate_ubuntu', action='store_true',
                        help='Whether to run evaluate on ubuntu dataset')
    parser.add_argument('--train_douban', action='store_true',
                        help='Whether to run training on douban dataset')
    parser.add_argument('--evaluate_douban', action='store_true',
                        help='Whether to run evaluate on douban dataset')

    parser.add_argument('--gpu_nums', default=2, type=int, help='How many gpu will use')
    parser.add_argument('--train_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=100, type=int)
    parser.add_argument('--use_CuDNNRNN', action='store_true',
                        help='Whether use CuDNNGRU or CuDNNLSTM')

    parser.add_argument('--random_seed', default=10000, type=int, help='random seed')

    parser.add_argument('--use_word_embeddings', action='store_true')
    parser.add_argument('--word_vocab_file', default='./data/ubuntu/vocab_ubuntu', type=str)
    parser.add_argument('--word_embeddings_file', default='./data/ubuntu/glove_ubuntu.txt', type=str)
    parser.add_argument('--word_embeddings_dim', default=200, type=int)
    parser.add_argument('--word_vocab_size', default=297989, type=int)
    parser.add_argument('--max_utterance_num', default=10, type=int)
    parser.add_argument('--max_utterance_len', default=50, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)

    parser.add_argument('--use_char_embeddings', action='store_true')
    parser.add_argument('--char_vocab', default='./data/ubuntu/char_vocab', type=str)
    parser.add_argument('--char_vocab_size', default=300, type=int)
    parser.add_argument('--char_embeddings_dim', default=64, type=int)
    parser.add_argument('--max_token_len', default=15, type=int)
    parser.add_argument('--char_features_dim', default=200, type=int)
    parser.add_argument('--char_kernel_shape', default=3, type=int)

    args = parser.parse_args()

    if args.config_file:
        with codecs.open(args.config_file, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            for k, v in settings:
                if k not in ['config_file']:
                    args.__dict__[k] = v

    # check args validation
    experiment_on_ubuntu = args.train_ubuntu or args.evaluate_ubuntu
    experiment_on_douban = args.train_douban or args.evaluate_douban
    if experiment_on_ubuntu and experiment_on_douban:
        raise AssertionError("You have to do an experiment in one dataset at the same time!")
    if not experiment_on_ubuntu and not experiment_on_douban:
        raise AssertionError('Must do something')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        logger.info("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not args.use_word_embeddings and not args.use_char_embeddings:
        raise AssertionError('At least specified one input!')

    args.task = 'ubuntu' if experiment_on_ubuntu else 'douban'

    # set seed
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print(args)
    if args.train_ubuntu or args.train_douban:
        # init model
        with tf.device('/cpu:0'):
            model = TripleNetModel(args).model
            single_model = model
        if args.gpu_nums > 1:
            model = multi_gpu_model(model, args.gpu_nums)
        model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
        model_save_callback = SaveModelCallback(args, single_model)
        print('Training model.....')
        model.fit_generator(generator=data_generator(args, True),
                            steps_per_epoch=1000000 / args.train_batch_size, epochs=args.epochs,
                            callbacks=[model_save_callback])

    if args.evaluate_ubuntu:
        with tf.device('/cpu:0'):
            model = TripleNetModel(args).model
        model.load_weights(args.model_weight_file)
        if args.gpu_nums > 1:
            model = multi_gpu_model(model, args.gpu_nums)
        evaluate_ubuntu(args, model)
    if args.evaluate_douban:
        with tf.device('/cpu:0'):
            model = TripleNetModel(args).model
        model.load_weights(args.model_weight_file)
        if args.gpu_nums > 1:
            model = multi_gpu_model(model, args.gpu_nums)
        evaluate_douban(args, model)


if __name__ == '__main__':
    main()
