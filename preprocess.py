from keras.preprocessing.sequence import pad_sequences
import numpy as np
import codecs

def load_stopwords(fname):
    vocab_file = codecs.open(fname, 'r', encoding='utf-8')
    stopword_list = []
    for word in vocab_file:
        stopword_list.append(word.strip())
    return stopword_list


def load_vocab(fname):
    vocab_file = codecs.open(fname, 'r', encoding='utf-8')
    word2id_dict = {}
    word_list = vocab_file.readlines()
    for index, item in enumerate(word_list):
        word = item.strip()
        word2id_dict[word] = index + 1
    return word2id_dict


def word2id_list(tokens, word2id_dict):
    id_list = []
    for token in tokens:
        if token in word2id_dict:
            token_id = word2id_dict[token]
            id_list.append(token_id)
        else:
            token_id = word2id_dict['unk']
            id_list.append(token_id)
    return id_list


def load_char_vocab(fname):
    vocab_file = codecs.open(fname, 'r', encoding='utf-8')
    char2id_dict = {}
    char_list = vocab_file.readlines()
    for idx, char in enumerate(char_list):
        char2id_dict[str(char.replace('\n', '').strip())] = idx + 1
    return char2id_dict


def word2char_id_list(word, char2id_dict):
    word_char_id_list = []

    for char in word:
        word_char_id_list.append(char2id_dict.get(char, 299))
    return word_char_id_list


def generate_word_match_feature(word_list, candidate_set, stopword_list):
    feature_list = []
    for word in word_list:
        if word in candidate_set and word not in stopword_list:
            feature_list.append(1.0)
        else:
            feature_list.append(0.0)
    return feature_list


def load_word_embeddings(config, fname, word2id_dict):
    print('Loading pre-trained Glove embeddings......')
    embeddings_idx = {}
    embeddings_file = codecs.open(fname, 'r', encoding='utf-8')
    for line in embeddings_file:
        line_split = line.replace('\n', '').strip().split(' ')
        word = line_split[0]
        embedding = np.asanyarray(line_split[1:], dtype='float32')
        embeddings_idx[word] = embedding

    embeddings_file.close()

    count = 0
    embeddings_matrix = np.zeros((config.word_vocab_size, config.word_embeddings_dim))
    for word, word_id in word2id_dict.items():
        embedding_vector = embeddings_idx.get(word)
        if embedding_vector is not None:
            count += 1
            embeddings_matrix[word_id] = embedding_vector
    print(f'Loading {count} words in embeddings file')
    return embeddings_matrix


def load_test_label(config):
    data_file = codecs.open(config.test_file, 'r', encoding='utf-8')

    example_id_list = []
    label_list = []

    example_id = 0
    num = 0
    for line in data_file:
        label = int(line.strip().split('\t')[0])

        if config.task == 'ubuntu':
            if label == 1:
                example_id += 1
        elif config.task == 'douban':
            if num == 0:
                example_id = 0
            elif num % 10 == 0:
                example_id += 1
            num += 1
        else:
            raise AssertionError('Invalid task!')

        example_id_list.append(example_id)
        label_list.append(label)

    label_list = np.asarray(label_list)
    return example_id_list, label_list


def data_generator(config, is_train=True):
    """return batch data"""

    # for char embedding
    padding_type = 'pre'
    if config.use_char_embeddings:
        char2id_dict = load_char_vocab(config.char_vocab)

    while True:
        data_file_name = config.train_file if is_train else config.test_file
        data_file = codecs.open(data_file_name, 'r', encoding='utf-8')

        batch_size = config.train_batch_size if is_train else config.test_batch_size

        # for word embedding
        context_matrix = []
        context_feature_matrix = []
        query_matrix = []
        query_feature_matrix = []
        reply_matrix = []
        reply_feature_matrix = []
        y_matrix = []

        word2id_dict = load_vocab(config.word_vocab_file)
        stopword_list = load_stopwords(config.stopword_file)

        context_char_matrix = []
        query_char_matrix = []
        reply_char_matrix = []

        count = 0

        for line in data_file:
            count += 1
            utterances = line.strip().split('\t')

            while utterances.count(''):
                utterances.remove('')

            utterance_word_id_list = []
            utterance_feature_list = []
            query_word_id_list = []
            query_feature_list = []
            reply_word_id_list = []
            reply_feature_list = []
            context_word_candidate = []

            # for char embedding
            utt_word_char_id_matrix = []  # (turn_num, turn_len, max_word_char_num)
            query_word_char_id_list = []
            reply_word_char_id_list = []

            reply_words = utterances[-1].split(' ')

            for idx in range(1, len(utterances)):
                utterance = utterances[idx]
                utterance_tokens = utterance.split(' ')

                while utterance_tokens.count(' '):
                    utterance_tokens.remove(' ')
                while utterance_tokens.count(''):
                    utterance_tokens.remove('')

                # deal with context
                if idx < len(utterances) - 1:
                    utterance_id_list = word2id_list(utterance_tokens, word2id_dict)
                    feature_list = generate_word_match_feature(utterance_tokens, reply_words, stopword_list)

                    context_word_candidate.extend(utterance_tokens)

                    if config.use_char_embeddings:
                        utt_word_char_id_list = []
                        for word in utterance_tokens:
                            utt_word_char_id_list.append(word2char_id_list(word, char2id_dict))
                        while len(utt_word_char_id_list) < config.max_utterance_len:
                            if padding_type == 'pre':
                                utt_word_char_id_list.insert(0, np.zeros(config.max_token_len))
                            else:
                                utt_word_char_id_list.append(np.zeros(config.max_token_len))

                        if len(utt_word_char_id_list) > config.max_utterance_len:
                            utt_word_char_id_list = utt_word_char_id_list[-1 * config.max_utterance_len:]
                        utt_word_char_id_list = pad_sequences(utt_word_char_id_list, maxlen=config.max_token_len)

                    if len(utterance_word_id_list) < config.max_utterance_num:
                        utterance_word_id_list.append(utterance_id_list)
                        utterance_feature_list.append(feature_list)
                        if config.use_char_embeddings:
                            utt_word_char_id_matrix.append(utt_word_char_id_list)

                    else:
                        utterance_word_id_list.pop(0)
                        utterance_feature_list.pop(0)
                        utterance_word_id_list.append(utterance_id_list)
                        utterance_feature_list.append(feature_list)

                        if config.use_char_embeddings:
                            utt_word_char_id_matrix.pop(0)
                            utt_word_char_id_matrix.append(utt_word_char_id_list)

                    # handle query
                    if idx == len(utterances) - 2:
                        query_word_id_list = utterance_id_list
                        query_feature_list = feature_list
                        if config.use_char_embeddings:
                            query_word_char_id_list = utt_word_char_id_list

                # deal with reply
                else:
                    reply_word_id_list = word2id_list(utterance_tokens, word2id_dict)
                    reply_feature_list = generate_word_match_feature(utterance_tokens,
                                                                     context_word_candidate,
                                                                     stopword_list)

                    if config.use_char_embeddings:
                        for word in utterance_tokens:
                            reply_word_char_id_list.append(word2char_id_list(word, char2id_dict))
                        while len(reply_word_char_id_list) < config.max_utterance_len:
                            reply_word_char_id_list.insert(0, np.zeros(config.max_token_len))
                        if len(reply_word_char_id_list) > config.max_utterance_len:
                            reply_word_char_id_list = reply_word_char_id_list[-1 * config.max_utterance_len:]
                        reply_word_char_id_list = pad_sequences(reply_word_char_id_list, maxlen=config.max_token_len)

            label = int(utterances[0])

            # padding utterance
            while len(utterance_word_id_list) < config.max_utterance_num:
                utterance_word_id_list.append(np.zeros(config.max_utterance_len))
                if config.use_char_embeddings:
                    utt_word_char_id_matrix.append(np.zeros((config.max_utterance_len,
                                                             config.max_token_len)))
            utterance_word_id_list = pad_sequences(utterance_word_id_list,
                                                   config.max_utterance_len, padding=padding_type)
            utterance_word_id_list = np.reshape(utterance_word_id_list,
                                                config.max_utterance_num * config.max_utterance_len)

            while len(utterance_feature_list) < config.max_utterance_num:
                utterance_feature_list.append(np.zeros(config.max_utterance_len))
            utterance_feature_list = pad_sequences(utterance_feature_list,
                                                   maxlen=config.max_utterance_len, padding=padding_type)
            utterance_feature_list = np.reshape(utterance_feature_list,
                                                config.max_utterance_num * config.max_utterance_len)

            context_matrix.append(utterance_word_id_list)
            context_feature_matrix.append(utterance_feature_list)
            query_matrix.append(query_word_id_list)
            query_feature_matrix.append(query_feature_list)
            reply_matrix.append(reply_word_id_list)
            reply_feature_matrix.append(reply_feature_list)
            y_matrix.append(label)

            if config.use_char_embeddings:
                context_char_matrix.append(utt_word_char_id_matrix)
                query_char_matrix.append(query_word_char_id_list)
                reply_char_matrix.append(reply_word_char_id_list)

            if count % batch_size == 0:
                context_word_input = np.asarray(context_matrix)
                context_feature_input = np.asarray(context_feature_matrix)
                query_word_input = pad_sequences(query_matrix, config.max_utterance_len,
                                                 padding=padding_type)
                query_feature_input = pad_sequences(query_feature_matrix, config.max_utterance_len,
                                                    padding=padding_type)
                reply_word_input = pad_sequences(reply_matrix, config.max_utterance_len,
                                                 padding=padding_type)
                reply_feature_input = pad_sequences(reply_feature_matrix, config.max_utterance_len,
                                                    padding=padding_type)
                y_input = np.asarray(y_matrix)

                if config.use_char_embeddings:
                    context_char_input = np.asarray(context_char_matrix)
                    query_char_input = np.asarray(query_char_matrix)
                    reply_char_input = np.asarray(reply_char_matrix)

                input_dict = {'context_feature_input': context_feature_input,
                              'query_feature_input': query_feature_input,
                              'reply_feature_input': reply_feature_input}
                if config.use_word_embeddings:
                    word_input_dict = {'context_word_input': context_word_input,
                                       'query_word_input': query_word_input,
                                       'reply_word_input': reply_word_input}
                    input_dict.update(word_input_dict)
                if config.use_char_embeddings:
                    char_input_dict = {'context_char_input': context_char_input,
                                       'query_char_input': query_char_input,
                                       'reply_char_input': reply_char_input}
                    input_dict.update(char_input_dict)
                if is_train:
                    yield (input_dict, {'score': y_input})
                else:
                    yield input_dict

                context_matrix = []
                context_feature_matrix = []
                query_matrix = []
                query_feature_matrix = []
                reply_matrix = []
                reply_feature_matrix = []
                y_matrix = []

                # for char embedding
                context_char_matrix = []
                query_char_matrix = []
                reply_char_matrix = []

        data_file.close()
        if not is_train:
            break
