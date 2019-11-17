from keras.layers import Input, CuDNNLSTM, LSTM, Dropout, Dense, Activation, Permute, \
    BatchNormalization, Reshape, Lambda, GlobalMaxPooling1D, Embedding, Conv1D
from keras.models import Model
import tensorflow as tf
from keras.layers.wrappers import Bidirectional, TimeDistributed
import keras.layers as ly
from keras import backend as K
from preprocess import load_word_embeddings, load_vocab

class TripleNetModel(object):

    def __init__(self, config):
        self.config = config
        self.embedding_dim = 0
        self.input_list = []
        if self.config.use_CuDNNRNN:
            self.RNN = CuDNNLSTM
        else:
            self.RNN = LSTM
        self.shared_encoding_layer = Bidirectional(self.RNN(units=self.config.hidden_dim, return_sequences=True, \
            input_shape=(self.config.max_utterance_len, self.embedding_dim)), merge_mode='concat')
        self.model = self.init_model()

    def init_model(self):
        if self.config.use_char_embeddings:
            context_emb, context_char_emb, query_emb, query_char_emb, reply_emb, reply_char_emb = self.embedding_layer()
        else:
            context_emb, query_emb, reply_emb = self.embedding_layer()

        context_word_repr = TimeDistributed(self.shared_encoding_layer)(context_emb)
        context_word_repr = Dropout(0.2)(context_word_repr)
        query_word_repr = self.shared_encoding_layer(query_emb)
        reply_word_repr = self.shared_encoding_layer(reply_emb)

        if self.config.use_char_embeddings:
            context_char_mix_repr, query_char_mix_repr, reply_char_mix_repr = self.triple_attention_layer(
            Reshape((self.config.max_utterance_num*self.config.max_utterance_len, self.config.char_features_dim))(context_char_emb),
            query_char_emb, reply_char_emb, dim=self.config.char_features_dim)

        context_word_mix_repr, query_word_mix_repr, reply_word_mix_repr = self.triple_attention_layer(
            Reshape((self.config.max_utterance_num*self.config.max_utterance_len, 2*self.config.hidden_dim))(context_word_repr),
            query_word_repr, reply_word_repr, 2 * self.config.hidden_dim)

        # utterance repr
        context_utt_repr = self.self_attention_sentence_embedding(context_word_repr)

        context_utt_mix_repr, query_utt_mix_repr, reply_utt_mix_repr = self.triple_attention_layer( \
            context_utt_repr, query_word_repr, reply_word_repr, 2 * self.config.hidden_dim)

        # context level repr
        context_history_repr = Bidirectional(self.RNN(units=self.config.hidden_dim, return_sequences=True), \
            merge_mode='concat')(context_utt_repr)

        context_history_mix_repr, query_history_mix_repr, reply_history_mix_repr = self.triple_attention_layer( \
        context_history_repr, query_word_repr, reply_word_repr, 2 * self.config.hidden_dim)

        #triple matching
        if self.config.use_char_embeddings:
            reply_context_char_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_char_mix_repr,\
                context_char_mix_repr])
            reply_context_char_match = Reshape((self.config.max_utterance_len, self.config.max_utterance_num, \
                self.config.max_utterance_len))(reply_context_char_match)
            reply_context_char_match = TimeDistributed(Permute((2, 1)))(reply_context_char_match)
            reply_context_char_match = TimeDistributed(GlobalMaxPooling1D())(reply_context_char_match)
            reply_query_char_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_char_mix_repr, query_char_mix_repr])

        reply_context_word_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_word_mix_repr,context_word_mix_repr])
        reply_context_word_match = Reshape((self.config.max_utterance_len, self.config.max_utterance_num, \
            self.config.max_utterance_len))(reply_context_word_match)
        reply_context_word_match = TimeDistributed(Permute((2, 1)))(reply_context_word_match)
        reply_context_word_match = TimeDistributed(GlobalMaxPooling1D())(reply_context_word_match)

        reply_context_utt_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_utt_mix_repr, context_utt_mix_repr])
        reply_context_history_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_history_mix_repr, \
            context_history_mix_repr])

        reply_query_word_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_word_mix_repr, query_word_mix_repr])
        reply_query_utt_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_utt_mix_repr, query_utt_mix_repr])
        reply_query_history_match = ly.dot(axes=(2, 2), normalize=True, inputs=[reply_history_mix_repr, query_history_mix_repr])

        if self.config.use_char_embeddings:
            char_match = ly.concatenate(inputs=[reply_context_char_match, reply_query_char_match])
        word_match = ly.concatenate(inputs=[reply_context_word_match, reply_query_word_match])
        utt_match = ly.concatenate(inputs=[reply_context_utt_match, reply_query_utt_match])
        history_match = ly.concatenate(inputs=[reply_context_history_match, reply_query_history_match])

        # fusion
        if self.config.use_char_embeddings:
            char_match = Reshape((self.config.max_utterance_len, self.config.max_utterance_num + self.config.max_utterance_len,\
                1))(char_match)
        word_match = Reshape((self.config.max_utterance_len, self.config.max_utterance_num + self.config.max_utterance_len, 1))(word_match)
        utt_match = Reshape((self.config.max_utterance_len, self.config.max_utterance_num + self.config.max_utterance_len, 1))(utt_match)
        history_match = Reshape((self.config.max_utterance_len, self.config.max_utterance_num + self.config.max_utterance_len,\
            1))(history_match)
        if self.config.use_char_embeddings:
            fusion_input = ly.concatenate(inputs=[char_match, word_match, utt_match, history_match])
        else:
            fusion_input = ly.concatenate(inputs=[word_match, utt_match, history_match])

        fusion_result = self.fusion_layer(fusion_input)

        # prediction
        score = Activation('sigmoid', name='score')(fusion_result)

        model = Model(inputs=self.input_list, outputs=score)
        model.summary()
        return model

    def embedding_layer(self):
        context_emb = None
        query_emb = None
        reply_emb = None
        if self.config.use_word_embeddings:
            self.embedding_dim += self.config.word_embeddings_dim
            context_word_input = Input(shape=(self.config.max_utterance_num * self.config.max_utterance_len,),
                dtype='int32', name='context_word_input')
            query_word_input = Input(shape=(self.config.max_utterance_len,), dtype='int32',
                name='query_word_input')
            reply_word_input = Input(shape=(self.config.max_utterance_len,), dtype='int32',
                name='reply_word_input')

            self.input_list.extend([context_word_input, query_word_input, reply_word_input])

            # load word embeddings
            emb_matrix = load_word_embeddings(self.config, self.config.word_embeddings_file, load_vocab(self.config.word_vocab_file))
            word_emb_layer = Embedding(input_dim=self.config.word_vocab_size, output_dim=self.config.word_embeddings_dim,
                weights=[emb_matrix], trainable=True)
            context_word_emb = word_emb_layer(context_word_input)
            context_word_emb = Reshape((self.config.max_utterance_num, self.config.max_utterance_len, \
                self.config.word_embeddings_dim))(context_word_emb)
            query_word_emb = word_emb_layer(query_word_input)
            reply_word_emb = word_emb_layer(reply_word_input)

            context_emb = context_word_emb
            query_emb = query_word_emb
            reply_emb = reply_word_emb
        if self.config.use_char_embeddings:
            self.embedding_dim += self.config.char_features_dim
            context_char_input = Input(shape=(self.config.max_utterance_num, self.config.max_utterance_len,
                self.config.max_token_len), dtype='int32', name='context_char_input')
            query_char_input = Input(shape=(self.config.max_utterance_len, self.config.max_token_len),
                dtype='int32', name='query_char_input')
            reply_char_input = Input(shape=(self.config.max_utterance_len, self.config.max_token_len),
                dtype='int32', name='reply_char_input')

            self.input_list.extend([context_char_input, query_char_input, reply_char_input])

            char_emb_layer = Embedding(input_dim=self.config.char_vocab_size, output_dim=self.config.char_embeddings_dim)
            context_char_emb = Reshape((self.config.max_utterance_num * self.config.max_utterance_len * self.config.max_token_len,\
                ))(context_char_input)
            query_char_emb = Reshape((self.config.max_utterance_len * self.config.max_token_len,))(query_char_input)
            reply_char_emb = Reshape((self.config.max_utterance_len * self.config.max_token_len,))(reply_char_input)
            context_char_emb = char_emb_layer(context_char_emb)
            query_char_emb = char_emb_layer(query_char_emb)
            reply_char_emb = char_emb_layer(reply_char_emb)

            char_cnn_layer = Conv1D(filters=self.config.char_features_dim, kernel_size=self.config.char_kernel_shape,\
                activation='tanh')
            context_char_emb = Reshape((self.config.max_utterance_num * self.config.max_utterance_len, \
                self.config.max_token_len, self.config.char_embeddings_dim))(context_char_emb)
            context_char_emb = TimeDistributed(char_cnn_layer)(context_char_emb)
            context_char_emb = TimeDistributed(GlobalMaxPooling1D())(context_char_emb)
            context_char_emb = Reshape((self.config.max_utterance_num, self.config.max_utterance_len,
                self.config.char_features_dim))(context_char_emb)

            query_char_emb = Reshape((self.config.max_utterance_len, self.config.max_token_len, \
                self.config.char_embeddings_dim))(query_char_emb)
            query_char_emb = TimeDistributed(char_cnn_layer)(query_char_emb)
            query_char_emb = TimeDistributed(GlobalMaxPooling1D())(query_char_emb)
            query_char_emb = Reshape((self.config.max_utterance_len, self.config.char_features_dim))(query_char_emb)

            reply_char_emb = Reshape((self.config.max_utterance_len, self.config.max_token_len, \
                self.config.char_embeddings_dim))(reply_char_emb)
            reply_char_emb = TimeDistributed(char_cnn_layer)(reply_char_emb)
            reply_char_emb = TimeDistributed(GlobalMaxPooling1D())(reply_char_emb)
            reply_char_emb = Reshape((self.config.max_utterance_len, self.config.char_features_dim))(reply_char_emb)

            if context_emb is not None:
                context_emb = ly.concatenate(inputs=[context_emb, context_char_emb])
                query_emb = ly.concatenate(inputs=[query_emb, query_char_emb])
                reply_emb = ly.concatenate(inputs=[reply_emb, reply_char_emb])
            else:
                context_emb = context_char_emb
                query_emb = query_char_emb
                reply_emb = reply_char_emb

        self.embedding_dim += 1
        context_feature_input = Input(shape=(self.config.max_utterance_num * self.config.max_utterance_len,),\
            dtype='float32', name='context_feature_input')
        query_feature_input = Input(shape=(self.config.max_utterance_len,), \
            dtype='float32', name='query_feature_input')
        reply_feature_input = Input(shape=(self.config.max_utterance_len,),\
            dtype='float32', name='reply_feature_input')

        self.input_list.extend([context_feature_input, query_feature_input, reply_feature_input])
        #assert context_emb is not None

        reply_emb = Dropout(self.config.dropout_rate)(reply_emb)
        context_feature = Reshape((self.config.max_utterance_num, self.config.max_utterance_len, 1))(context_feature_input)
        context_emb = ly.concatenate(inputs=[context_emb, context_feature])
        query_feature = Reshape((self.config.max_utterance_len, 1))(query_feature_input)
        query_emb = ly.concatenate(inputs=[query_emb, query_feature])
        reply_feature = Reshape((self.config.max_utterance_len, 1))(reply_feature_input)
        reply_emb = ly.concatenate(inputs=[reply_emb, reply_feature])

        if self.config.use_char_embeddings:
            return context_emb, context_char_emb, query_emb, query_char_emb, reply_emb, reply_char_emb
        else:
            return context_emb, query_emb, reply_emb


    def bi_directional_attention_fuction(self, context_repr, query_repr, dim, reverse=True):
        context_query_dot = ly.dot(axes=(2, 2), inputs=[context_repr, Dense(dim, activation='tanh')(query_repr)])
        context_query_attention = Activation('softmax')(context_query_dot)
        query_to_context = ly.dot(axes=(2, 1), inputs=[context_query_attention, query_repr])
        context_query_sub = ly.subtract(inputs=[context_repr, query_to_context])
        context_query_sub = BatchNormalization()(context_query_sub)
        if reverse:
            query_context_dot = Permute((2, 1))(context_query_dot)
            query_context_attention = Activation('softmax')(query_context_dot)
            query_context_attention = ly.dot(axes=(2, 1), inputs=[query_context_attention, context_repr])
            query_context_sub = ly.subtract(inputs=[query_repr, query_context_attention])
            query_context_sub = BatchNormalization()(query_context_sub)
            return context_query_sub, query_context_sub
        else:
            return context_query_sub


    def triple_attention_layer(self, context_repr, query_repr, reply_repr, dim):
        context_query_sub, query_context_sub = self.bi_directional_attention_fuction(context_repr, query_repr, dim=dim)
        context_reply_sub, reply_context_sub = self.bi_directional_attention_fuction(context_repr, reply_repr, dim=dim)
        reply_query_sub, query_reply_sub = self.bi_directional_attention_fuction(reply_repr, query_repr, dim=dim)

        context_fusion_repr = ly.add(inputs=[context_query_sub, context_reply_sub])
        context_fusion_repr = BatchNormalization()(context_fusion_repr)
        query_fusion_repr = ly.add(inputs=[query_reply_sub, query_context_sub])
        query_fusion_repr = BatchNormalization()(query_fusion_repr)
        reply_fusion_repr = ly.add(inputs=[reply_query_sub, reply_context_sub])
        reply_fusion_repr = BatchNormalization()(reply_fusion_repr)

        return context_fusion_repr, query_fusion_repr, reply_fusion_repr

    def time_distributed_dot_layer(self, x):
        def time_distributed_dot(x):
            x1 = x[0]
            x2 = x[1]

            batch_dim = x1.shape[0]
            first_dim = x1.shape[1]
            second_dim = x1.shape[2]
            hidden_dim = x1.shape[3]

            x = K.reshape(x1, (-1, second_dim, hidden_dim))
            y = K.reshape(x2, (-1, second_dim))
            xy = K.batch_dot(x, y, axes=(1, 1))
            xy = K.reshape(xy, (-1, first_dim, hidden_dim))
            return xy

        def time_distributed_dot_output(input_shape):
            return tuple((input_shape[0][0], input_shape[0][1], input_shape[0][3]))

        return Lambda(function=time_distributed_dot, output_shape=time_distributed_dot_output)(x)

    def self_attention_sentence_embedding(self, utterances):
        """
        :param utterances: context_word_repr
        :return: context_utterance_repr
        """

        W_s1 = TimeDistributed(Dense(units=100, use_bias=False, activation='tanh'))
        self_attn = TimeDistributed(W_s1)(utterances)
        W_s2 = TimeDistributed(Dense(1, use_bias=False))
        self_attn = TimeDistributed(W_s2)(self_attn)
        self_attn = Reshape((self.config.max_utterance_num, self.config.max_utterance_len))(self_attn)
        self_attn = TimeDistributed(Activation('softmax'))(self_attn)
        context_utterance_repr = self.time_distributed_dot_layer([utterances, self_attn])
        return context_utterance_repr

    def fusion_layer(self, fusion_input):
         # TODO
        multi_repr = TimeDistributed(Bidirectional(self.RNN(units=self.config.hidden_dim, return_sequences=True), \
            merge_mode='concat'))(fusion_input)
        multi_repr = TimeDistributed(GlobalMaxPooling1D())(multi_repr)
        fusion_output = Bidirectional(self.RNN(units=self.config.hidden_dim, return_sequences=True),\
            merge_mode='concat')(multi_repr)
        fusion_output = GlobalMaxPooling1D()(fusion_output)
        fusion_output = Dense(1)(fusion_output)
        return fusion_output
