# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: linking_model.py

@time: 2019/5/9 20:06

@desc:

"""

from itertools import groupby
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback
from models.base_model import BaseModel
from utils.other import pad_sequences_1d
from layers.attention import InteractiveAttention, SingleSideAttention


class LinkMetrics(Callback):
    def __init__(self, link_model, valid_data, vocab, mention_to_entity, entity_desc, max_desc_len, max_erl_len,
                 use_relative_pos):
        super(LinkMetrics, self).__init__()
        self.link_model = link_model
        self.valid_data = valid_data
        self.vocab = vocab
        self.mention_to_entity = mention_to_entity
        self.entity_desc = entity_desc
        self.max_desc_len = max_desc_len
        self.max_erl_len = max_erl_len
        self.use_relative_pos = use_relative_pos

        self.text_data, self.pred_mentions, self.gold_mention_entities = [], [], []
        for data in self.valid_data:
            self.text_data.append(data['text'])
            self.pred_mentions.append(data['mention_data'])
            self.gold_mention_entities.append(data['mention_data'])

    def evaluate(self, text_data, pred_mentions, gold_mention_entities):
        match, n_true, n_pred = 1e-10, 1e-10, 1e-10
        for text, pred_mention, gold_mention_entity in zip(text_data, pred_mentions, gold_mention_entities):
            pred = set(LinkModel.entity_link(text, pred_mention, self.link_model, self.vocab, self.mention_to_entity,
                                             self.entity_desc, self.max_desc_len, self.max_erl_len,
                                             self.use_relative_pos))
            true = set(gold_mention_entity)
            match += len(pred & true)
            n_pred += len(pred)
            n_true += len(true)

        r = match / n_true
        p = match / n_pred
        f1 = 2 * match / (n_pred + n_true)
        return r, p, f1

    def on_epoch_end(self, epoch, logs=None):
        r, p, f1 = self.evaluate(self.text_data, self.pred_mentions, self.gold_mention_entities)
        logs['val_r'] = r
        logs['val_p'] = p
        logs['val_f1'] = f1
        print('Epoch {}: val_r: {}, val_p: {}, val_f1: {}'.format(epoch, r, p, f1))


class LinkModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(LinkModel, self).__init__(config)
        self.model, self.link_model = self.build(**kwargs)
        if 'swa' in self.config.callbacks_to_add or 'swa_clr' in self.config.callbacks_to_add:
            self.swa_model, _ = self.build(**kwargs)

    def add_metrics(self, valid_data):
        self.callbacks.append(LinkMetrics(self.link_model, valid_data, self.config.vocab, self.config.mention_to_entity,
                                          self.config.entity_desc, self.config.max_desc_len, self.config.max_erl_len,
                                          self.config.use_relative_pos))
        print('Logging Info - Callback Added: Metrics...')

    def build(self, score_func='cosine', margin=0.04, max_mention=False, avg_mention=False, add_cnn=None,
              encoder_type='self_attend_max', ent_attend_type='add'):

        '''1. prepare input'''
        model_inputs = []
        link_model_inputs = []

        input_erl_text = Input(shape=(None,))
        model_inputs.append(input_erl_text)
        link_model_inputs.append(input_erl_text)

        input_begin = Input(shape=(1,))
        input_end = Input(shape=(1,))
        model_inputs.extend([input_begin, input_end])
        link_model_inputs.extend([input_begin, input_end])

        if self.config.use_relative_pos:
            input_relative_pos = Input(shape=(None,))
            model_inputs.append(input_relative_pos)
            link_model_inputs.append(input_relative_pos)

        input_pos_desc = Input(shape=(None,))
        input_neg_desc = Input(shape=(None,))
        model_inputs.extend([input_pos_desc, input_neg_desc])
        link_model_inputs.append(input_pos_desc)

        # CUDALSTM (or CNN) doesn't support masking, so we don't use mask_zero in embedding layer, instead we apply
        # masking on our own
        get_mask_layer = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), K.floatx()))
        erl_text_mask = get_mask_layer(input_erl_text)
        pos_desc_mask = get_mask_layer(input_pos_desc)
        neg_desc_mask = get_mask_layer(input_neg_desc)
        apply_mask_layer = Lambda(lambda x: x[0] * x[1])

        '''2. prepare embedding'''
        if self.config.embeddings is not None:
            embedding_layer = Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embed_dim,
                                        weights=[self.config.embeddings], trainable=self.config.embed_trainable)
        else:
            embedding_layer = Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embed_dim)

        erl_text_embed = SpatialDropout1D(0.2)(embedding_layer(input_erl_text))
        if self.config.use_relative_pos:
            rel_pos_embedding_layer = Embedding(input_dim=self.config.n_rel_pos_embed,
                                                output_dim=self.config.rel_pos_embed_dim)
            rel_pos_embed = rel_pos_embedding_layer(input_relative_pos)
            erl_text_embed = concatenate([erl_text_embed, rel_pos_embed])
        erl_text_embed = apply_mask_layer([erl_text_embed, erl_text_mask])

        pos_desc_embed = SpatialDropout1D(0.2)(embedding_layer(input_pos_desc))
        neg_desc_embed = SpatialDropout1D(0.2)(embedding_layer(input_neg_desc))
        pos_desc_embed = apply_mask_layer([pos_desc_embed, pos_desc_mask])
        neg_desc_embed = apply_mask_layer([neg_desc_embed, neg_desc_mask])

        '''3. encode mention & entity representation'''
        if add_cnn == 'before':
            erl_text_embed = Conv1D(filters=self.config.embed_dim, kernel_size=3, padding='same',
                                    activation='relu')(erl_text_embed)
        erl_text_lstm = Bidirectional(CuDNNLSTM(units=self.config.embed_dim // 2,
                                                return_sequences=True))(erl_text_embed)
        if add_cnn == 'after':
            erl_text_lstm = Conv1D(filters=self.config.embed_dim, kernel_size=3, padding='same',
                                   activation='relu')(erl_text_lstm)
        erl_text_lstm = apply_mask_layer([erl_text_lstm, erl_text_mask])

        if add_cnn == 'before':
            ent_cnn_layer = Conv1D(filters=self.config.embed_dim, kernel_size=3, padding='same', activation='relu')
            pos_desc_embed = ent_cnn_layer(pos_desc_embed)
            neg_desc_embed = ent_cnn_layer(neg_desc_embed)
        ent_lstm_layer = Bidirectional(CuDNNLSTM(units=self.config.embed_dim // 2, return_sequences=True))
        pos_desc_lstm = ent_lstm_layer(pos_desc_embed)
        neg_desc_lstm = ent_lstm_layer(neg_desc_embed)
        if add_cnn == 'after':
            ent_cnn_layer = Conv1D(filters=self.config.embed_dim, kernel_size=3, padding='same', activation='relu')
            pos_desc_lstm = ent_cnn_layer(pos_desc_lstm)
            neg_desc_lstm = ent_cnn_layer(neg_desc_lstm)
        pos_desc_lstm = apply_mask_layer([pos_desc_lstm, pos_desc_mask])
        neg_desc_lstm = apply_mask_layer([neg_desc_lstm, neg_desc_mask])

        if encoder_type in ['self_attend_max', 'self_attend_single_attend']:
            '''mention presentation based on self_attention, first token & last token, max or avg pooling (optional)'''
            # first token & last token of mention
            index_layer = Lambda(
                lambda x: tf.gather_nd(x[0], tf.concat([tf.expand_dims(tf.range(tf.shape(x[0])[0]), 1),
                                                        tf.to_int32(x[1])], axis=1)))
            mention_begin_embed = index_layer([erl_text_lstm, input_begin])
            mention_end_embed = index_layer([erl_text_lstm, input_end])
            mention_spand_embed, mention_index, mention_mask = Lambda(self.span_index)(
                [erl_text_lstm, input_begin, input_end])

            # soft head attention
            head_score = TimeDistributed(Dense(1, activation='tanh'))(erl_text_lstm)
            mention_head_score = Lambda(lambda x: tf.squeeze(tf.gather_nd(x[0], tf.to_int32(x[1])), 2))(
                [head_score, mention_index])  # [batch_size, max_mention_length]
            # self attention
            mention_head_score = Lambda(self.softmax_with_mask)([mention_head_score, mention_mask])
            mention_attention = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], 2), 1))(
                [mention_spand_embed, mention_head_score])
            mention_embed = concatenate([mention_begin_embed, mention_end_embed, mention_attention])

            # max pooling & avg pooling
            if max_mention:
                mention_max_embed = Lambda(lambda x: K.max(x[0] - (1 - K.expand_dims(x[1], 2)) * 1e10, 1))(
                    [mention_spand_embed, mention_mask])
                mention_embed = concatenate([mention_embed, mention_max_embed])
            if avg_mention:
                mention_avg_embed = Lambda(lambda x: K.sum(x[0], 1) / K.sum(x[1], 1, keepdims=True))(
                    [mention_spand_embed, mention_mask])
                mention_embed = concatenate([mention_embed, mention_avg_embed])
            mention_embed = Dense(self.config.embed_dim, activation='relu')(mention_embed)  # [batch_size, embed_dim]
            mention_pos_embed = mention_embed
            mention_neg_embed = mention_embed

            if encoder_type == 'self_attend_max':
                '''entity representation based on max pooling'''
                pos_embed = Lambda(self.seq_maxpool)([pos_desc_lstm, pos_desc_mask])
                neg_embed = Lambda(self.seq_maxpool)([neg_desc_lstm, neg_desc_mask])
            else:
                '''entity representation based on single sided attention using mention representation as query'''
                ent_attend_layer = SingleSideAttention(ent_attend_type)
                pos_embed = ent_attend_layer([mention_embed, pos_desc_lstm])
                neg_embed = ent_attend_layer([mention_embed, neg_desc_lstm])
        elif encoder_type in ['co_attend', 'max_co_attend']:
            attend_layer = InteractiveAttention(attend_type=encoder_type)
            mention_pos_embed, pos_embed = attend_layer([erl_text_lstm, pos_desc_lstm])
            mention_neg_embed, neg_embed = attend_layer([erl_text_lstm, neg_desc_lstm])
        else:
            raise ValueError('encoder_type not understood'.format(encoder_type))
        if score_func == 'dense':
            hidden_layer = Dense(self.config.embed_dim, activation='relu')
            score_layer = Dense(1, activation='sigmoid')
            pos_score = score_layer(hidden_layer(concatenate([mention_pos_embed, pos_embed,
                                                              multiply([mention_pos_embed, pos_embed]),
                                                              subtract([mention_pos_embed, pos_embed])])))
            neg_score = score_layer(hidden_layer(concatenate([mention_neg_embed, neg_embed,
                                                              multiply([mention_neg_embed, neg_embed]),
                                                              subtract([mention_neg_embed, neg_embed])])))
        else:
            score_layer = self.get_score_layer(score_func)
            pos_score = score_layer([mention_pos_embed, pos_embed])
            neg_score = score_layer([mention_pos_embed, neg_embed])
        link_model = Model(link_model_inputs, pos_score)
        loss = K.mean(K.relu(margin + neg_score - pos_score))
        train_model = Model(model_inputs, [pos_score, neg_score])
        train_model.add_loss(loss)
        train_model.compile(optimizer=self.config.optimizer)
        return train_model, link_model

    def train(self, train_generator, valid_data):
        self.callbacks = []
        self.add_metrics(valid_data)
        self.init_callbacks(train_generator.data_size)

        print('Logging Info - Start training...')
        self.model.fit_generator(generator=train_generator, epochs=self.config.n_epoch, callbacks=self.callbacks)
        print('Logging Info - Training end...')

    @staticmethod
    def entity_link(text, pred_mention, link_model, vocab, mention_to_entity, entity_desc, max_desc_len,
                    max_erl_len, use_relative_pos):
        link_result = []

        text_ids = [vocab.get(token, 1) for token in text]

        cand_mention_entity = {}
        ent_owner = []
        ent_desc, begin, end, relative_pos = [], [], [], []
        for mention in pred_mention:
            if mention not in cand_mention_entity:  # there might be duplicated mention in pred_mention
                cand_mention_entity[mention] = mention_to_entity.get(mention[0], [])
                _begin = mention[1]
                _end = _begin + len(mention[0]) - 1
                if use_relative_pos:
                    _rel_pos = []
                    for i in range(len(text_ids)):
                        if i < _begin:
                            _rel_pos.append(max(i - _begin + max_erl_len, 1))  # transform to id
                        elif _begin <= i <= _end:
                            _rel_pos.append(0 + max_erl_len)
                        else:
                            _rel_pos.append(min(i - _end + max_erl_len, 2 * max_erl_len - 1))
                for ent_id in cand_mention_entity[mention]:
                    desc = entity_desc[ent_id]
                    desc_ids = [vocab.get(token, 1) for token in desc]
                    ent_desc.append(desc_ids)

                    begin.append([_begin])
                    end.append([_end])
                    if use_relative_pos:
                        relative_pos.append(_rel_pos)

                    ent_owner.append(mention)

        if ent_desc:
            model_inputs = []
            repeat_text_ids = np.repeat([text_ids], len(ent_desc), 0)
            model_inputs.append(repeat_text_ids)
            begin = np.array(begin)
            end = np.array(end)
            model_inputs.extend([begin, end])
            if use_relative_pos:
                relative_pos = pad_sequences_1d(relative_pos)
                model_inputs.append(relative_pos)
            ent_desc = pad_sequences_1d(ent_desc, max_desc_len)
            model_inputs.append(ent_desc)

            scores = link_model.predict(model_inputs)[:, 0]
            for k, v in groupby(zip(ent_owner, scores), key=lambda x: x[0]):
                score_to_rank = np.array([j[1] for j in v])
                ent_id = cand_mention_entity[k][np.argmax(score_to_rank)]
                link_result.append((k[0], k[1], ent_id))
        return link_result

    def evaluate(self, text_data, pred_mentions, gold_mention_entities):
        assert len(text_data) == len(pred_mentions) == len(gold_mention_entities)
        match, n_true, n_pred = 1e-10, 1e-10, 1e-10
        for text, pred_mention, gold_mention_entity in zip(text_data, pred_mentions, gold_mention_entities):
            pred = set(self.entity_link(text, pred_mention, self.link_model, self.config.vocab,
                                        self.config.mention_to_entity, self.config.entity_desc,
                                        self.config.max_desc_len, self.config.max_erl_len, self.config.use_relative_pos))
            true = set(gold_mention_entity)
            match += len(pred & true)
            n_pred += len(pred)
            n_true += len(true)

        r = match / n_true
        p = match / n_pred
        f1 = 2 * match / (n_pred + n_true)
        print('Logging Info - Recall: {}, Precision: {}, F1: {}'.format(r, p, f1))
        return r, p, f1

    def predict(self, text_data, pred_mentions):
        assert len(text_data) == len(pred_mentions)
        return [self.entity_link(text, pred_mention, self.link_model, self.config.vocab,
                                 self.config.mention_to_entity, self.config.entity_desc,
                                 self.config.max_desc_len, self.config.max_erl_len, self.config.use_relative_pos)
                for text, pred_mention in zip(text_data, pred_mentions)]

    @staticmethod
    def span_index(x):
        text_embed, begin, end = x
        mention_length = end - begin + 1  # [batch_size, 1]
        max_mention_length = tf.reduce_max(mention_length)
        mention_index = tf.range(max_mention_length) + begin  # [batch_size, max_mention_length]
        mention_mask = tf.cast(tf.less_equal(mention_index, end), tf.float32)
        mention_index = tf.minimum(mention_index, end)  # [batch_size, max_mention_length]

        batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(text_embed)[0]), 1), [1, tf.shape(mention_index)[1]])
        mention_index = tf.stack([batch_index, tf.to_int32(mention_index)], axis=2)
        span_mention = tf.gather_nd(text_embed, mention_index)
        return [span_mention, mention_index, mention_mask]

    @staticmethod
    def softmax_with_mask(x):
        score, mask = x
        score = K.exp(score)
        score *= mask
        return score / (K.sum(score, axis=1, keepdims=True) + K.epsilon())

    @staticmethod
    def seq_maxpool(x):
        # maxpooling with masking
        seq, mask = x
        seq -= (1 - mask) * 1e10
        return K.max(seq, 1)

    @staticmethod
    def get_score_layer(score_func='cosine', gamma=1, c=1, d=2):
        '''
        cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        polynomial: (gamma * dot(a, b) + c) ^ d
        sigmoid: tanh(gamma * dot(a, b) + c)
        rbf: exp(-gamma * l2_norm(a-b) ^ 2)
        euclidean: 1 / (1 + l2_norm(a - b))
        exponential: exp(-gamma * l2_norm(a - b))
        gesd: euclidean * sigmoid
        aesd: (euclidean + sigmoid) / 2
        '''

        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        l2_norm = lambda a, b: K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))

        if score_func == 'cosine':
            return Lambda(
                lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon()))
        elif score_func == 'polynomial':
            return Lambda(lambda x: (gamma * dot(x[0], x[1]) + c) ** d)
        elif score_func == 'sigmoid':
            return Lambda(lambda x: K.tanh(gamma * dot(x[0], x[1]) + c))
        elif score_func == 'rbf':
            return Lambda(lambda x: K.exp(-1 * gamma * l2_norm(x[0], x[1]) ** 2))
        elif score_func == 'euclidean':
            return Lambda(lambda x: 1 / (1 + l2_norm(x[0], x[1])))
        elif score_func == 'exponential':
            return Lambda(lambda x: K.exp(-1 * gamma * l2_norm(x[0], x[1])))
        elif score_func == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / (1 + K.exp(-1 * gamma * (dot(x[0], x[1]) + c)))
            return Lambda(lambda x: euclidean(x) * sigmoid(x))
        elif score_func == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * gamma * (dot(x[0], x[1]) + c)))
            return Lambda(lambda x: euclidean(x) + sigmoid(x))
        else:
            raise ValueError('Invalid similarity score: {}'.format(score_func))
