# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: recognition_model.py

@time: 2019/5/9 20:06

@desc:

"""

import re
from keras.layers import *
from keras.models import Model
from keras.callbacks import Callback
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_bert import load_trained_model_from_checkpoint
from models.base_model import BaseModel


class RecognitionMetric(Callback):
    def __init__(self, valid_generator, label_schema, idx2label, mention_to_entity, remove_cls_seq):
        self.valid_generator = valid_generator
        self.raw_data = self.valid_generator.data
        self.text_ids = self.valid_generator.convert_to_ids()
        self.label_schema = label_schema
        self.idx2label = idx2label
        self.mention_to_entity = mention_to_entity
        self.remove_cls_seq = remove_cls_seq
        super(RecognitionMetric, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        pred_results = self.model.predict(self.text_ids)
        pred_mentions = RecognitionModel.label_decode(self.label_schema, self.idx2label,
                                                      [x['text'] for x in self.raw_data], pred_results,
                                                      self.mention_to_entity, self.remove_cls_seq)
        match, n_true, n_pred = 1e-10, 1e-10, 1e-10
        for i in range(len(self.raw_data)):
            pred = set(pred_mentions[i])
            true = set([(mention[0], mention[1]) for mention in self.raw_data[i]['mention_data']])
            match += len(pred & true)
            n_pred += len(pred)
            n_true += len(true)
        r = match / n_true
        p = match / n_pred
        f1 = 2 * match / (n_pred + n_true)
        logs['val_r'] = r
        logs['val_p'] = p
        logs['val_f1'] = f1
        print('Epoch {}: val_r: {}, val_p: {}, val_f1: {}'.format(epoch, r, p, f1))


class RecognitionModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(RecognitionModel, self).__init__(config)
        self.model = self.build(**kwargs)
        if 'swa' in self.config.callbacks_to_add or 'swa_clr' in self.config.callbacks_to_add:
            self.swa_model = self.build(**kwargs)

    def add_metrics(self, valid_generator):
        self.callbacks.append(RecognitionMetric(valid_generator, self.config.label_schema,
                                                self.config.idx2label[self.config.label_schema],
                                                self.config.mention_to_entity,
                                                remove_cls_seq=self.config.use_bert_input))

    def build(self, encoder_type='bilstm_cnn', use_crf=True):
        model_inputs = []
        input_embed = []

        '''1. emebedding layer'''
        if self.config.use_char_input:
            if self.config.embeddings is not None:
                embedding_layer = Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embed_dim,
                                            weights=[self.config.embeddings], trainable=self.config.embed_trainable)
            else:
                embedding_layer = Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embed_dim)

            input_text = Input(shape=(self.config.bert_seq_len if self.config.use_bert_input else None,))
            model_inputs.append(input_text)
            input_embed.append(SpatialDropout1D(0.2)(embedding_layer(input_text)))

        if self.config.use_bert_input:
            bert_model = load_trained_model_from_checkpoint(self.config.bert_config_file(self.config.bert_type),
                                                            self.config.bert_model_file(self.config.bert_type),
                                                            trainable=self.config.bert_trainable,
                                                            output_layer_num=self.config.bert_layer_num,
                                                            seq_len=self.config.bert_seq_len)
            model_inputs.extend(bert_model.inputs)
            input_bert_embed = SpatialDropout1D(0.2)(bert_model.output)
            input_embed.append(input_bert_embed)

        if self.config.use_bichar_input:
            input_bichar = Input(shape=(self.config.bert_seq_len if self.config.use_bert_input else None, ))
            model_inputs.append(input_bichar)
            if self.config.bichar_embeddings is not None:
                bichar_embedding_layer = Embedding(input_dim=self.config.bichar_vocab_size,
                                                   output_dim=self.config.bichar_embed_dim,
                                                   weights=[self.config.bichar_embeddings],
                                                   trainable=self.config.bichar_embed_trainable)
            else:
                bichar_embedding_layer = Embedding(input_dim=self.config.bichar_vocab_size,
                                                   output_dim=self.config.bichar_embed_dim)
            input_bichar_embed = SpatialDropout1D(0.2)(bichar_embedding_layer(input_bichar))
            input_embed.append(input_bichar_embed)

        if self.config.use_word_input:
            input_word = Input(shape=(self.config.bert_seq_len if self.config.use_bert_input else None, ))
            model_inputs.append(input_word)
            if self.config.word_embeddings is not None:
                word_embedding_layer = Embedding(input_dim=self.config.word_vocab_size,
                                                 output_dim=self.config.word_embed_dim,
                                                 weights=[self.config.word_embeddings],
                                                 trainable=self.config.word_embed_trainable)
            else:
                word_embedding_layer = Embedding(input_dim=self.config.word_vocab_size,
                                                 output_dim=self.config.word_embed_dim)
            input_word_embed = SpatialDropout1D(0.2)(word_embedding_layer(input_word))
            input_embed.append(input_word_embed)

        if self.config.use_charpos_input:
            input_charpos = Input(shape=(self.config.bert_seq_len if self.config.use_bert_input else None, ))
            model_inputs.append(input_charpos)
            if self.config.charpos_embeddings is not None:
                charpos_embedding_layer = Embedding(input_dim=self.config.charpos_vocab_size,
                                                    output_dim=self.config.charpos_embed_dim,
                                                    weights=[self.config.charpos_embeddings],
                                                    trainable=self.config.charpos_embed_trainable)
            else:
                charpos_embedding_layer = Embedding(input_dim=self.config.charpos_vocab_size,
                                                    output_dim=self.config.charpos_embed_dim)
            input_charpos_embed = SpatialDropout1D(0.2)(charpos_embedding_layer(input_charpos))
            input_embed.append(input_charpos_embed)

        if self.config.use_softword_input:
            input_softword = Input(shape=(self.config.bert_seq_len if self.config.use_bert_input else None, ))
            model_inputs.append(input_softword)
            softword_embedding_layer = Embedding(input_dim=4, output_dim=self.config.softword_embed_dim)
            input_softword_embed = SpatialDropout1D(0.2)(softword_embedding_layer(input_softword))
            input_embed.append(input_softword_embed)

        if self.config.use_dictfeat_input:
            input_dictfeat = Input(shape=(self.config.bert_seq_len if self.config.use_bert_input else None, 14))
            model_inputs.append(input_dictfeat)
            input_embed.append(input_dictfeat)

        if self.config.use_maxmatch_input:
            input_maxmatch = Input(shape=(self.config.bert_seq_len if self.config.use_bert_input else None, ))
            model_inputs.append(input_maxmatch)
            maxmatch_embedding_layer = Embedding(input_dim=4, output_dim=self.config.maxmatch_embed_dim)
            input_maxmatch_embed = SpatialDropout1D(0.2)(maxmatch_embedding_layer(input_maxmatch))
            input_embed.append(input_maxmatch_embed)

        '''
        CUDALSTM (or CNN) doesn't support masking, so we don't use mask_zero in embedding layer, instead we apply 
        masking on our own
        '''
        input_mask = Lambda(lambda x: K.cast(K.greater(x, 0), K.floatx()))(model_inputs[0])
        mask_layer = Lambda(lambda x: x[0] * K.expand_dims(x[1], 2))

        input_embed = concatenate(input_embed) if len(input_embed) > 1 else input_embed[0]
        input_embed = mask_layer([SpatialDropout1D(0.2)(input_embed), input_mask])

        '''2. encoder layer'''
        def multi_cnn(_input):
            filter_lengths = [2, 3, 4, 5]
            outputs = []
            for filter_length in filter_lengths:
                conv_layer = Conv1D(filters=self.config.embed_dim // 2, kernel_size=filter_length, padding='same',
                                    strides=1, activation='relu')(_input)
                outputs.append(conv_layer)
            return concatenate(outputs)

        # iterated dilated convolution
        def idcnn(_input, repeat_times=4):
            layer_input = Conv1D(self.config.embed_dim, 3, padding='same')(_input)
            dilated_block = [Conv1D(self.config.embed_dim, 3, padding='same', dilation_rate=1, activation='relu'),
                             Conv1D(self.config.embed_dim, 3, padding='same', dilation_rate=1, activation='relu'),
                             Conv1D(self.config.embed_dim, 3, padding='same', dilation_rate=2, activation='relu')]
            outputs = []
            for i in range(repeat_times):
                for j in range(len(dilated_block)):
                    conv = dilated_block[j](layer_input)
                    if j == len(dilated_block) - 1:
                        outputs.append(conv)
                    layer_input = conv
            final_output = concatenate(outputs)
            return final_output

        if encoder_type == 'None':
            input_encode = input_embed
        elif encoder_type in ['lstm', 'lstm_cnn', 'lstm_multicnn']:
            input_encode = CuDNNLSTM(self.config.embed_dim, return_sequences=True, )(input_embed)
        elif encoder_type in ['bilstm', 'bilstm_cnn', 'bilstm_multicnn', 'bilstm_idcnn']:
            input_encode = Bidirectional(CuDNNLSTM(self.config.embed_dim // 2, return_sequences=True))(input_embed)
        elif encoder_type in ['relstm', 'relstm_cnn', 'relstm_multicnn', 'relstm_idcnn']:
            input_encode_1 = Bidirectional(CuDNNLSTM(self.config.embed_dim // 2, return_sequences=True))(input_embed)
            input_encode_2 = Bidirectional(CuDNNLSTM(self.config.embed_dim // 2, return_sequences=True))(input_encode_1)
            input_encode = add([input_encode_1, input_encode_2])    # residual connection to the first bilstm
        elif encoder_type in ['mullstm', 'mullstm_cnn', 'mullstm_multicnn', 'relstm_idcnn']:
            input_encode = Bidirectional(CuDNNLSTM(self.config.embed_dim // 2, return_sequences=True))(input_embed)
            input_encode = Bidirectional(CuDNNLSTM(self.config.embed_dim // 2, return_sequences=True))(input_encode)
        elif encoder_type in ['stlstm', 'stlstm_cnn', 'stlstm_multicnn', 'stlstm_idcnn']:
            input_reverse = Lambda(lambda x: K.reverse(x, 1))(input_embed)
            backward_lstm = CuDNNLSTM(self.config.embed_dim, return_sequences=True)(input_reverse)
            backward_lstm = Lambda(lambda x: K.reverse(x, 1))(backward_lstm)
            input_encode = CuDNNLSTM(self.config.embed_dim, return_sequences=True)(backward_lstm)
        elif encoder_type in ['gru', 'gru_cnn', 'gru_multicnn', 'gru_idcnn']:
            input_encode = CuDNNGRU(self.config.embed_dim, return_sequences=True)(input_embed)
        elif encoder_type in ['bigru', 'bigru_cnn', 'bigru_multicnn', 'bigru_idcnn']:
            input_encode = Bidirectional(CuDNNGRU(self.config.embed_dim // 2, return_sequences=True))(input_embed)
        elif encoder_type in ['cnn', 'cnn_bilstm', 'cnn_bigru', 'cnn_gru', 'cnn_lstm', 'cnn_relstm', 'cnn_stlstm']:
            input_encode = Conv1D(filters=self.config.embed_dim, kernel_size=3, padding='same',
                                  activation='relu')(input_embed)
        elif encoder_type in ['idcnn', 'idcnn_bilstm', 'idcnn_bigru', 'idcnn_gru', 'idcnn_lstm', 'idcnn_relstm',
                              'idcnn_stlstm']:
            input_encode = idcnn(input_embed)
        elif encoder_type in ['multicnn', 'multicnn_bilstm', 'multicnn_bigru', 'multicnn_gru', 'multicnn_lstm',
                              'multicnn_relstm', 'multicnn_stlstm']:
            input_encode = multi_cnn(input_embed)
        else:
            raise ValueError('Encoder Type Not Understood : {}'.format(encoder_type))

        if '_cnn' in encoder_type:
            input_encode = Conv1D(filters=self.config.embed_dim, kernel_size=3, padding='same',
                                  activation='relu')(input_encode)
        elif '_idcnn' in encoder_type:
            input_encode = idcnn(input_embed)
        elif '_multicnn' in encoder_type:
            input_encode = multi_cnn(input_encode)
        elif '_lstm' in encoder_type:
            input_encode = CuDNNLSTM(self.config.embed_dim, return_sequences=True)(input_encode)
        elif '_bilstm' in encoder_type:
            input_encode = Bidirectional(CuDNNLSTM(self.config.embed_dim // 2, return_sequences=True))(input_encode)
        elif '_gru' in encoder_type:
            input_encode = CuDNNGRU(self.config.embed_dim, return_sequences=True)(input_encode)
        elif '_bigru' in encoder_type:
            input_encode = Bidirectional(CuDNNGRU(self.config.embed_dim // 2, return_sequences=True))(input_encode)

        input_encode = mask_layer([input_encode, input_mask])

        '''3. predict layer'''
        if use_crf:
            crf = CRF(units=self.config.n_class[self.config.label_schema], name='ner_tag')
            ner_tag = crf(input_encode)
            ner_loss = crf_loss
            ner_metrics = crf.accuracy
        else:
            ner_tag = TimeDistributed(Dense(self.config.n_class[self.config.label_schema],
                                            activation='softmax'), name='ner_tag')(input_encode)
            ner_loss = 'categorical_crossentropy'
            ner_metrics = 'accuracy'
        ner_model = Model(model_inputs if len(model_inputs) > 1 else model_inputs[0], ner_tag)
        ner_model.compile(self.config.optimizer, loss=ner_loss, metrics=[ner_metrics])

        return ner_model

    def train(self, train_generator, valid_generator):
        self.callbacks = []
        self.add_metrics(valid_generator)
        self.init_callbacks(train_generator.data_size)

        print('Logging Info - Start training...')
        self.model.fit_generator(generator=train_generator, epochs=self.config.n_epoch, callbacks=self.callbacks,
                                 validation_data=valid_generator)
        print('Logging Info - Training end...')

    def predict(self, generator):
        # return predict mention
        pred_results = self.model.predict(generator.convert_to_ids())
        pred_mentions = self.label_decode(self.config.label_schema, self.config.idx2label[self.config.label_schema],
                                          [x['text'] for x in generator.data], pred_results,
                                          self.config.mention_to_entity,
                                          remove_cls_seq=self.config.use_bert_input)
        return pred_mentions

    def predict_prob(self, generator):
        # return predict probability
        pred_results = self.model.predict(generator.convert_to_ids())
        if self.config.use_bert_input:
            pred_results = pred_results[:, 1:, :]
        return pred_results

    def evaluate(self, generator):
        pred_mentions = self.predict(generator)
        data = generator.data
        match, n_true, n_pred = 1e-10, 1e-10, 1e-10
        for i in range(len(data)):
            pred = set(pred_mentions[i])
            true = set([(mention[0], mention[1]) for mention in data[i]['mention_data']])
            match += len(pred & true)
            n_pred += len(pred)
            n_true += len(true)
        r = match / n_true
        p = match / n_pred
        f1 = 2 * match / (n_pred + n_true)
        print('Logging Info - Recall: {}, Precision: {}, F1: {}'.format(r, p, f1))
        return r, p, f1

    @staticmethod
    def label_decode(label_schema, idx2label, text_data, pred_results, mention_to_entity, remove_cls_seq=False):
        assert label_schema in ['BIO', 'BIOES']
        assert len(text_data) == pred_results.shape[0]
        pattern = r'BI*' if label_schema == 'BIO' else r'BI*E|S'
        seq_len = pred_results.shape[1]

        pred_mentions = []
        for i in range(len(text_data)):
            text = text_data[i]

            if remove_cls_seq:
                index_range = range(1, min(1+len(text), seq_len))   # ignore <CLS> and <SEQ> (when using bert as input)
            else:
                index_range = range(min(len(text), seq_len))

            text_label = ''.join([idx2label[np.argmax(pred_results[i][j])] for j in index_range])

            mention = []
            for m in re.finditer(pattern, text_label):
                start = m.start(0)
                end = m.end(0)
                cand = text[start:end]
                if cand in mention_to_entity and mention_to_entity[cand]:
                    mention.append((cand, start))

            pred_mentions.append(mention)
        return pred_mentions

