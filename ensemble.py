# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: ensemble.py

@time: 2019/7/4 20:43

@desc:

"""

import os
import gc
import json
import numpy as np
from keras import optimizers
import keras.backend as K
from config import ModelConfig, PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, MENTION_TO_ENTITY_FILENAME, \
    ENTITY_DESC_FILENAME, EMBEDDING_MATRIX_TEMPLATE, SUBMIT_DIR
from itertools import groupby
from models.recognition_model import RecognitionModel
from models.linking_model import LinkModel
from utils.data_loader import RecognitionDataGenerator, load_data
from utils.io import pickle_load, format_filename, submit_result, pickle_dump
from utils.other import pad_sequences_1d

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


def recognition(model_name, label_schema='BIOES', batch_size=32, n_epoch=50, learning_rate=0.001, optimizer_type='adam',
                use_char_input=True, embed_type=None, embed_trainable=True,
                use_bert_input=False, bert_type='bert', bert_trainable=True, bert_layer_num=1,
                use_bichar_input=False, bichar_embed_type=None, bichar_embed_trainable=True,
                use_word_input=False, word_embed_type=None, word_embed_trainable=True,
                use_charpos_input=False, charpos_embed_type=None, charpos_embed_trainable=True,
                use_softword_input=False, use_dictfeat_input=False, use_maxmatch_input=False,
                callbacks_to_add=None, swa_type=None, predict_on_dev=False, predict_on_final_test=True, **kwargs):
    config = ModelConfig()
    config.model_name = model_name
    config.label_schema = label_schema
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    config.embed_type = embed_type
    config.use_char_input = use_char_input
    if embed_type:
        config.embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type=embed_type))
        config.embed_trainable = embed_trainable
        config.embed_dim = config.embeddings.shape[1]
    else:
        config.embeddings = None
        config.embed_trainable = True
    config.callbacks_to_add = callbacks_to_add or ['modelcheckpoint', 'earlystopping']

    config.vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='char'))
    config.vocab_size = len(config.vocab) + 2
    config.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))

    if config.use_char_input:
        config.exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(model_name,
                                                        config.embed_type if config.embed_type else 'random',
                                                        'tune' if config.embed_trainable else 'fix', batch_size,
                                                        optimizer_type, learning_rate, label_schema)
    else:
        config.exp_name = '{}_{}_{}_{}_{}'.format(model_name, batch_size, optimizer_type, learning_rate, label_schema)
    if kwargs:
        config.exp_name += '_' + '_'.join([str(k) + '_' + str(v) for k, v in kwargs.items()])
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    config.use_bert_input = use_bert_input
    config.bert_type = bert_type
    config.bert_trainable = bert_trainable
    config.bert_layer_num = bert_layer_num
    assert config.use_char_input or config.use_bert_input
    if config.use_bert_input:
        config.exp_name += '_{}_layer_{}_{}'.format(bert_type, bert_layer_num, 'tune' if config.bert_trainable else 'fix')
    config.use_bichar_input = use_bichar_input
    if config.use_bichar_input:
        config.bichar_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='bichar'))
        config.bichar_vocab_size = len(config.bichar_vocab) + 2
        if bichar_embed_type:
            config.bichar_embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE,
                                                               type=bichar_embed_type))
            config.bichar_embed_trainable = bichar_embed_trainable
            config.bichar_embed_dim = config.bichar_embeddings.shape[1]
        else:
            config.bichar_embeddings = None
            config.bichar_embed_trainable = True
        config.exp_name += '_bichar_{}_{}'.format(bichar_embed_type if bichar_embed_type else 'random',
                                                  'tune' if config.bichar_embed_trainable else 'fix')
    config.use_word_input = use_word_input
    if config.use_word_input:
        config.word_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'))
        config.word_vocab_size = len(config.word_vocab) + 2
        if word_embed_type:
            config.word_embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE,
                                                             type=word_embed_type))
            config.word_embed_trainable = word_embed_trainable
            config.word_embed_dim = config.word_embeddings.shape[1]
        else:
            config.word_embeddings = None
            config.word_embed_trainable = True
        config.exp_name += '_word_{}_{}'.format(word_embed_type if word_embed_type else 'random',
                                                'tune' if config.word_embed_trainable else 'fix')
    config.use_charpos_input = use_charpos_input
    if config.use_charpos_input:
        config.charpos_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='charpos'))
        config.charpos_vocab_size = len(config.charpos_vocab) + 2
        if charpos_embed_type:
            config.charpos_embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE,
                                                                type=charpos_embed_type))
            config.charpos_embed_trainable = charpos_embed_trainable
            config.charpos_embed_dim = config.charpos_embeddings.shape[1]
        else:
            config.charpos_embeddings = None
            config.charpos_embed_trainable = True
        config.exp_name += '_charpos_{}_{}'.format(charpos_embed_type if charpos_embed_type else 'random',
                                                   'tune' if config.charpos_embed_trainable else 'fix')
    config.use_softword_input = use_softword_input
    if config.use_softword_input:
        config.exp_name += '_softword'
    config.use_dictfeat_input = use_dictfeat_input
    if config.use_dictfeat_input:
        config.exp_name += '_dictfeat'
    config.use_maxmatch_input = use_maxmatch_input
    if config.use_maxmatch_input:
        config.exp_name += '_maxmatch'

    test_prob_file = os.path.join(SUBMIT_DIR, config.exp_name)
    if predict_on_final_test:
        test_prob_file += '_final'
    test_pred_prob = None
    dev_prob_file = os.path.join(SUBMIT_DIR, config.exp_name + 'dev')
    dev_pred_prob = None

    if os.path.exists(test_prob_file):
        test_pred_prob = pickle_load(test_prob_file)
    if predict_on_dev and os.path.exists(dev_prob_file):
        dev_pred_prob = pickle_load(dev_prob_file)

    if test_pred_prob is None or (predict_on_dev and dev_pred_prob is None):
        print('Logging Info - Experiment: %s' % config.exp_name)
        model = RecognitionModel(config, **kwargs)

        if swa_type is None:
            model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
            if not os.path.exists(model_save_path):
                raise FileNotFoundError('Recognition model not exist: {}'.format(model_save_path))
            model.load_best_model()
        elif 'swa' in callbacks_to_add:
            model_save_path = os.path.join(config.checkpoint_dir, '{}_{}.hdf5'.format(config.exp_name,
                                                                                      swa_type))
            if not os.path.exists(model_save_path):
                raise FileNotFoundError('Recognition model not exist: {}'.format(model_save_path))
            model.load_swa_model(swa_type)

        print('Logging Info - Generate recognition prediction for test data:')
        if predict_on_final_test:
            test_data_type = 'test_final'
        else:
            test_data_type = 'test'
        test_generator = RecognitionDataGenerator(test_data_type, config.batch_size, config.label_schema,
                                                  config.label_to_one_hot[config.label_schema],
                                                  config.vocab if config.use_char_input else None,
                                                  config.bert_vocab_file(
                                                      config.bert_type) if config.use_bert_input else None,
                                                  config.bert_seq_len, config.bichar_vocab, config.word_vocab,
                                                  config.use_word_input, config.charpos_vocab,
                                                  config.use_softword_input, config.use_dictfeat_input,
                                                  config.use_maxmatch_input)
        test_pred_prob = model.predict_prob(test_generator)
        pickle_dump(test_prob_file, test_pred_prob)

        if predict_on_dev:
            dev_data_type = 'dev'
            valid_generator = RecognitionDataGenerator(dev_data_type, config.batch_size, config.label_schema,
                                                       config.label_to_one_hot[config.label_schema],
                                                       config.vocab if config.use_char_input else None,
                                                       config.bert_vocab_file(
                                                           config.bert_type) if config.use_bert_input else None,
                                                       config.bert_seq_len, config.bichar_vocab, config.word_vocab,
                                                       config.use_word_input, config.charpos_vocab,
                                                       config.use_softword_input, config.use_dictfeat_input,
                                                       config.use_maxmatch_input)
            print('Logging Info - Evaluate over valid data:')
            model.evaluate(valid_generator)
            print('Logging Info - Generate recognition prediction for dev data:')
            dev_pred_prob = model.predict_prob(valid_generator)
            pickle_dump(dev_prob_file, dev_pred_prob)

        del model
        gc.collect()
        K.clear_session()
    if predict_on_dev:
        return test_pred_prob, dev_pred_prob
    else:
        return test_pred_prob


def link(model_name, batch_size=32, n_epoch=50, learning_rate=0.001, optimizer_type='adam', embed_type=None,
         embed_trainable=True, use_relative_pos=False, n_neg=1, omit_one_cand=True, callbacks_to_add=None,
         swa_type=None, **kwargs):
    config = ModelConfig()
    config.model_name = model_name
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    config.embed_type = embed_type
    if embed_type:
        config.embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type=embed_type))
        config.embed_trainable = embed_trainable
    else:
        config.embeddings = None
        config.embed_trainable = True

    config.callbacks_to_add = callbacks_to_add or ['modelcheckpoint', 'earlystopping']

    config.vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='char'))
    config.vocab_size = len(config.vocab) + 2
    config.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))
    config.entity_desc = pickle_load(format_filename(PROCESSED_DATA_DIR, ENTITY_DESC_FILENAME))

    config.exp_name = '{}_{}_{}_{}_{}_{}'.format(model_name, embed_type if embed_type else 'random',
                                                 'tune' if embed_trainable else 'fix', batch_size, optimizer_type,
                                                 learning_rate)
    config.use_relative_pos = use_relative_pos
    if config.use_relative_pos:
        config.exp_name += '_rel'
    config.n_neg = n_neg
    if config.n_neg > 1:
        config.exp_name += '_neg_{}'.format(config.n_neg)
    config.omit_one_cand = omit_one_cand
    if not config.omit_one_cand:
        config.exp_name += '_not_omit'
    if kwargs:
        config.exp_name += '_' + '_'.join([str(k) + '_' + str(v) for k, v in kwargs.items()])
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    print('Logging Info - Experiment: %s' % config.exp_name)
    model = LinkModel(config, **kwargs)

    if swa_type is None:
        model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
        if not os.path.exists(model_save_path):
            raise FileNotFoundError('Linking model not exist: {}'.format(model_save_path))
        model.load_best_model()
    elif 'swa' in callbacks_to_add:
        model_save_path = os.path.join(config.checkpoint_dir, '{}_{}.hdf5'.format(config.exp_name,
                                                                                  swa_type))
        if not os.path.exists(model_save_path):
            raise FileNotFoundError('Linking model not exist: {}'.format(model_save_path))
        model.load_swa_model(swa_type)

    return model


def link_ensemble(link_model_list, text_data, pred_mentions, mention_to_entity, entity_desc, vocab, max_desc_len,
                  max_erl_len):
    print('Logging Info - Generate linking result:')
    pred_mention_entities = []
    for text, pred_mention in zip(text_data, pred_mentions):
        link_result = []
        text_ids = [vocab.get(token, 1) for token in text]

        cand_mention_entity = {}
        ent_owner = []
        ent_desc, begin, end, relative_pos = [], [], [], []
        for mention in pred_mention:
            if mention not in cand_mention_entity:  # there might be duplicated mention in pred_mentions
                cand_mention_entity[mention] = mention_to_entity.get(mention[0], [])

            _begin = mention[1]
            _end = _begin + len(mention[0]) - 1

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
                relative_pos.append(_rel_pos)
                ent_owner.append(mention)

        if ent_desc:
            cv_scores_list = []
            for model in link_model_list:
                model_inputs = []
                repeat_text_ids = np.repeat([text_ids], len(ent_desc), 0)
                model_inputs.append(repeat_text_ids)

                begin = np.array(begin)
                end = np.array(end)
                model_inputs.extend([begin, end])
                if model.config.use_relative_pos:
                    relative_pos = pad_sequences_1d(relative_pos)
                    model_inputs.append(relative_pos)
                ent_desc = pad_sequences_1d(ent_desc, max_desc_len)
                model_inputs.append(ent_desc)
                scores = model.link_model.predict(model_inputs)
                cv_scores_list.append(scores)

            ensemble_scores = np.mean(np.stack(cv_scores_list, axis=-1), axis=-1)[:, 0]
            for k, v in groupby(zip(ent_owner, ensemble_scores), key=lambda x: x[0]):
                score_to_rank = np.array([j[1] for j in v])
                ent_id = cand_mention_entity[k][np.argmax(score_to_rank)]
                link_result.append((k[0], k[1], ent_id))

        pred_mention_entities.append(link_result)
    return pred_mention_entities


def link_evaluate(pred_mention_entities, gold_mention_entities):
    assert len(pred_mention_entities) == len(gold_mention_entities)
    match, n_true, n_pred = 1e-10, 1e-10, 1e-10
    for pred_mention_entity, gold_mention_entity in zip(pred_mention_entities, gold_mention_entities):
        pred = set(pred_mention_entity)
        true = set(gold_mention_entity)
        match += len(pred & true)
        n_pred += len(pred)
        n_true += len(true)

    r = match / n_true
    p = match / n_pred
    f1 = 2 * match / (n_pred + n_true)
    print('Logging Info - Recall: {}, Precision: {}, F1: {}'.format(r, p, f1))
    return r, p, f1


def rec_evaluate(data, pred_mentions):
    print('Logging Info - Evaluate over valid data:')
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


if __name__ == '__main__':
    predict_on_final_test = True
    if predict_on_final_test:
        test_data_type = 'test_final'
    else:
        test_data_type = 'test'

    '''entity recognition model'''
    dev_pred_prob_list = []
    test_pred_prob_list = []

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v', charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='bilstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='stlstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='ernie', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=False, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=False, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    test_pred, dev_pred = recognition('2step_er', label_schema='BIOES', batch_size=32,
                                      use_char_input=True, embed_type='c2v', embed_trainable=False,
                                      use_bert_input=True, bert_type='bert_wwm', bert_trainable=False, bert_layer_num=1,
                                      use_bichar_input=True, bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                      use_word_input=True, word_embed_type='w2v', word_embed_trainable=False,
                                      use_charpos_input=True, charpos_embed_type='cpos2v',
                                      charpos_embed_trainable=False,
                                      use_softword_input=True, use_dictfeat_input=True, use_maxmatch_input=True,
                                      encoder_type='mullstm_cnn', use_crf=True,
                                      callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                      swa_type='swa', predict_on_dev=True, predict_on_final_test=predict_on_final_test)
    test_pred_prob_list.append(test_pred)
    dev_pred_prob_list.append(dev_pred)

    '''entity linking model'''
    el_model_list = []

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=5, omit_one_cand=False, use_relative_pos=True, max_mention=True,
                      add_cnn='after')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=4, omit_one_cand=False, use_relative_pos=True, max_mention=True,
                      add_cnn='after')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=4, use_relative_pos=True, max_mention=True)
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=5, use_relative_pos=True, max_mention=True, add_cnn='after')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=4, use_relative_pos=True, max_mention=True, add_cnn='after')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=4, max_mention=True,
                      encoder_type='self_attend_single_attend', ent_attend_type='add')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=4, max_mention=True,
                      use_relative_pos=True, encoder_type='self_attend_single_attend',
                      ent_attend_type='mul')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=4, max_mention=True,
                      use_relative_pos=True, add_cnn='after', encoder_type='self_attend_single_attend',
                      ent_attend_type='scaled_dot')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=5, use_relative_pos=True, max_mention=True)
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=5, max_mention=True)
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=4, max_mention=True, add_cnn='after')
    el_model_list.append(link_model)

    link_model = link('2step_el', batch_size=32, embed_type='c2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      swa_type='swa', n_neg=5, max_mention=True, add_cnn='after')
    el_model_list.append(link_model)

    '''model ensemble'''
    er_group_1 = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35]
    er_group_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30,
                  31, 32, 34, 35]
    el_group = list(range(12))

    for er_group in [er_group_2, er_group_1]:
        print('Loggin Info - ER group:', er_group)
        test_group_pred_prob_list = [test_pred_prob_list[i] for i in er_group]
        dev_group_pred_prob_list = [dev_pred_prob_list[i] for i in er_group]

        config = ModelConfig()
        config.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))

        test_pred_prob_ensemble = np.mean(np.stack(test_group_pred_prob_list, axis=-1), axis=-1)
        test_data = load_data(test_data_type)
        test_text_data = [data['text'] for data in test_data]
        test_pred_mentions = RecognitionModel.label_decode('BIOES', config.idx2label['BIOES'],
                                                           [x['text'] for x in test_data], test_pred_prob_ensemble,
                                                           config.mention_to_entity, remove_cls_seq=False)

        dev_pred_prob_ensemble = np.mean(np.stack(dev_group_pred_prob_list, axis=-1), axis=-1)
        dev_data = load_data('dev')
        dev_text_data, dev_gold_mention_entities = [], []
        for data in dev_data:
            dev_text_data.append(data['text'])
            dev_gold_mention_entities.append(data['mention_data'])
        dev_pred_mentions = RecognitionModel.label_decode('BIOES', config.idx2label['BIOES'],
                                                          [x['text'] for x in dev_data], dev_pred_prob_ensemble,
                                                          config.mention_to_entity, remove_cls_seq=False)
        print('Logging Info - Evaluate over valid data:')
        rec_evaluate(dev_data, dev_pred_mentions)

        config = el_model_list[0].config

        print('Loggin Info - EL group:', el_group)
        ensemble_model_list = [el_model_list[i] for i in el_group]

        # print('Logging Info - Evaluate over valid data based on gold mention:')
        # dev_pred_mention_entities = link_ensemble(ensemble_model_list, dev_text_data, dev_gold_mention_entities,
        #                                           config.mention_to_entity, config.entity_desc, config.vocab,
        #                                           config.max_desc_len, config.max_erl_len)
        # _, _, gold_f1 = link_evaluate(dev_pred_mention_entities, dev_gold_mention_entities)

        # print('Logging Info - Evaluate over valid data based on predicted mention:')
        # dev_pred_mention_entities = link_ensemble(ensemble_model_list, dev_text_data, dev_pred_mentions,
        #                                           config.mention_to_entity, config.entity_desc, config.vocab,
        #                                           config.max_desc_len, config.max_erl_len)
        # _, _, pred_f1 = link_evaluate(dev_pred_mention_entities, dev_gold_mention_entities)
        # el_results[el_group] = (gold_f1, pred_f1)

        print('Logging Info - Generate submission for test data:')
        test_pred_mention_entities = link_ensemble(ensemble_model_list, test_text_data, test_pred_mentions,
                                                   config.mention_to_entity, config.entity_desc, config.vocab,
                                                   config.max_desc_len, config.max_erl_len)
        test_submit_file = 'er_{}_ensemble_el_{}_ensemble_{}submit.json'.format(er_group, el_group,
                                                                                'final_' if predict_on_final_test else '')
        submit_result(test_submit_file, test_data, test_pred_mention_entities)

    '''results ensemble'''
    submit_file_1 = 'er_{}_ensemble_el_{}_ensemble_{}submit.json'.format(er_group_1, el_group,
                                                                         'final_' if predict_on_final_test else '')

    submit_file_2 = 'er_{}_ensemble_el_{}_ensemble_{}submit.json'.format(er_group_2, el_group,
                                                                         'final_' if predict_on_final_test else '')
    combine_file = 'final_submit.json'

    pred_group_1 = []
    with open(os.path.join(SUBMIT_DIR, submit_file_1), 'r') as reader:
        for line in reader:
            pred_group_1.append(json.loads(line))

    pred_group_2 = []
    with open(os.path.join(SUBMIT_DIR, submit_file_2), 'r') as reader:
        for line in reader:
            pred_group_2.append(json.loads(line))

    with open(os.path.join(SUBMIT_DIR, combine_file), 'w') as writer:
        for pred_1, pred_2 in zip(pred_group_1, pred_group_2):
            if len(pred_1['mention_data']) < len(pred_2['mention_data']):
                json.dump(pred_1, writer, ensure_ascii=False)
            elif len(pred_1['mention_data']) > len(pred_2['mention_data']):
                json.dump(pred_2, writer, ensure_ascii=False)
            else:
                json.dump(pred_2, writer, ensure_ascii=False)
            writer.write('\n')


