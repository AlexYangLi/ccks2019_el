# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: 2step_predict.py

@time: 2019/5/18 14:08

@desc:

"""
import os
import time
import numpy as np
from keras import optimizers
from config import ModelConfig, PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, MENTION_TO_ENTITY_FILENAME, \
    ENTITY_DESC_FILENAME, EMBEDDING_MATRIX_TEMPLATE, LOG_DIR, PERFORMANCE_LOG
from models.recognition_model import RecognitionModel
from models.linking_model import LinkModel
from utils.data_loader import RecognitionDataGenerator, load_data
from utils.io import pickle_load, format_filename, submit_result, write_log

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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


def recognition(model_name, predict_log, label_schema='BIOES', batch_size=32, n_epoch=50, learning_rate=0.001,
                optimizer_type='adam', use_char_input=True, embed_type=None, embed_trainable=True,
                use_bert_input=False, bert_type='bert', bert_trainable=True, bert_layer_num=1,
                use_bichar_input=False, bichar_embed_type=None, bichar_embed_trainable=True,
                use_word_input=False, word_embed_type=None, word_embed_trainable=True,
                use_charpos_input=False, charpos_embed_type=None, charpos_embed_trainable=True,
                use_softword_input=False, use_dictfeat_input=False, use_maxmatch_input=False, callbacks_to_add=None,
                swa_type=None, predict_on_dev=True, predict_on_final_test=True, **kwargs):
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

    # logger to log output of training process
    predict_log.update({'er_exp_name': config.exp_name, 'er_batch_size': batch_size, 'er_optimizer': optimizer_type,
                        'er_epoch': n_epoch, 'er_learning_rate': learning_rate, 'er_other_params': kwargs})

    print('Logging Info - Experiment: %s' % config.exp_name)
    model = RecognitionModel(config, **kwargs)

    dev_data_type = 'dev'
    if predict_on_final_test:
        test_data_type = 'test_final'
    else:
        test_data_type = 'test'
    valid_generator = RecognitionDataGenerator(dev_data_type, config.batch_size, config.label_schema,
                                               config.label_to_one_hot[config.label_schema],
                                               config.vocab if config.use_char_input else None,
                                               config.bert_vocab_file(
                                                   config.bert_type) if config.use_bert_input else None,
                                               config.bert_seq_len, config.bichar_vocab, config.word_vocab,
                                               config.use_word_input, config.charpos_vocab, config.use_softword_input,
                                               config.use_dictfeat_input, config.use_maxmatch_input)
    test_generator = RecognitionDataGenerator(test_data_type, config.batch_size, config.label_schema,
                                              config.label_to_one_hot[config.label_schema],
                                              config.vocab if config.use_char_input else None,
                                              config.bert_vocab_file(
                                                  config.bert_type) if config.use_bert_input else None,
                                              config.bert_seq_len, config.bichar_vocab, config.word_vocab,
                                              config.use_word_input, config.charpos_vocab, config.use_softword_input,
                                              config.use_dictfeat_input, config.use_maxmatch_input)

    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not os.path.exists(model_save_path):
        raise FileNotFoundError('Recognition model not exist: {}'.format(model_save_path))

    if swa_type is None:
        model.load_best_model()
    elif 'swa' in callbacks_to_add:
        model.load_swa_model(swa_type)
        predict_log['er_exp_name'] += '_{}'.format(swa_type)

    if predict_on_dev:
        print('Logging Info - Generate submission for valid data:')
        dev_pred_mentions = model.predict(valid_generator)
    else:
        dev_pred_mentions = None
    print('Logging Info - Generate submission for test data:')
    test_pred_mentions = model.predict(test_generator)

    return dev_pred_mentions, test_pred_mentions


def link(model_name, dev_pred_mentions, test_pred_mentions, predict_log, batch_size=32, n_epoch=50, learning_rate=0.001,
         optimizer_type='adam', embed_type=None, embed_trainable=True, use_relative_pos=False, n_neg=1,
         omit_one_cand=True, callbacks_to_add=None, swa_type=None, predict_on_final_test=True, **kwargs):
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

    # logger to log output of training process
    predict_log.update({'el_exp_name': config.exp_name, 'el_batch_size': batch_size, 'el_optimizer': optimizer_type,
                        'el_epoch': n_epoch, 'el_learning_rate': learning_rate, 'el_other_params': kwargs})

    print('Logging Info - Experiment: %s' % config.exp_name)
    model = LinkModel(config, **kwargs)

    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not os.path.exists(model_save_path):
        raise FileNotFoundError('Recognition model not exist: {}'.format(model_save_path))
    if swa_type is None:
        model.load_best_model()
    elif 'swa' in callbacks_to_add:
        model.load_swa_model(swa_type)
        predict_log['er_exp_name'] += '_{}'.format(swa_type)

    dev_data_type = 'dev'
    dev_data = load_data(dev_data_type)
    dev_text_data, dev_gold_mention_entities = [], []
    for data in dev_data:
        dev_text_data.append(data['text'])
        dev_gold_mention_entities.append(data['mention_data'])

    if predict_on_final_test:
        test_data_type = 'test_final'
    else:
        test_data_type = 'test'
    test_data = load_data(test_data_type)
    test_text_data = [data['text'] for data in test_data]

    if dev_pred_mentions is not None:
        print('Logging Info - Evaluate over valid data based on predicted mention:')
        r, p, f1 = model.evaluate(dev_text_data, dev_pred_mentions, dev_gold_mention_entities)
        dev_performance = 'dev_performance' if swa_type is None else '%s_dev_performance' % swa_type
        predict_log[dev_performance] = (r, p, f1)
    print('Logging Info - Generate submission for test data:')
    test_pred_mention_entities = model.predict(test_text_data, test_pred_mentions)
    test_submit_file = predict_log['er_exp_name']+'_'+config.exp_name+'_%s%ssubmit.json' % (swa_type+'_' if swa_type else '',
                                                                                            'final_' if predict_on_final_test else '')
    submit_result(test_submit_file, test_data, test_pred_mention_entities)

    predict_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG, model_type='2step'), log=predict_log, mode='a')
    return predict_log


if __name__ == '__main__':
    predict_log = dict()
    dev_pred_mentions, test_pred_mentions = recognition('2step_er', predict_log, label_schema='BIOES', batch_size=32,
                                                        use_char_input=True, embed_type='c2v', embed_trainable=False,
                                                        use_bert_input=True, bert_type='ernie', bert_trainable=False,
                                                        bert_layer_num=1, use_bichar_input=False,
                                                        bichar_embed_type='bic2v', bichar_embed_trainable=False,
                                                        use_word_input=True, word_embed_type='w2v',
                                                        word_embed_trainable=False, use_charpos_input=False,
                                                        charpos_embed_type='cpos2v', charpos_embed_trainable=False,
                                                        use_softword_input=True, use_dictfeat_input=True,
                                                        use_maxmatch_input=True, encoder_type='bilstm_cnn',
                                                        use_crf=True,
                                                        callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                                                        swa_type='swa', predict_on_dev=True, predict_on_final_test=True)
    link('2step_el', dev_pred_mentions, test_pred_mentions, predict_log, batch_size=32, embed_type='c2v',
         embed_trainable=False, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'], swa_type='swa',
         n_neg=5, omit_one_cand=False, use_relative_pos=True, score_func='cosine', margin=0.04, max_mention=True,
         add_cnn='after', predict_on_final_test=True)
