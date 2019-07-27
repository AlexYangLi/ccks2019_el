# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: config.py

@time: 2019/5/8 20:37

@desc:

"""

from os import path
from keras.optimizers import Adam

RAW_DATA_DIR = './raw_data'
PROCESSED_DATA_DIR = './data'
LOG_DIR = './log'
MODEL_SAVED_DIR = './ckpt'
SUBMIT_DIR = './submit'
IMG_DIR = './img'

CCKS_DIR = path.join(RAW_DATA_DIR, 'ccks2019_el')
CCKS_TRAIN_FILENAME = path.join(CCKS_DIR, 'train.json')
CCKS_TEST_FILENAME = path.join(CCKS_DIR, 'develop.json')
CCKS_TEST_FINAL_FILENAME = path.join(CCKS_DIR, 'eval722.json')
KB_FILENAME = path.join(CCKS_DIR, 'kb_data')

TRAIN_DATA_FILENAME = 'erl_train.pkl'
DEV_DATA_FILENAME = 'erl_dev.pkl'
TEST_DATA_FILENAME = 'erl_test.pkl'
TEST_FINAL_DATA_FILENAME = 'erl_test_final.pkl'

ENTITY_DESC_FILENAME = 'entity_desc.pkl'
MENTION_TO_ENTITY_FILENAME = 'mention_to_entity.pkl'
ENTITY_TO_MENTION_FILENAME = 'entity_to_mention.pkl'
ENTITY_TYPE_FILENAME = 'entity_type.pkl'

VOCABULARY_TEMPLATE = '{level}_vocab.pkl'
IDX2TOKEN_TEMPLATE = 'idx2{level}.pkl'
EMBEDDING_MATRIX_TEMPLATE = '{type}_embeddings.npy'
PERFORMANCE_LOG = '{model_type}_performance.log'

EXTERNAL_EMBEDDINGS_DIR = path.join(RAW_DATA_DIR, 'embeddings')
EXTERNAL_EMBEDDINGS_FILENAME = {
    'bert': path.join(EXTERNAL_EMBEDDINGS_DIR, 'chinese_L-12_H-768_A-12'),
    'ernie': path.join(EXTERNAL_EMBEDDINGS_DIR, 'baidu_ernie'),
    'bert_wwm': path.join(EXTERNAL_EMBEDDINGS_DIR, 'chinese_wwm_L-12_H-768_A-12')
}


class ModelConfig(object):
    def __init__(self):
        # input base config
        self.embed_dim = 300
        self.embed_trainable = True
        self.embeddings = None
        self.vocab = None   # character embedding as base input
        self.vocab_size = None
        self.mention_to_entity = None
        self.entity_desc = None

        # input config for entity recognition model
        self.label_schema = 'BIO'   # 'BIO' or 'BIOES'
        self.label_to_one_hot = {'BIO': {'O': [1, 0, 0], 'B': [0, 1, 0], 'I': [0, 0, 1], 'P': [0, 0, 0]},
                                 'BIOES': {'O': [1, 0, 0, 0, 0], 'B': [0, 1, 0, 0, 0], 'I': [0, 0, 1, 0, 0],
                                           'E': [0, 0, 0, 1, 0], 'S': [0, 0, 0, 0, 1], 'P': [0, 0, 0, 0, 0]}}
        self.idx2label = {'BIO': {0: 'O', 1: 'B', 2: 'I'},
                          'BIOES': {0: 'O', 1: 'B', 2: 'I', 3: 'E', 4: 'S'}}
        self.n_class = {'BIO': 3, 'BIOES': 5}

        self.use_char_input = True

        self.use_bert_input = False
        self.bert_type = 'bert'
        self.bert_model_file = lambda x: path.join(EXTERNAL_EMBEDDINGS_FILENAME[x], 'bert_model.ckpt')
        self.bert_config_file = lambda x: path.join(EXTERNAL_EMBEDDINGS_FILENAME[x], 'bert_config.json')
        self.bert_vocab_file = lambda x: path.join(EXTERNAL_EMBEDDINGS_FILENAME[x], 'vocab.txt')
        self.bert_layer_num = 1
        self.bert_seq_len = 50
        self.bert_trainable = True

        self.use_bichar_input = False
        self.bichar_vocab = None
        self.bichar_embed_dim = 50
        self.bichar_embed_trainable = False
        self.bichar_embeddings = None
        self.bichar_vocab_size = None

        self.use_word_input = False
        self.word_vocab = None
        self.word_embed_dim = 300
        self.word_embed_trainable = False
        self.word_embeddings = None
        self.word_vocab_size = None

        self.use_charpos_input = False
        self.charpos_vocab = None
        self.charpos_embed_dim = 300
        self.charpos_embed_trainable = False
        self.charpos_embeddings = None
        self.charpos_vocab_size = None

        self.use_softword_input = False
        self.softword_embed_dim = 50

        self.use_dictfeat_input = False

        self.use_maxmatch_input = False
        self.maxmatch_embed_dim = 50

        # input config for entity linking model
        self.max_cand_mention = 10
        self.max_cand_entity = 10
        self.max_desc_len = 400
        self.use_relative_pos = False  # use relative position (to mention) as input
        self.max_erl_len = 30
        self.n_rel_pos_embed = 60
        self.rel_pos_embed_dim = 50
        self.omit_one_cand = True
        self.n_neg = 1

        # model structure configuration
        self.exp_name = None
        self.model_name = None
        self.rnn_units = 300
        self.dense_units = 128

        # model training configuration
        self.batch_size = 64
        self.n_epoch = 50
        self.learning_rate = 0.001
        self.optimizer = Adam(self.learning_rate)
        self.threshold = 0.5

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_f1'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_f1'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1

        self.callbacks_to_add = None

        # config for learning rating scheduler and ensembler
        self.swa_start = 3
