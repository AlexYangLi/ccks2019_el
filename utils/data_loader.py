# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_loader.py

@time: 2019/5/8 22:07

@desc:

"""

import copy
import numpy as np
import codecs
import jieba
from random import sample
from keras.utils import Sequence, to_categorical
from keras_bert import Tokenizer
from utils.other import pad_sequences_1d, pad_sequences_2d
from utils.io import pickle_load, format_filename
from config import PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME, DEV_DATA_FILENAME, TEST_DATA_FILENAME, \
    MENTION_TO_ENTITY_FILENAME, TEST_FINAL_DATA_FILENAME


def load_data(data_type):
    if data_type == 'train':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME))
    elif data_type == 'dev':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_FILENAME))
    elif data_type == 'test':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME))
    elif data_type == 'test_final':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TEST_FINAL_DATA_FILENAME))
    else:
        raise ValueError('data tye not understood: {}'.format(data_type))
    return data


class RecognitionDataGenerator(Sequence):
    """
    Data Generator for entity recognition model
    """
    def __init__(self, data_type, batch_size, label_schema, label_to_onehot, char_vocab=None, bert_vocab=None,
                 bert_seq_len=None, bichar_vocab=None, word_vocab=None, use_word_input=False, charpos_vocab=None,
                 use_softword_input=False, use_dictfeat_input=False, use_maxmatch_input=False, shuffle=True):
        self.data_type = data_type
        self.data = load_data(data_type)
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.steps = int(np.ceil(self.data_size / self.batch_size))

        assert label_schema in ['BIO', 'BIOES']
        self.label_schema = label_schema
        self.label_to_onehot = label_to_onehot

        # main input
        self.char_vocab = char_vocab
        self.use_char_input = False if self.char_vocab is None else True

        # additional feature input
        self.bert_vocab = bert_vocab
        self.use_bert_input = False if self.bert_vocab is None else True
        self.bert_seq_len = bert_seq_len if self.use_bert_input else None
        assert self.use_char_input or self.use_bert_input
        if self.use_bert_input:
            self.token_dict = {}
            with codecs.open(self.bert_vocab, 'r', 'utf8') as reader:
                for line in reader:
                    token = line.strip()
                    self.token_dict[token] = len(self.token_dict)
            self.bert_tokenizer = Tokenizer(self.token_dict)

        self.bichar_vocab = bichar_vocab
        self.use_bichar_input = False if self.bichar_vocab is None else True

        self.word_vocab = word_vocab
        self.use_word_input = use_word_input
        assert not (self.use_word_input and self.word_vocab is None)

        self.charpos_vocab = charpos_vocab
        self.use_charpos_input = False if self.charpos_vocab is None else True

        self.use_softword_input = use_softword_input
        self.use_dictfeat_input = use_dictfeat_input
        self.use_maxmatch_input = use_maxmatch_input

        self.mention_to_entity = None
        if self.use_word_input or self.use_charpos_input or self.use_softword_input:
            self.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))
            for mention in self.mention_to_entity.keys():
                jieba.add_word(mention, freq=1000000)
        if (self.use_dictfeat_input or self.use_maxmatch_input) and self.mention_to_entity is None:
            self.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))

        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_char_ids, batch_bert_ids, batch_bert_seg_ids, batch_labels = [], [], [], []
        batch_bichar_ids, batch_word_ids, batch_charpos_ids, batch_seg_labels, batch_dict_feats = [], [], [], [], []
        batch_maxmatches = []

        for i in batch_index:
            data = self.data[i]
            raw_text = data['text']

            if self.use_char_input:
                char_ids = [self.char_vocab.get(c, 1) for c in raw_text]
                if self.use_bert_input:
                    char_ids = [1] + char_ids + [1]     # use <UNK> to represent <CLS> and <SEQ>
                batch_char_ids.append(char_ids)
            if self.use_bert_input:
                raw_text_with_space = ' '.join(list(raw_text))  # to avoid non-chinese token being seg into subtokens
                indices, segments = self.bert_tokenizer.encode(first=raw_text_with_space, max_len=self.bert_seq_len)
                batch_bert_ids.append(indices)
                batch_bert_seg_ids.append(segments)
            if self.use_bichar_input:
                batch_bichar_ids.append(self.get_bichar_ids(raw_text))
            if self.use_word_input or self.use_charpos_input or self.use_softword_input:
                word_cut = jieba.lcut(raw_text)
            if self.use_word_input:
                batch_word_ids.append(self.get_word_ids(word_cut))
            if self.use_charpos_input:
                batch_charpos_ids.append(self.get_charpos_ids(word_cut))
            if self.use_softword_input:
                batch_seg_labels.append(self.get_softword_ids(word_cut))
            if self.use_dictfeat_input:
                batch_dict_feats.append(self.get_dictfeat_ids(raw_text))
            if self.use_maxmatch_input:
                batch_maxmatches.append(self.get_maxmatch_ids(raw_text))

            if 'mention_data' in data:
                label = [self.label_to_onehot['O']] * len(raw_text)
                for mention in data['mention_data']:
                    left = mention[1]
                    right = left + len(mention[0])
                    if self.label_schema == 'BIO':
                        for j in range(left, right):
                            if j == left:
                                label[j] = self.label_to_onehot['B']
                            else:
                                label[j] = self.label_to_onehot['I']
                    else:
                        if left + 1 == right:
                            label[left] = self.label_to_onehot['S']
                        else:
                            for j in range(left, right):
                                if j == left:
                                    label[j] = self.label_to_onehot['B']
                                elif j == right - 1:
                                    try:
                                        label[j] = self.label_to_onehot['E']
                                    except IndexError:
                                        print(data)
                                        exit(1)
                                else:
                                    label[j] = self.label_to_onehot['I']
                if self.use_bert_input:
                    label = [self.label_to_onehot['O']] + label + [self.label_to_onehot['O']]   # add <CLS> and <SEQ>
                batch_labels.append(label)

        batch_text_ids = []
        if self.use_char_input:
            # if not use bert input, bert_seq_len will be None
            batch_text_ids.append(pad_sequences_1d(batch_char_ids, max_len=self.bert_seq_len))
        if self.use_bert_input:
            batch_text_ids.append(pad_sequences_1d(batch_bert_ids, max_len=self.bert_seq_len))
            batch_text_ids.append(pad_sequences_1d(batch_bert_seg_ids, max_len=self.bert_seq_len))
        if self.use_bichar_input:
            batch_text_ids.append(pad_sequences_1d(batch_bichar_ids, max_len=self.bert_seq_len))
        if self.use_word_input:
            batch_text_ids.append(pad_sequences_1d(batch_word_ids, max_len=self.bert_seq_len))
        if self.use_charpos_input:
            batch_text_ids.append(pad_sequences_1d(batch_charpos_ids, max_len=self.bert_seq_len))
        if self.use_softword_input:
            batch_text_ids.append(pad_sequences_1d(batch_seg_labels, max_len=self.bert_seq_len))
        if self.use_dictfeat_input:
            batch_text_ids.append(pad_sequences_2d(batch_dict_feats, max_len_1=self.bert_seq_len, max_len_2=14))
        if self.use_maxmatch_input:
            batch_text_ids.append(pad_sequences_1d(batch_maxmatches, max_len=self.bert_seq_len))

        if len(batch_text_ids) == 1:
            batch_text_ids = batch_text_ids[0]

        if not batch_labels:
            return batch_text_ids, None     # test data don't have labels
        else:
            max_len = batch_text_ids[0].shape[1] if isinstance(batch_text_ids, list) else batch_text_ids.shape[1]
            batch_labels_pad = []
            for label in batch_labels:
                label_pad = copy.deepcopy(label)
                label_pad = label_pad[:max_len]
                num_to_pad = max_len - len(label)
                for i in range(num_to_pad):
                    label_pad.append(self.label_to_onehot['P'])
                batch_labels_pad.append(label_pad)
            batch_labels_pad = np.array(batch_labels_pad)

            return batch_text_ids, batch_labels_pad

    def label_pad(self, batch_labels, max_len, one_hot_encode, num_classes, pad_value=0):
        batch_labels_pad = []
        for label in batch_labels:
            if one_hot_encode:
                label_pad = to_categorical(label, num_classes).tolist()
            else:
                label_pad = copy.deepcopy(label)
            label_pad = label_pad[:max_len]
            num_to_pad = max_len - len(label)
            for i in range(num_to_pad):
                label_pad.append([pad_value] * num_classes)
            batch_labels_pad.append(label_pad)
        return np.array(batch_labels_pad)

    def convert_to_ids(self):
        # used for evaluation and prediction on validation and test data
        all_char_ids, all_bert_ids, all_bert_seg_ids = [], [], []
        all_bichar_ids, all_word_ids, all_charpos_ids, all_seg_labels = [], [], [], []
        all_dict_feats, all_maxmatches = [], []
        for i in range(self.data_size):
            data = self.data[i]
            raw_text = data['text']

            if self.use_char_input:
                char_ids = [self.char_vocab.get(c, 1) for c in raw_text]
                if self.use_bert_input:
                    char_ids = [1] + char_ids + [1]     # use <UNK> to represent <CLS> and <SEQ>
                all_char_ids.append(char_ids)
            if self.use_bert_input:
                raw_text_with_space = ' '.join(list(raw_text))  # to avoid non-chinese token being seg into subtokens
                indices, segments = self.bert_tokenizer.encode(first=raw_text_with_space, max_len=self.bert_seq_len)
                all_bert_ids.append(indices)
                all_bert_seg_ids.append(segments)
            if self.use_bichar_input:
                all_bichar_ids.append(self.get_bichar_ids(raw_text))
            if self.use_word_input or self.use_charpos_input or self.use_softword_input:
                word_cut = jieba.lcut(raw_text)
            if self.use_word_input:
                all_word_ids.append(self.get_word_ids(word_cut))
            if self.use_charpos_input:
                all_charpos_ids.append(self.get_charpos_ids(word_cut))
            if self.use_softword_input:
                all_seg_labels.append(self.get_softword_ids(word_cut))
            if self.use_dictfeat_input:
                all_dict_feats.append(self.get_dictfeat_ids(raw_text))
            if self.use_maxmatch_input:
                all_maxmatches.append(self.get_maxmatch_ids(raw_text))

        all_inputs = []
        if self.use_char_input:
            all_inputs.append(pad_sequences_1d(all_char_ids, max_len=self.bert_seq_len))
        if self.use_bert_input:
            all_inputs.append(pad_sequences_1d(all_bert_ids, max_len=self.bert_seq_len))
            all_inputs.append(pad_sequences_1d(all_bert_seg_ids, max_len=self.bert_seq_len))
        if self.use_bichar_input:
            all_inputs.append(pad_sequences_1d(all_bichar_ids, max_len=self.bert_seq_len))
        if self.use_word_input:
            all_inputs.append(pad_sequences_1d(all_word_ids, max_len=self.bert_seq_len))
        if self.use_charpos_input:
            all_inputs.append(pad_sequences_1d(all_charpos_ids, max_len=self.bert_seq_len))
        if self.use_softword_input:
            all_inputs.append(pad_sequences_1d(all_seg_labels, max_len=self.bert_seq_len))
        if self.use_dictfeat_input:
            all_inputs.append(pad_sequences_2d(all_dict_feats, max_len_1=self.bert_seq_len, max_len_2=14))
        if self.use_maxmatch_input:
            all_inputs.append(pad_sequences_1d(all_maxmatches, max_len=self.bert_seq_len))

        if len(all_inputs) == 1:
            all_inputs = all_inputs[0]
        return all_inputs

    def get_bichar_ids(self, raw_text):
        bichar_ids = []
        for j in range(len(raw_text)):
            c = raw_text[j] + '</end>' if j == len(raw_text) - 1 else raw_text[j:j + 2]
            bichar_ids.append(self.bichar_vocab.get(c, 1))
        if self.use_bert_input:
            bichar_ids = [1] + bichar_ids + [1]     # add <CLS> and <SEQ>
        return bichar_ids

    def get_word_ids(self, word_cut):
        word_ids = []
        for word in word_cut:
            for _ in word:
                word_ids.append(self.word_vocab.get(word, 1))  # all char in one word share the same word embedding
        if self.use_bert_input:
            word_ids = [1] + word_ids + [1]  # add <CLS> and <SEQ>
        return word_ids

    def get_charpos_ids(self, word_cut):
        charpos_ids = []
        for word in word_cut:
            if len(word) == 1:
                charpos_ids.append(self.charpos_vocab.get(word + '<S>', self.charpos_vocab['<S>']))
            else:
                for t in range(len(word)):
                    if t == 0:
                        charpos_ids.append(self.charpos_vocab.get(word[t] + '<B>', self.charpos_vocab['<B>']))
                    elif t == len(word) - 1:
                        charpos_ids.append(self.charpos_vocab.get(word[t] + '<E>', self.charpos_vocab['<E>']))
                    else:
                        charpos_ids.append(self.charpos_vocab.get(word[t] + '<M>', self.charpos_vocab['<M>']))
        if self.use_bert_input:
            charpos_ids = [self.charpos_vocab['<S>']] + charpos_ids + [self.charpos_vocab['<S>']]   # add <CLS> and <SEQ>
        return charpos_ids

    def get_softword_ids(self, word_cut):
        seg_label = []  # use BMES schema: B:0, M:1, E: 2, S: 3
        for word in word_cut:
            if len(word) == 1:
                seg_label.append(3)
            else:
                for k in range(len(word)):
                    if k == 0:
                        seg_label.append(0)
                    elif k == len(word) - 1:
                        seg_label.append(2)
                    else:
                        seg_label.append(1)
        if self.use_bert_input:
            seg_label = [3] + seg_label + [3]   # add <CLS> and <SEQ>
        return seg_label

    def get_dictfeat_ids(self, raw_text):
        dict_feat = []
        for l in range(len(raw_text)):
            match = [0] * 14
            for n in range(7):
                if l - n - 1 >= 0 and raw_text[l - n - 1:l + 1] in self.mention_to_entity:
                    match[2 * n] = 1
                if l + n + 2 <= len(raw_text) and raw_text[l:l + n + 2] in self.mention_to_entity:
                    match[2 * n + 1] = 1
            dict_feat.append(match)
        if self.use_bert_input:
            dict_feat = [[0] * 14] + dict_feat + [[0] * 14]
        return dict_feat

    def get_maxmatch_ids(self, raw_text):
        '''bidirectional maximum matching'''
        # forward maximum matching
        result_f = []
        index = 0
        text_len = len(raw_text)
        while text_len > index:
            for size in range(text_len, index, -1):
                piece = raw_text[index:size]
                if piece in self.mention_to_entity:
                    index = size - 1
                    break
            result_f.append(piece)
            index += 1

        # backward maximum matching
        result_b = []
        index = text_len
        while index > 0:
            for size in range(0, index):
                piece = raw_text[size:index]
                if piece in self.mention_to_entity:
                    index = size + 1
                    break
            result_b.append(piece)
            index -= 1
        result_b.reverse()
        result = result_f if len(result_f) < len(result_b) else result_b

        match_label = []    # use BMEO schema: B:0, M:1, E: 2, O: 3
        for word in result:
            if len(word) == 1:
                match_label.append(3)
            else:
                for i in range(len(word)):
                    if i == 0:
                        match_label.append(0)
                    elif i == len(word) -1:
                        match_label.append(2)
                    else:
                        match_label.append(1)
        if self.use_bert_input:
            match_label = [3] + match_label + [3]
        return match_label


class LinkDataGenerator(Sequence):
    """
    Data Generator for entity linking model
    """
    def __init__(self, data_type, char_vocab, mention_to_entity, entity_desc, batch_size, max_desc_len,
                 max_erl_len, use_relative_pos=False, n_neg=1, omit_one_cand=True, shuffle=True):
        self.data_type = data_type
        self.data = load_data(data_type)
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.steps = int(np.ceil(self.data_size / self.batch_size))

        self.char_vocab = char_vocab
        self.mention_to_entity = mention_to_entity
        self.entity_desc = entity_desc
        self.max_desc_len = max_desc_len

        self.use_relative_pos = use_relative_pos  # use relative position (to mention) as model's input
        self.max_erl_len = max_erl_len
        self.n_neg = n_neg      # how many negative sample
        self.omit_one_cand = omit_one_cand  # exclude those samples whose mention only has one candidate entity
        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_text_ids, batch_begin, batch_end, batch_rel_pos = [], [], [], []
        batch_pos_desc_ids, batch_neg_desc_ids = [], []
        batch_labels = [], []
        for i in batch_index:
            data = self.data[i]
            raw_text = data['text']
            text_ids = [self.char_vocab.get(c, 1) for c in raw_text]

            if 'mention_data' in data:
                for mention in data['mention_data']:
                    begin = mention[1]
                    end = begin + len(mention[0]) - 1
                    if self.use_relative_pos:
                        relative_pos = self.get_relative_pos(begin, end, len(text_ids))

                    pos_ent = mention[2]    # positive entity
                    cand_ents = copy.deepcopy(self.mention_to_entity[mention[0]])
                    while pos_ent in cand_ents:
                        cand_ents.remove(pos_ent)
                    # generate negative samples
                    if len(cand_ents) == 0:
                        if self.omit_one_cand:
                            continue
                        else:
                            neg_ents = sample(self.entity_desc.keys(), self.n_neg)
                    elif len(cand_ents) < self.n_neg:
                        neg_ents = cand_ents + sample(self.entity_desc.keys(), self.n_neg - len(cand_ents))
                    else:
                        neg_ents = sample(cand_ents, self.n_neg)

                    pos_desc_ids = [self.char_vocab.get(c, 1) for c in self.entity_desc[pos_ent]]
                    for neg_ent in neg_ents:
                        neg_desc_ids = [self.char_vocab.get(c, 1) for c in self.entity_desc[neg_ent]]
                        batch_neg_desc_ids.append(neg_desc_ids)
                        batch_pos_desc_ids.append(pos_desc_ids)

                    for _ in range(self.n_neg):
                        batch_text_ids.append(text_ids)
                        batch_begin.append([begin])
                        batch_end.append([end])
                        if self.use_relative_pos:
                            batch_rel_pos.append(relative_pos)

        batch_inputs = []
        batch_text_ids = pad_sequences_1d(batch_text_ids)
        batch_inputs.append(batch_text_ids)
        batch_begin = np.array(batch_begin)
        batch_end = np.array(batch_end)
        batch_inputs.extend([batch_begin, batch_end])

        if self.use_relative_pos:
            batch_rel_pos = pad_sequences_1d(batch_rel_pos)
            batch_inputs.append(batch_rel_pos)
        batch_pos_desc_ids = pad_sequences_1d(batch_pos_desc_ids, max_len=self.max_desc_len)
        batch_neg_desc_ids = pad_sequences_1d(batch_neg_desc_ids, max_len=self.max_desc_len)
        batch_inputs.extend([batch_pos_desc_ids, batch_neg_desc_ids])
        return batch_inputs, None

    def get_relative_pos(self, begin, end, text_len):
        relative_pos = []
        for i in range(text_len):
            if i < begin:
                relative_pos.append(max(i - begin + self.max_erl_len, 1))   # transform to id
            elif begin <= i <= end:
                relative_pos.append(0 + self.max_erl_len)
            else:
                relative_pos.append(min(i - end+self.max_erl_len, 2*self.max_erl_len-1))
        return relative_pos

