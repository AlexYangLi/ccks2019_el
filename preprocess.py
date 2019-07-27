# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: preprocess.py

@time: 2019/5/8 20:37

@desc:

"""

import os
import json
from collections import defaultdict
from tqdm import tqdm
import jieba
import numpy as np
from config import PROCESSED_DATA_DIR, LOG_DIR, SUBMIT_DIR, MODEL_SAVED_DIR, KB_FILENAME, MENTION_TO_ENTITY_FILENAME, \
    ENTITY_TO_MENTION_FILENAME, ENTITY_DESC_FILENAME, ENTITY_TYPE_FILENAME, CCKS_TRAIN_FILENAME, VOCABULARY_TEMPLATE, \
    IDX2TOKEN_TEMPLATE, TRAIN_DATA_FILENAME, DEV_DATA_FILENAME, TEST_DATA_FILENAME, CCKS_TEST_FILENAME, \
    EMBEDDING_MATRIX_TEMPLATE, CCKS_TEST_FINAL_FILENAME, TEST_FINAL_DATA_FILENAME, IMG_DIR
from utils.io import pickle_dump, format_filename
from utils.embedding import train_w2v, train_glove, train_fasttext


def load_kb_data(kb_file):
    """process knowledge base file"""
    mention_to_entity = defaultdict(list)
    entity_to_mention = defaultdict(list)
    entity_desc = defaultdict()
    entity_type = defaultdict(list)
    with open(kb_file) as reader:
        for line in tqdm(reader):
            kb_data = json.loads(line)
            entity_id = kb_data['subject_id']

            desc = '\n'.join('%s：%s' % (data['predicate'], data['object']) for data in kb_data['data'])
            desc.lower()
            if not desc:
                continue
            entity_desc[entity_id] = desc

            mentions = list(set(kb_data.get('alias', []) + [kb_data['subject']]))
            mentions = [mention.lower() for mention in mentions]
            entity_to_mention[entity_id] = mentions

            for mention in mentions:
                if entity_id not in mention_to_entity[mention]:
                    mention_to_entity[mention].append(entity_id)

            for _type in kb_data['type']:
                entity_type[entity_id].append(_type)

    return mention_to_entity, entity_to_mention, entity_desc, entity_type


def load_train_data(erl_file):
    train_data = []
    with open(erl_file) as reader:
        for line in tqdm(reader):
            data = json.loads(line)
            erl_text = data['text'].lower()

            mention_data = []
            for x in data['mention_data']:
                mention_text, offset, entity = x['mention'].lower(), int(x['offset']), x['kb_id']
                if entity == 'NIL':
                    continue
                if erl_text[offset: offset + len(mention_text)] != mention_text:
                    offset = erl_text.find(mention_text)
                if offset == -1:
                    continue
                mention_data.append((mention_text, offset, entity))
            train_data.append({'text': erl_text, 'mention_data': mention_data})
    return train_data


def load_test_data(erl_file):
    test_data = []
    with open(erl_file) as reader:
        for line in tqdm(reader):
            data = json.loads(line)
            test_data.append({'text_id': data['text_id'], 'text': data['text'].lower(), 'raw_text': data['text']})
    return test_data


def load_char_vocab_and_corpus(entity_desc, train_data, min_count=2):
    chars = dict()
    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        for c in desc:
            chars[c] = chars.get(c, 0) + 1
        corpus.append(list(desc))
    for data in tqdm(iter(train_data)):
        for c in data['text']:
            chars[c] = chars.get(c, 0) + 1
        corpus.append(list(data['text']))
    chars = {i: j for i, j in chars.items() if j >= min_count}
    idx2char = {i + 2: j for i, j in enumerate(chars)}  # 0: mask, 1: padding
    char2idx = {j: i for i, j in idx2char.items()}
    return char2idx, idx2char, corpus


def load_bichar_vocab_and_corpus(entity_desc, train_data, min_count=2):
    bichars = dict()
    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        bigrams = []
        for i in range(len(desc)):
            c = desc[i] + '</end>' if i == len(desc) - 1 else desc[i:i+2]
            bigrams.append(c)
            bichars[c] = bichars.get(c, 0) + 1
        corpus.append(bigrams)
    for data in tqdm(iter(train_data)):
        bigrams = []
        for i in range(len(data['text'])):
            c = data['text'][i] + '</end>' if i == len(data['text']) - 1 else data['text'][i:i+2]
            bigrams.append(c)
            bichars[c] = bichars.get(c, 0) + 1
        corpus.append(bigrams)
    bichars = {i: j for i, j in bichars.items() if j >= min_count}
    idx2bichar = {i + 2: j for i, j in enumerate(bichars)}  # 0: mask, 1: padding
    bichar2idx = {j: i for i, j in idx2bichar.items()}
    return bichar2idx, idx2bichar, corpus


def load_word_vocab_and_corpus(entity_desc, train_data, min_count=2):
    words = dict()
    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        desc_cut = jieba.lcut(desc)
        for w in desc_cut:
            words[w] = words.get(w, 0) + 1
        corpus.append(desc_cut)
    for data in tqdm(iter(train_data)):
        text_cut = jieba.lcut(data['text'])
        for w in text_cut:
            words[w] = words.get(w, 0) + 1
        corpus.append(text_cut)
    words = {i: j for i, j in words.items() if j >= min_count}
    idx2word = {i + 2: j for i, j in enumerate(words)}  # 0: mask, 1: padding
    word2idx = {j: i for i, j in idx2word.items()}
    return word2idx, idx2word, corpus


def load_charpos_vocab_and_corpus(char2idx, entity_desc, train_data):
    """build position aware character vocabulary by assign 4 positional tags: <B> <M> <E> <S>"""
    charpos2idx = {'<B>': 2, '<M>': 3, '<E>': 4, '<S>': 5}
    for c in char2idx.keys():
        charpos2idx[c+'<B>'] = len(charpos2idx) + 2
        charpos2idx[c+'<M>'] = len(charpos2idx) + 2
        charpos2idx[c+'<E>'] = len(charpos2idx) + 2
        charpos2idx[c+'<S>'] = len(charpos2idx) + 2
    idx2charpos = dict((idx, c) for c, idx in charpos2idx.items())

    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        desc_cut = jieba.lcut(desc)
        desc_pos = []
        for word in desc_cut:
            if len(word) == 1:
                desc_pos.append(word+'<S>')     # single character as one word
            else:
                for i in range(len(word)):
                    if i == 0:
                        desc_pos.append(word[i]+'<B>')  # begin
                    elif i == len(word) - 1:
                        desc_pos.append(word[i]+'<E>')  # end
                    else:
                        desc_pos.append(word[i]+'<M>')  # middle
        corpus.append(desc_pos)
    for data in tqdm(iter(train_data)):
        text_cut = jieba.lcut(data['text'])
        text_pos = []
        for word in text_cut:
            if len(word) == 1:
                text_pos.append(word + '<S>')  # single character as one word
            else:
                for i in range(len(word)):
                    if i == 0:
                        text_pos.append(word[i] + '<B>')  # begin
                    elif i == len(word) - 1:
                        text_pos.append(word[i] + '<E>')  # end
                    else:
                        text_pos.append(word[i] + '<M>')  # middle
        corpus.append(text_pos)
    return charpos2idx, idx2charpos, corpus


def train_valid_split(train_data):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)

    dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == 0]
    train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != 0]

    return train_data, dev_data


if __name__ == '__main__':
    # create directory
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    if not os.path.exists(SUBMIT_DIR):
        os.makedirs(SUBMIT_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # load knowledge base data
    mention_to_entity, entity_to_mention, entity_desc, entity_type = load_kb_data(KB_FILENAME)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME), mention_to_entity)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_DESC_FILENAME), entity_desc)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_TYPE_FILENAME), entity_type)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_TO_MENTION_FILENAME), entity_to_mention)

    # load training data
    train_data = load_train_data(CCKS_TRAIN_FILENAME)

    # prepare character embedding
    char_vocab, idx2char, char_corpus = load_char_vocab_and_corpus(entity_desc, train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='char'), char_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='char'), idx2char)
    c2v = train_w2v(char_corpus, char_vocab)
    c_fastext = train_fasttext(char_corpus, char_vocab)
    c_glove = train_glove(char_corpus, char_vocab)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='c2v'), c2v)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='c_fasttext'), c_fastext)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='c_glove'), c_glove)

    # prepare bigram embedding
    bichar_vocab, idx2bichar, bichar_corpus = load_bichar_vocab_and_corpus(entity_desc, train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='bichar'), bichar_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='bichar'), idx2bichar)
    bic2v = train_w2v(bichar_corpus, bichar_vocab, embedding_dim=50)
    bic_fastext = train_fasttext(bichar_corpus, bichar_vocab, embedding_dim=50)
    bic_glove = train_glove(bichar_corpus, bichar_vocab, embedding_dim=50)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='bic2v'), bic2v)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='bic_fasttext'), bic_fastext)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='bic_glove'), bic_glove)

    for mention in mention_to_entity.keys():
        jieba.add_word(mention, freq=1000000)
    # prepare word embedding
    word_vocab, idx2word, word_corpus = load_word_vocab_and_corpus(entity_desc, train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'), word_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='word'), idx2word)
    w2v = train_w2v(word_corpus, word_vocab)
    w_fastext = train_fasttext(word_corpus, word_vocab)
    w_glove = train_glove(word_corpus, word_vocab)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='w2v'), w2v)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='w_fasttext'), w_fastext)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='w_glove'), w_glove)

    # prepare position-based character embedding
    charpos_vocab, idx2charpos, charpos_corpus = load_charpos_vocab_and_corpus(char_vocab, entity_desc, train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='charpos'), charpos_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='charpos'), idx2charpos)
    cpos2v = train_w2v(charpos_corpus, charpos_vocab)
    cpos_fastext = train_fasttext(charpos_corpus, charpos_vocab)
    cpos_glove = train_glove(charpos_corpus, charpos_vocab)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='cpos2v'), cpos2v)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='cpos_fasttext'), cpos_fastext)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='cpos_glove'), cpos_glove)

    # hold out split
    train_data, dev_data = train_valid_split(train_data)
    test_data = load_test_data(CCKS_TEST_FILENAME)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME), train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_DATA_FILENAME), dev_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME), test_data)

    # load test data
    test_data = load_test_data(CCKS_TEST_FILENAME)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME), test_data)
    test_final_data = load_test_data(CCKS_TEST_FINAL_FILENAME)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_FINAL_DATA_FILENAME), test_final_data)
