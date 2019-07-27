# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: other.py

@time: 2019/5/8 22:08

@desc:

"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences


def all_spans(begin_idx, end_idx, max_width, min_width):
    for left in range(begin_idx, end_idx):
        for length in range(min_width, max_width+1):
            if left + length > end_idx:
                break
            yield left, left+length


def pad_sequences_1d(sequences, max_len=None, padding='post', truncating='post', value=0.):
    """pad sequence for [[a, b, c, ...]]"""
    return pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating, value=value)


def pad_sequences_2d(sequences, max_len_1=None, max_len_2=None, dtype='int32', padding='post', truncating='post',
                     value=0.):
    """pad sequence for [[[a, b, c, ...]]]"""
    lengths_1, lengths_2 = [], []
    for s in sequences:
        lengths_1.append(len(s))
        for t in s:
            lengths_2.append(len(t))
    if max_len_1 is None:
        max_len_1 = np.max(lengths_1)
    if max_len_2 is None:
        max_len_2 = np.max(lengths_2)

    num_samples = len(sequences)
    x = (np.ones((num_samples, max_len_1, max_len_2)) * value).astype(dtype)
    for i, s in enumerate(sequences):
        if not len(s):
            continue    # empty list was found

        if truncating == 'pre':
            s = s[-max_len_1:]
        elif truncating == 'post':
            s = s[:max_len_1]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        y = (np.ones((len(s), max_len_2)) * value).astype(dtype)
        for j, t in enumerate(s):
            if not len(t):
                continue

            if truncating == 'pre':
                trunc = t[-max_len_2:]
            elif truncating == 'post':
                trunc = t[:max_len_2]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            trunc = np.asarray(trunc, dtype=dtype)

            if padding == 'post':
                y[j, :len(trunc)] = trunc
            elif padding == 'pre':
                y[j, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)

        if padding == 'post':
            x[i, :y.shape[0], :] = y
        elif padding == 'pre':
            x[i, -y.shape[0]:, :] = y
        else:
            raise ValueError('Padding type "%s" not understood' % padding)

    return x


def pad_sequences_3d(sequences, max_len_1=None, max_len_2=None, max_len_3=None, dtype='int32', padding='post',
                     truncating='post', value=0.):
    """pad sequences for [[[[]]]]"""
    lengths_1, lengths_2, lengths_3 = [], [], []
    for s in sequences:
        lengths_1.append(len(s))
        for t in s:
            lengths_2.append(len(t))
            for r in t:
                lengths_3.append(len(r))
    if max_len_1 is None:
        max_len_1 = np.max(lengths_1)
    if max_len_2 is None:
        max_len_2 = np.max(lengths_2)
    if max_len_3 is None:
        max_len_3 = np.max(lengths_3)

    num_samples = len(sequences)
    x = np.full(shape=(num_samples, max_len_1, max_len_2, max_len_3), fill_value=value, dtype=dtype)

    for i, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'post':
            s = s[:max_len_1]
        elif truncating == 'pre':
            s = s[-max_len_1:]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        y = np.full((len(s), max_len_2, max_len_3), value, dtype)
        for j, t in enumerate(s):
            if not len(t):
                continue
            if truncating == 'post':
                t = t[:max_len_2]
            elif truncating == 'pre':
                t = t[-max_len_2:]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            z = np.full((len(t), max_len_3), value, dtype)
            for k, r in enumerate(t):
                if not len(r):
                    continue
                if truncating == 'post':
                    r = r[:max_len_2]
                elif truncating == 'pre':
                    r = r[-max_len_2:]
                else:
                    raise ValueError('Truncating type "%s" not understood' % truncating)

                if padding == 'post':
                    z[k, :len(r)] = r
                elif padding == 'pre':
                    z[k, -len(r):] = r
                else:
                    raise ValueError('Padding type "%s" not understood' % padding)

            if padding == 'post':
                y[j, :z.shape[0], :] = z
            elif padding == 'pre':
                y[j, -z.shape[0]:, :] = z
            else:
                raise ValueError('Padding type "%s" not understood' % padding)

        if padding == 'post':
            x[i, :y.shape[0], :, :] = y
        elif padding == 'pre':
            x[i, -y.shape[0]:, :, :] = y
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
