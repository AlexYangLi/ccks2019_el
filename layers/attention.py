# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: attention.py

@time: 2019/5/11 8:32

@desc:

"""

from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class IntraSelfAttention(Layer):
    """
    Intra-Sentence self attention, support masking
    """
    def __init__(self, attend_type='dot', return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.attend_type = attend_type
        self.return_attend_weight = return_attend_weight
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)
        self.supports_masking = False
        super(IntraSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        input_a_shape, mask_shape = input_shape
        if len(input_a_shape) != 3 or len(mask_shape) != 2:
            raise ValueError('Input into IntraSelfAttention should be a 3D input tensor & a 2D mask tensor')

    def call(self, inputs, mask=None):
        input_a, input_mask = inputs

        if self.attend_type in ['dot', 'scaled_dot']:
            e = K.exp(K.batch_dot(input_a, input_a, axes=2))
            if self.attend_type == 'scaled_dot':
                e *= K.int_shape(input_a)[-1] ** -0.5
        else:
            raise ValueError('attend type not understand: {}'.format(self.attend_type))
        # elif self.attend_type == 'add':
        #     K.dot()

        # apply mask before normalization (softmax)
        e *= K.expand_dims(K.cast(input_mask, K.floatx()), 2)
        e *= K.expand_dims(K.cast(input_mask, K.floatx()), 1)

        # normalization
        e = e / K.cast(K.sum(e, axis=-1, keepdims=True) + K.epsilon(), K.floatx())

        if self.return_attend_weight:
            return e

        attend = K.batch_dot(e, input_a, axes=(2, 1))
        return attend

    def compute_output_shape(self, input_shape):
        input_a_shape, mask_shape = input_shape
        if self.return_attend_weight:
            return input_a_shape[0], input_a_shape[1], input_a_shape[1]
        return input_a_shape


class InteractiveAttention(Layer):
    """interactive attention between two sentence. Supporting Masking
    1. 'co_attend': Mention and Entity Description Co-Attention for entity Disambiguation. Nie et al.
                    https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16382
    2. 'max_mul_attend': Interactive Attention Networks for Aspect-Level Sentiment Classification. Ma et al.
                         https://www.ijcai.org/proceedings/2017/0568.pdf
    """
    def __init__(self, attend_type, initializer='orthogonal', regularizer=None, constraint=None, **kwargs):
        self.attend_type = attend_type
        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint
        self.supports_masking = False
        super(InteractiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        input_a_shape, input_b_shape = input_shape
        last_a, last_b = input_a_shape[-1], input_a_shape[-1]

        if self.attend_type == 'co_attend':
            self.w_alpha = self.add_weight(self.name+'_w_alpha', (last_a, last_b), initializer=self.initializer,
                                           regularizer=self.regularizer, constraint=self.constraint)
            self.w_a = self.add_weight(self.name+'_w_a', (last_a, last_a), initializer=self.initializer,
                                       regularizer=self.regularizer, constraint=self.constraint)
            self.w_b = self.add_weight(self.name+'_w_b', (last_b, last_a), initializer=self.initializer,
                                       regularizer=self.regularizer, constraint=self.constraint)
            self.w_ha = self.add_weight(self.name+'_w_ha', (last_a, 1), initializer=self.initializer,
                                        regularizer=self.regularizer, constraint=self.constraint)
            self.w_hb = self.add_weight(self.name+'_w_hb', (last_a, 1), initializer=self.initializer,
                                        regularizer=self.regularizer, constraint=self.constraint)
        elif self.attend_type == 'max_co_attend':
            self.w_a = self.add_weight(self.name+'_w_a', (last_a, last_b), initializer=self.initializer,
                                       regularizer=self.regularizer, constraint=self.constraint)
            self.w_b = self.add_weight(self.name+'_w_b', (last_b, last_a), initializer=self.initializer,
                                       regularizer=self.regularizer, constraint=self.constraint)
        else:
            raise ValueError('attend_type not understood: {}'.format(self.attend_type))
        super(InteractiveAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_a, input_b = inputs
        input_a_mask, input_b_mask = K.cast(K.any(input_a, 2), K.floatx()), K.cast(K.any(input_b, 2), K.floatx())

        if self.attend_type == 'co_attend':
            output_a, output_b = self.co_attend(input_a, input_b, input_a_mask, input_b_mask)
        elif self.attend_type == 'max_co_attend':
            output_a, output_b = self.max_co_attend(input_a, input_b, input_a_mask, input_b_mask)
        else:
            raise ValueError('attend_type not understood: {}'.format(self.attend_type))

        return [output_a, output_b]

    def co_attend(self, input_a, input_b, input_a_mask, input_b_mask):
        """compute affinity matrix"""
        attend = K.batch_dot(K.dot(input_a, self.w_alpha), input_b, axes=2)  # [batch_size, time_a, time_b]

        """apply softmax to normalize"""
        attend = K.exp(attend)
        attend *= K.expand_dims(input_a_mask, 2)
        attend *= K.expand_dims(input_b_mask, 1)
        attend_a = attend / (K.sum(attend, axis=2,
                                   keepdims=True) + K.epsilon())  # attend weights over input_b for each word in input_a
        attend_b = attend / (K.sum(attend, axis=1,
                                   keepdims=True) + K.epsilon())  # attend weights over input_a for each word in input_b

        """recompute attend probabilities"""
        h_a = K.tanh(K.dot(input_a, self.w_a) + K.batch_dot(attend_a, K.dot(input_b, self.w_b),
                                                            axes=(2, 1)))  # [batch_size, time_a, embed]
        h_b = K.tanh(K.dot(input_b, self.w_b) + K.batch_dot(attend_b, K.dot(input_a, self.w_a),
                                                            axes=1))  # [batch_size, time_b, embed]
        attend_a_new = K.squeeze(K.dot(h_a, self.w_ha), axis=2)
        attend_b_new = K.squeeze(K.dot(h_b, self.w_hb), axis=2)

        """apply softmax"""
        attend_a_new = K.exp(attend_a_new)
        attend_a_new *= input_a_mask
        attend_a_new = attend_a_new / (K.sum(attend_a_new, axis=1, keepdims=True) + K.epsilon())
        attend_b_new = K.exp(attend_b_new)
        attend_b_new *= input_b_mask
        attend_b_new = attend_b_new / (K.sum(attend_b_new, axis=1, keepdims=True) + K.epsilon())

        """apply attention"""
        output_a = K.sum(input_a * K.expand_dims(attend_a_new), axis=1)
        output_b = K.sum(input_b * K.expand_dims(attend_b_new), axis=1)

        return output_a, output_b

    def max_co_attend(self, input_a, input_b, input_a_mask, input_b_mask):
        input_max_a = K.max(input_a, axis=1)
        input_max_b = K.max(input_b, axis=1)

        # attention over input_a with input_max_b
        attend_a = K.tanh(K.batch_dot(K.dot(input_a, self.w_a), input_max_b, axes=(2, 1)))
        attend_a = K.exp(attend_a)
        attend_a *= input_a_mask
        attend_a /= (K.sum(attend_a, axis=1, keepdims=True) + K.epsilon())

        # attendtion over input_b with input_max_a
        attend_b = K.tanh(K.batch_dot(K.dot(input_b, self.w_b), input_max_a, axes=(2, 1)))
        attend_b = K.exp(attend_b)
        attend_b *= input_b_mask
        attend_b /= (K.sum(attend_b, axis=1, keepdims=True) + K.epsilon())

        # apply attention
        output_a = K.sum(input_a * K.expand_dims(attend_a), axis=1)
        output_b = K.sum(input_b * K.expand_dims(attend_b), axis=1)

        return output_a, output_b

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        input_a_shape, input_b_shape = input_shape
        return [(input_a_shape[0], input_a_shape[-1]), (input_b_shape[0], input_b_shape[-1])]


class SingleSideAttention(Layer):
    """use one sentence representation to attend to another sentence representation. Support Masking."""
    def __init__(self, attend_type, return_sequence=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.attend_type = attend_type
        self.return_sequence = return_sequence
        self.initializer = initializer
        self.regularizer = regularizer
        self.constraint = constraint
        self.supports_masking = False
        super(SingleSideAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        input_a_shape, input_b_shape = input_shape
        if len(input_a_shape) != 2 or len(input_b_shape) != 3:
            raise ValueError('input should be 2D query tensor and 3D key tensor!')
        last_a, last_b = input_a_shape[-1], input_a_shape[-1]

        if self.attend_type == 'mul':
            self.w_mul = self.add_weight(self.name+'_w_mul', (last_b, last_a), initializer=self.initializer,
                                         regularizer=self.regularizer, constraint=self.constraint)
        elif self.attend_type == 'add':
            self.w_a = self.add_weight(self.name+'_w_a', (last_a, last_a), initializer=self.initializer,
                                       regularizer=self.regularizer, constraint=self.constraint)
            self.w_b = self.add_weight(self.name+'_w_b', (last_b, last_a), initializer=self.initializer,
                                       regularizer=self.regularizer, constraint=self.constraint)
            self.w_attend = self.add_weight(self.name+'w_attend', (last_a, 1), initializer=self.initializer,
                                            regularizer=self.regularizer, constraint=self.constraint)
        elif self.attend_type in ['dot', 'scaled_dot']:
            if last_a != last_b:
                raise ValueError('last dimension should be same when using dot product attention...')
        else:
            raise ValueError('attend_type not understood: {}'.format(self.attend_type))

        super(SingleSideAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_a, input_b = inputs
        input_mask = K.cast(K.any(input_b, 2), K.floatx())

        if self.attend_type == 'add':
            attend = K.dot(K.tanh(K.dot(input_b, self.w_b) + K.expand_dims(K.dot(input_a, self.w_a), 1)), self.w_attend)
            attend = K.exp(K.squeeze(attend, 2))
        elif self.attend_type == 'mul':
            attend = K.tanh(K.batch_dot(K.dot(input_b, self.w_mul), input_a, axes=(2, 1)))
        elif self.attend_type in ['dot', 'scaled_dot']:
            attend = K.batch_dot(input_b, input_a, axes=(2, 1))
            if self.attend_type == 'scaled_dot':
                attend *= K.int_shape(input_a)[-1] ** -0.5
        else:
            raise ValueError('attend_type not understood: {}'.format(self.attend_type))

        attend *= input_mask
        attend /= (K.sum(attend, axis=1, keepdims=True) + K.epsilon())
        output = input_b * K.expand_dims(attend, 2)
        if not self.return_sequence:
            output = K.sum(output, axis=1)
        return output

    def compute_mask(self, inputs, mask=None):
        pass

    def compute_output_shape(self, input_shape):
        input_a_shape, input_b_shape = input_shape
        if self.return_sequence:
            return input_b_shape
        else:
            return input_b_shape[0], input_b_shape[2]


class MultiHeadAttention(Layer):
    """
    Multi-head Attention introduced in Transformer, support masking
    """
    def __init__(self, num_units=100, num_heads=3, residual=True, normalize=True, initializer='orthogonal',
                 regularizer=None, constraint=None, **kwargs):
        self.num_units = num_units
        self.num_heads = num_heads
        self.model_units = self.num_units * self.num_heads
        self.residual = residual
        self.normalize = normalize
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)
        self.supports_masking = False
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        input_a_shape, mask_shape = input_shape
        if len(input_a_shape) != 3 or len(mask_shape) != 2:
            raise ValueError('Input into IntraSelfAttention should be a 3D input tensor & a 2D mask tensor')

        self.w_q = self.add_weight(name='w_q', shape=(input_a_shape[-1], self.model_units),
                                   initializer=self.initializer, regularizer=self.regularizer,
                                   constraint=self.constraint)
        self.w_k = self.add_weight(name='w_k', shape=(input_a_shape[-1], self.model_units),
                                   initializer=self.initializer, regularizer=self.regularizer,
                                   constraint=self.constraint)
        self.w_v = self.add_weight(name='w_v', shape=(input_a_shape[-1], self.model_units),
                                   initializer=self.initializer, regularizer=self.regularizer,
                                   constraint=self.constraint)
        self.w_final = self.add_weight(name='w_v', shape=(self.model_units, self.model_units),
                                       initializer=self.initializer, regularizer=self.regularizer,
                                       constraint=self.constraint)
        if self.normalize:
            self.gamma = self.add_weight(name='gamma', shape=(self.model_units, ), initializer='one',
                                         regularizer=self.regularizer, constraint=self.constraint)
            self.beta = self.add_weight(name='beta', shape=(self.model_units, ), initializer='zero',
                                        regularizer=self.regularizer, constraint=self.constraint)
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        input_a, input_mask = inputs

        """convert to query, key, value vectors, shaped [batch_size*num_head, time_step, embed_dim]"""
        multihead_query = K.concatenate(tf.split(K.dot(input_a, self.w_q), self.num_heads, axis=2), axis=0)
        multihead_key = K.concatenate(tf.split(K.dot(input_a, self.w_k), self.num_heads, axis=2), axis=0)
        multihead_value = K.concatenate(tf.split(K.dot(input_a, self.w_v), self.num_heads, axis=2), axis=0)

        """scaled dot product"""
        scaled = K.int_shape(input_a)[-1] ** -0.5
        attend = K.batch_dot(multihead_query, multihead_key, axes=2) * scaled   # [batch_size*num_head, time_step, time_step]
        # apply mask before normalization (softmax)
        multihead_mask = K.tile(input_mask, [self.num_heads, 1])
        attend *= K.expand_dims(K.cast(multihead_mask, K.floatx()), 2)
        attend *= K.expand_dims(K.cast(multihead_mask, K.floatx()), 1)
        # normalization
        attend = attend / K.cast(K.sum(attend, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        # apply attention
        attend = K.batch_dot(attend, multihead_value, axes=(2, 1))  # [batch_size*num_head, time_step, embed_dim]
        attend = tf.concat(tf.split(attend, self.num_heads, axis=0), axis=2)
        attend = K.dot(attend, self.w_final)

        if self.residual:
            attend = attend + input_a
        if self.normalize:
            mean = K.mean(attend, axis=-1, keepdims=True)
            std = K.mean(attend, axis=-1, keepdims=True)
            attend = self.gamma * (attend - mean) / (std + K.epsilon()) + self.beta

        return attend

    def compute_output_shape(self, input_shape):
        input_a_shape, mask_shape = input_shape
        return input_a_shape[0], input_a_shape[1], self.num_units*self.num_heads








