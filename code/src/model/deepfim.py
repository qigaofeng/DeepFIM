# -*- coding:utf-8 -*-

import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomNormal, glorot_normal, Zeros
from tensorflow.python.keras.layers import (Dense, Embedding, Lambda, add,
                                            multiply, Layer)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from src.input_embedding import (create_singlefeat_inputdict,
                                 get_embedding_vec_list, get_inputs_list,
                                 get_linear_logit)
from src.layers.core import MLP, PredictionLayer
from src.utils import check_feature_config_dict
from tensorflow.python.keras.layers import Layer, Concatenate


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)


class ATT(Layer):
    """
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

      Arguments
        - **attention_factor** : Positive integer, dimensionality of the
         attention network output space.

        - **l2_reg_w** : float between 0 and 1. L2 regularizer strength
         applied to attention network.

        - **keep_prob** : float between 0 and 1. Fraction of the attention net output units to keep.

        - **seed** : A Python integer to use as random seed.
    """

    def __init__(self, attention_factor=4, l2_reg_w=0, keep_prob=1.0, seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.keep_prob = keep_prob
        self.seed = seed
        super(ATT, self).__init__(**kwargs)

    def build(self, input_shape):
        embedding_size = input_shape[-1].value

        self.attention_W = self.add_weight(shape=(embedding_size,
                                                  self.attention_factor), initializer=glorot_normal(seed=self.seed),
                                           regularizer=l2(self.l2_reg_w), name="attention_W")
        self.attention_b = self.add_weight(
            shape=(self.attention_factor,), initializer=Zeros(), name="attention_b")
        self.projection_h = self.add_weight(shape=(self.attention_factor, 1),
                                            initializer=glorot_normal(seed=self.seed), name="projection_h")
        self.projection_p = self.add_weight(shape=(
            embedding_size, 1), initializer=glorot_normal(seed=self.seed), name="projection_p")

        # Be sure to call this somewhere!
        super(ATT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inner_product = inputs  # concat_fun(ans,axis=1)

        bi_interaction = inner_product

        attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
            bi_interaction, self.attention_W, axes=(-1, 0)), self.attention_b))
        #  Dense(self.attention_factor,'relu',kernel_regularizer=l2(self.l2_reg_w))(bi_interaction)
        self.normalized_att_score = tf.nn.softmax(tf.tensordot(
            attention_temp, self.projection_h, axes=(-1, 0)), dim=1)
        attention_output = tf.reduce_sum(
            self.normalized_att_score * bi_interaction, axis=1)

        attention_output = tf.nn.dropout(
            attention_output, self.keep_prob, seed=1024)
        # Dropout(1-self.keep_prob)(attention_output)
        attention_output = tf.tensordot(
            attention_output, self.projection_p, axes=(-1, 0))
        return attention_output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `AFMLayer` layer should be called '
                             'on a list of inputs.')
        return (None, self.attention_factor)

    def get_config(self, ):
        config = {'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w, 'keep_prob': self.keep_prob, 'seed': self.seed}
        base_config = super(ATT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class INT(Layer):
    def __init__(self, attention_factor=4, l2_reg_w=0, keep_prob=1.0, seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.keep_prob = keep_prob
        self.seed = seed
        super(INT, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('layer should be called '
                             'on a list of at least 2 inputs')

        shape_set = set()
        reduced_input_shape = [shape.as_list() for shape in input_shape]
        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_input_shape[i]))

        if len(shape_set) > 1:
            raise ValueError('A layer requires '
                             'inputs with same shapes '
                             'Got different shapes: %s' % (shape_set))

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A layer requires '
                             'inputs of a list with same shape tensor like\
                             (None, 1, embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))

        embedding_size = input_shape[0][-1].value
        filed_num = len(input_shape)
        # print(embedding_size,filed_num)

        self.filed_W = self.add_weight(shape=(filed_num,
                                              self.attention_factor), initializer=glorot_normal(seed=self.seed),
                                       regularizer=l2(self.l2_reg_w), name="attention_W")

        # Be sure to call this somewhere!
        super(INT, self).build(input_shape)

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embeds_vec_list = inputs

        bi_interaction = []
        for r, c in itertools.combinations(range(len(embeds_vec_list)), 2):
            field_score = tf.reduce_sum(self.filed_W[r] * self.filed_W[c])

            embed = embeds_vec_list[r] * embeds_vec_list[c]
            bi_interaction.append(field_score * embed)

        out = concat_fun(bi_interaction, axis=1)

        return out

    def compute_output_shape(self, input_shape):

        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        input_shape = input_shape[0]
        embed_size = input_shape[-1]
        # return [(input_shape[0],1,embed_size) for _ in range(num_pairs)]
        if False:
            return (input_shape[0], num_pairs, 1)
        else:
            return (input_shape[0], num_pairs, embed_size)

    def get_config(self, ):
        config = {'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w, 'keep_prob': self.keep_prob, 'seed': self.seed}
        base_config = super(INT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def DeepFIM(feature_dim_dict, embedding_size=4, hidden_size=(128, 128),
            l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_deep=0,
            init_std=0.0001, seed=1024, final_activation='sigmoid', include_linear=True, use_bn=True, reduce_sum=False,
            pooling_method=True, keep_prob=1, att_factor=4):
    """

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part.
    :param l2_reg_deep: float . L2 regularizer strength applied to deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param include_linear: bool,whether include linear term or not
    :param use_bn: bool,whether use bn after ffm out or not
    :param reduce_sum: bool,whether apply reduce_sum on cross vector
    :return: A Keras model instance.
    """
    global fim_out, final_logit
    check_feature_config_dict(feature_dim_dict)
    if 'sequence' in feature_dim_dict and len(feature_dim_dict['sequence']) > 0:
        raise ValueError("now sequence input is not supported in NFFM")

    sparse_input_dict, dense_input_dict = create_singlefeat_inputdict(
        feature_dim_dict)

    Inter_sparse_embedding, Inter_dense_embedding, \
    Intra_sparse_embedding, Intra_dense_embedding, linear_embedding = get_embeddings(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear)

    # Inter-interaction
    deep_emb_list = get_embedding_vec_list(
        Intra_sparse_embedding, sparse_input_dict)
    Inter_embed_list = INT(att_factor)(deep_emb_list)

    # Intra-interaction
    Intra_embed_list = []
    for i, j in itertools.combinations(feature_dim_dict['sparse'], 2):
        element_wise_prod = multiply([Intra_sparse_embedding[i.name][j.name](
            sparse_input_dict[i.name]), Intra_sparse_embedding[j.name][i.name](sparse_input_dict[j.name])])
        if reduce_sum:
            element_wise_prod = Lambda(lambda element_wise_prod: K.sum(
                element_wise_prod, axis=-1))(element_wise_prod)
        Intra_embed_list.append(element_wise_prod)
    for i, j in itertools.combinations(feature_dim_dict['dense'], 2):
        element_wise_prod = multiply([Intra_dense_embedding[i.name][j.name](
            dense_input_dict[i.name]), Intra_dense_embedding[j.name][i.name](dense_input_dict[j.name])])
        if reduce_sum:
            element_wise_prod = Lambda(lambda element_wise_prod: K.sum(
                element_wise_prod, axis=-1))(element_wise_prod)
        Intra_embed_list.append(
            Lambda(lambda x: K.expand_dims(x, axis=1))(element_wise_prod))

    for i in feature_dim_dict['sparse']:
        for j in feature_dim_dict['dense']:
            element_wise_prod = multiply([Intra_sparse_embedding[i.name][j.name](sparse_input_dict[i.name]),
                                          Intra_dense_embedding[j.name][i.name](dense_input_dict[j.name])])

            if reduce_sum:
                element_wise_prod = Lambda(lambda element_wise_prod: K.sum(element_wise_prod, axis=-1))(
                    element_wise_prod)
            Intra_embed_list.append(element_wise_prod)
    Intra_embed_list = concat_fun(Intra_embed_list, axis=1)

    # fim layer
    fim = Inter_embed_list * Intra_embed_list

    # attention layer
    if pooling_method == 'att':
        fim_out = ATT(att_factor, keep_prob)(fim)
    else:
        print("error!")

    if use_bn:
        fim_out = tf.keras.layers.BatchNormalization()(Inter_embed_list)
    deep_out = MLP(hidden_size, l2_reg=l2_reg_deep)(fim_out)
    deep_logit = Dense(1, use_bias=False)(deep_out)

    linear_emb_list = get_embedding_vec_list(
        linear_embedding, sparse_input_dict)

    linear_logit = get_linear_logit(
        linear_emb_list, dense_input_dict, l2_reg_linear)

    if include_linear:
        final_logit = add([deep_logit, fim_out, linear_logit])
    output = PredictionLayer(final_activation)(final_logit)

    inputs_list = get_inputs_list(
        [sparse_input_dict, dense_input_dict])
    model = Model(inputs=inputs_list, outputs=output)
    return model


def get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    # Inter-field
    Inter_sparse_embedding = {j.name: {feat.name: Embedding(j.dimension, embedding_size,
                                                            embeddings_initializer=RandomNormal(
                                                                mean=0.0, stddev=0.0001, seed=seed),
                                                            embeddings_regularizer=l2(
                                                                l2_rev_V),
                                                            name='sparse_emb_' + str(j.name) + '_' + str(
                                                                i) + '-' + feat.name) for i, feat in
                                       enumerate(feature_dim_dict["sparse"] + feature_dim_dict['dense'])} for j in
                              feature_dim_dict["sparse"]}

    Inter_dense_embedding = {
        j.name: {feat.name: Dense(embedding_size, kernel_initializer=RandomNormal(mean=0.0, stddev=0.0001,
                                                                                  seed=seed), use_bias=False,
                                  kernel_regularizer=l2(l2_rev_V), name='sparse_emb_' + str(j.name) + '_' + str(
                i) + '-' + feat.name) for i, feat in
                 enumerate(feature_dim_dict["sparse"] + feature_dim_dict["dense"])} for j in
        feature_dim_dict["dense"]}

    # Inter-field
    Intra_sparse_embedding = {j.name: Embedding(j.dimension, embedding_size,
                                                embeddings_initializer=RandomNormal(
                                                    mean=0.0, stddev=0.0001, seed=seed),
                                                embeddings_regularizer=l2(
                                                    l2_rev_V),
                                                name='sparse_emb_' + str(j.name) + '_' + '-' + j.name) for j in
                              feature_dim_dict["sparse"]}

    Intra_dense_embedding = {j.name: Dense(embedding_size, kernel_initializer=RandomNormal(mean=0.0, stddev=0.0001,
                                                                                           seed=seed), use_bias=False,
                                           kernel_regularizer=l2(l2_rev_V), name='dense_emb_' + str(j.name) + '_' + str(
            j)) for j in
                             feature_dim_dict["dense"]}
    # linear part
    linear_embedding = {feat.name: Embedding(feat.dimension, 1,
                                             embeddings_initializer=RandomNormal(
                                                 mean=0.0, stddev=init_std, seed=seed),
                                             embeddings_regularizer=l2(
                                                 l2_reg_w),
                                             name='linear_emb_' + str(i) + '-' + feat.name) for
                        i, feat in enumerate(feature_dim_dict["sparse"])}

    return Inter_sparse_embedding, Inter_dense_embedding, \
           Intra_sparse_embedding, Intra_dense_embedding, linear_embedding
