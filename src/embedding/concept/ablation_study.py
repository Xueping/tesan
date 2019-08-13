import tensorflow as tf
from src.nn_utils.attention import scaled_tanh
from src.nn_utils.general import exp_mask_for_high_rank, mask_for_high_rank
from src.nn_utils.nn import linear, dropout, bn_dense_layer,get_logits,softsel


def temporal_delta_sa_with_dense(rep_tensor, rep_mask, delta_tensor, keep_prob=1.,
                                     is_train=None, wd=0., activation='relu', hn=None, is_scale=True):

    batch_size, code_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope('temporal_attention'):
        # mask generation
        attn_mask = tf.cast(tf.diag(- tf.ones([code_len], tf.int32)) + 1, tf.bool)  # batch_size, code_len, code_len

        # non-linear for context
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, code_len, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # non-linear for time interval
        time_rep_map = bn_dense_layer(delta_tensor, ivec, True, 0., 'bn_dense_map_time', activation,
                                 False, wd, keep_prob, is_train) # bs,sl,sl,vec
        time_rep_map_dp = dropout(time_rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))

            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec

            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            time_rep_etd = linear(time_rep_map_dp, ivec, False, scope='linear_time') # bs,sl,sl,vec
            # logits = scaled_tanh(dependent_etd + head_etd + time_rep_etd + f_bias, 5.0)  # bs,sl,sl,vec

            attention_fact = dependent_etd + head_etd + time_rep_etd + f_bias
            if is_scale:
                logits = scaled_tanh(attention_fact, 5.0)  # bs,sl,sl,vec
            else:
                fact_bias = tf.get_variable('fact_bias', [ivec], tf.float32, tf.constant_initializer(0.))
                logits = linear(tf.nn.tanh(attention_fact), ivec, False, scope='linear_attn_fact') + fact_bias

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)# bs,sl,vec

        return output


def self_attention_with_dense(rep_tensor, rep_mask, keep_prob=1.,
                                     is_train=None, wd=0., activation='relu', hn=None, is_scale=True):

    batch_size, code_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope('temporal_attention'):
        # mask generation
        attn_mask = tf.cast(tf.diag(- tf.ones([code_len], tf.int32)) + 1, tf.bool)  # batch_size, code_len, code_len

        # non-linear for context
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, code_len, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))

            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec

            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            attention_fact = dependent_etd + head_etd + f_bias
            if is_scale:
                logits = scaled_tanh(attention_fact, 5.0)  # bs,sl,sl,vec
            else:
                fact_bias = tf.get_variable('fact_bias', [ivec], tf.float32, tf.constant_initializer(0.))
                logits = linear(tf.nn.tanh(attention_fact), ivec, False, scope='linear_attn_fact')+fact_bias

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)# bs,sl,vec

        return output


def temporal_delta_with_dense(rep_tensor, rep_mask, delta_tensor, keep_prob=1.,
                                 is_train=None, wd=0., activation='relu', hn=None, is_scale=True):

    batch_size, code_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope('temporal_attention'):
        # mask generation
        attn_mask = tf.cast(tf.diag(- tf.ones([code_len], tf.int32)) + 1, tf.bool)  # batch_size, code_len, code_len

        # non-linear for context
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, code_len, 1, 1])  # bs,sl,sl,vec

        # non-linear for time interval
        time_rep_map = bn_dense_layer(delta_tensor, ivec, True, 0., 'bn_dense_map_time', activation,
                                 False, wd, keep_prob, is_train) # bs,sl,sl,vec
        time_rep_map_dp = dropout(time_rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))

            time_rep_etd = linear(time_rep_map_dp, ivec, False, scope='linear_time') # bs,sl,sl,vec

            attention_fact = time_rep_etd + f_bias
            if is_scale:
                logits = scaled_tanh(attention_fact, 5.0)  # bs,sl,sl,vec
            else:
                fact_bias = tf.get_variable('fact_bias', [ivec], tf.float32, tf.constant_initializer(0.))
                logits = linear(tf.nn.tanh(attention_fact), ivec, False, scope='linear_attn_fact') + fact_bias

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)# bs,sl,vec
        return output


def normal_attention(rep_tensor, rep_mask,keep_prob=1., is_train=None, wd=0., activation='elu'):

    batch_size, code_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.variable_scope('temporal_attention'):
        # mask generation
        attn_mask = tf.cast(tf.diag(- tf.ones([code_len], tf.int32)) + 1, tf.bool)  # batch_size, code_len, code_len

        # non-linear for context
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, code_len, 1, 1])  # bs,sl,sl,vec

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec

            align_scores = tf.matmul(rep_tensor, rep_tensor, transpose_b=True)  # [bs,slf,hn]*[bs,slt,hn]=>bs,slf,slt
            align_scores = tf.expand_dims(align_scores, -1)  # [bs,slf,slt,1]

            logits_masked = exp_mask_for_high_rank(align_scores, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1 - fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)  # bs,sl,vec

        return output
