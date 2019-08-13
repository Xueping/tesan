import tensorflow as tf
import math
from src.nn_utils.general import exp_mask_for_high_rank, mask_for_high_rank
from src.nn_utils.nn import linear, dropout, bn_dense_layer,get_logits,softsel


# # ----------------------fundamental-----------------------------
def scaled_tanh(x, scale=5.):
    return scale * tf.nn.tanh(1. / scale * x)


def traditional_attention(rep_tensor, rep_mask, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'traditional_attention'):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)

        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        return attn_res


def multi_dimensional_attention(rep_tensor, rep_mask, keep_prob=1., is_train=None, wd=0., activation='relu'):
    # bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope('multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        return attn_output


def directional_attention_with_dense(
        rep_tensor, rep_mask, direction=None, scope=None,
        keep_prob=1., is_train=None, wd=0., activation='elu',
        tensor_dict=None, name=None, hn=None):

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

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
            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


def normal_attention(rep_tensor, rep_mask, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None):
    batch_size, code_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'normal_attention'):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)

        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_result = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_tensor_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_tensor_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)# bs,sl,vec
        return output


def delta_with_dense(rep_tensor, rep_mask, delta_tensor, keep_prob=1.,
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


def temporal_date_sa_with_dense(rep_tensor, rep_mask, date_tensor, keep_prob=1.,
                                      is_train=None, wd=0., activation='relu', hn=None, is_scale=True):

        batch_size, code_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
        ivec = rep_tensor.get_shape().as_list()[2]
        ivec = hn or ivec
        with tf.variable_scope('temporal_attention'):
            # mask generation
            attn_mask = tf.cast(tf.diag(- tf.ones([code_len], tf.int32)) + 1, tf.bool)# batch_size, code_len, code_len

            # non-linear for context
            rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                     False, wd, keep_prob, is_train)
            rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, code_len, 1, 1])  # bs,sl,sl,vec
            rep_map_dp = dropout(rep_map, keep_prob, is_train)

            # non-linear for date
            date_rep_map = bn_dense_layer(date_tensor, ivec, True, 0., 'bn_dense_map_time', activation,
                                          False, wd, keep_prob, is_train)  # bs,sl,sl,vec
            date_rep_map_dp = dropout(date_rep_map, keep_prob, is_train)

            # attention
            with tf.variable_scope('attention'):  # bs,sl,sl,vec
                f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
                dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  #batch_size, code_len, vec_size
                dependent_etd = tf.expand_dims(dependent, 1)  #batch_size, code_len,code_len, vec_size
                head = linear(rep_map_dp, ivec, False, scope='linear_head')  #batch_size, code_len, vec_size
                head_etd = tf.expand_dims(head, 2)  #batch_size, code_len,code_len, vec_size

                date_dependent = linear(date_rep_map_dp, ivec, False, scope='linear_date_dependent')  # bs,sl,vec
                date_dependent_etd = tf.expand_dims(date_dependent, 1)

                date_head = linear(date_rep_map_dp, ivec, False, scope='linear_date_head')  # bs,sl,vec
                date_head_etd = tf.expand_dims(date_head, 2)

                # logits = scaled_tanh(dependent_etd + head_etd + date_dependent_etd + date_head_etd + f_bias, 5.0)
                attention_fact = dependent_etd + head_etd + date_dependent_etd + date_head_etd + f_bias
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
                o_bias = tf.get_variable('o_bias', [ivec], tf.float32, tf.constant_initializer(0.))
                # input gate
                fusion_gate = tf.nn.sigmoid(
                    linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                    linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                    o_bias)
                output = fusion_gate * rep_map + (1 - fusion_gate) * attn_result
                output = mask_for_high_rank(output, rep_mask)  # bs,sl,vec

            return output


def time_aware_attention(train_inputs, embed, mask, embedding_size, k):
    with tf.variable_scope('time_aware_attention'):
        attn_weights = tf.Variable(tf.truncated_normal([embedding_size, k], stddev=1.0 / math.sqrt(k)))
        attn_biases = tf.Variable(tf.zeros([k]))

        # weight add bias
        attn_embed = tf.nn.bias_add(attn_weights, attn_biases)

        # multiplying it with Ei
        attn_scalars = tf.tensordot(embed, attn_embed, axes=[[2], [0]])

        # get abs of distance
        train_delta = tf.abs(train_inputs[:, :, 1])

        # distance function is log(dist+1)
        dist_fun = tf.log(tf.to_float(train_delta) + 1.0)

        # reshape the dist_fun
        dist_fun = tf.reshape(dist_fun, [tf.shape(dist_fun)[0], tf.shape(dist_fun)[1], 1])

        # the attribution logits
        attn_logits = tf.multiply(attn_scalars, dist_fun)

        # the attribution logits sum
        attn_logits_sum = tf.reduce_sum(attn_logits, -1, keepdims=True)
        attn_logits_sum = exp_mask_for_high_rank(attn_logits_sum, mask)

        # get weights via softmax
        attn_softmax = tf.nn.softmax(attn_logits_sum, 1)

        # the weighted sum
        attn_embed_weighted = tf.multiply(attn_softmax, embed)
        attn_embed_weighted = mask_for_high_rank(attn_embed_weighted, mask)

        reduced_embed = tf.reduce_sum(attn_embed_weighted, 1)
        # obtain two scalars
        scalar1 = tf.log(tf.to_float(tf.shape(embed)[1]) + 1.0)
        scalar2 = tf.reduce_sum(tf.pow(attn_softmax, 2), 1)
        # the scalared embed
        reduced_embed = tf.multiply(reduced_embed, scalar1)
        reduced_embed = tf.multiply(reduced_embed, scalar2)

        return reduced_embed, attn_embed_weighted


def fusion_gate(rep1,rep2,wd, keep_prob, is_train):
    ivec = rep1.get_shape().as_list()[1]
    with tf.variable_scope('output'):
        o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
        # input gate
        fusion_g = tf.nn.sigmoid(
            linear(rep1, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
            linear(rep2, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
            o_bias)
        output = fusion_g * rep1 + (1-fusion_g) * rep2
    return output


def visit_temporal_date_sa_with_dense(rep_tensor, date_tensor,
                                      keep_prob=1.,is_train=None,
                                      wd=0., activation='relu',
                                      hn=None, is_scale=True,
                                      is_plus_date=True, is_plus_sa=True):

    batch_size, sw_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope('temporal_attention'):
        # mask generation
        attn_mask = tf.cast(tf.diag(- tf.ones([sw_len], tf.int32)) + 1, tf.bool)  # batch_size, code_len, code_len

        # non-linear for context
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sw_len, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # non-linear for date
        date_rep_map = bn_dense_layer(date_tensor, ivec, True, 0., 'bn_dense_map_time', activation,
                                      False, wd, keep_prob, is_train)  # bs,sl,sl,vec
        date_rep_map_dp = dropout(date_rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec

            f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # batch_size, code_len, vec_size
            dependent_etd = tf.expand_dims(dependent, 1)  # batch_size, code_len,code_len, vec_size
            head = linear(rep_map_dp, ivec, False, scope='linear_head')  # batch_size, code_len, vec_size
            head_etd = tf.expand_dims(head, 2)  # batch_size, code_len,code_len, vec_size

            date_dependent = linear(date_rep_map_dp, ivec, False, scope='linear_date_dependent')  # bs,sl,vec
            date_dependent_etd = tf.expand_dims(date_dependent, 1)
            date_head = linear(date_rep_map_dp, ivec, False, scope='linear_date_head')  # bs,sl,vec
            date_head_etd = tf.expand_dims(date_head, 2)

            if is_plus_sa and is_plus_date:
                attention_fact = dependent_etd + head_etd + date_dependent_etd + date_head_etd + f_bias
            elif is_plus_date:
                # logits = scaled_tanh(dependent_etd + head_etd + date_dependent_etd + date_head_etd + f_bias, 5.0)
                attention_fact = date_dependent_etd + date_head_etd + f_bias
            elif is_plus_sa:
                attention_fact = dependent_etd + head_etd + f_bias
            else:
                return rep_map

            if is_scale:
                logits = scaled_tanh(attention_fact, 5.0)  # bs,sl,sl,vec
            else:
                logits = linear(tf.nn.tanh(attention_fact), ivec, True, scope='linear_attn_fact')

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
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

        return output


def visit_sa_with_dense(rep_tensor, keep_prob=1.,is_train=None, wd=0.,
                        activation='relu', hn=None, is_scale=True, is_plus_sa=True):

    batch_size, sw_len, vec_size = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope('temporal_attention'):
        # mask generation
        attn_mask = tf.cast(tf.diag(- tf.ones([sw_len], tf.int32)) + 1, tf.bool)  # batch_size, code_len, code_len

        # non-linear for context
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sw_len, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec

            f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # batch_size, code_len, vec_size
            dependent_etd = tf.expand_dims(dependent, 1)  # batch_size, code_len,code_len, vec_size
            head = linear(rep_map_dp, ivec, False, scope='linear_head')  # batch_size, code_len, vec_size
            head_etd = tf.expand_dims(head, 2)  # batch_size, code_len,code_len, vec_size

            if is_plus_sa:
                attention_fact = dependent_etd + head_etd + f_bias
            else:
                return rep_map

            if is_scale:
                logits = scaled_tanh(attention_fact, 5.0)  # bs,sl,sl,vec
            else:
                logits = linear(tf.nn.tanh(attention_fact), ivec, True, scope='linear_attn_fact')

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
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

        return output


def visit_multi_dimensional_attention(rep_tensor, keep_prob=1., is_train=None, wd=0., activation='relu'):
    # bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]

    with tf.variable_scope('multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)

        soft = tf.nn.softmax(map2, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        return attn_output


def first_level_sa(rep_tensor, rep_mask, keep_prob=1., is_train=None, wd=0., activation='relu'):
    # bs, sw, cl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2], tf.shape(rep_tensor)[3]
    ivec = rep_tensor.get_shape()[3]
    with tf.variable_scope('first_level_sa'):
        print('original: ',rep_tensor.get_shape())
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,False, wd, keep_prob, is_train)
        print('map1: ',map1.get_shape())
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',False, wd, keep_prob, is_train)
        print('map2: ',map2.get_shape())
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 2)  # bs,sk,code_len,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 2)  # bs, sk, vec

        return attn_output

