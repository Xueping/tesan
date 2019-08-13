import tensorflow as tf
import math

from src.nn_utils.attention import multi_dimensional_attention, self_attention_with_dense,\
    temporal_delta_sa_with_dense, bn_dense_layer,time_aware_attention,\
    temporal_date_sa_with_dense, delta_with_dense
from src.embedding.concept.ablation_study import normal_attention
from src.nn_utils.general import mask_for_high_rank
from src.template.model import ModelTemplate


class ConceptModel(ModelTemplate):
    def __init__(self,scope,dataset):
        super(ConceptModel, self).__init__(scope,dataset)

        # ------ start ------
        self.context_fusion = None

        self.code_embeddings = None
        self.final_embeddings = None

        self.nce_weights = None
        self.final_weights = None

        self.final_wgt_sim = None
        self.final_emb_sim = None

        self.context_dates = None
        self.train_masks = None

        # ---- place holder -----
        if self.is_date_encoding:
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, None, 3], name='train_inputs')
            self.context_dates = self.train_inputs[:, :, 2]
        else:
            self.train_inputs = tf.placeholder(tf.int32, shape=[None, None, 2], name='train_inputs')
            self.train_masks = tf.placeholder(tf.int32, shape=[None, None, None], name='train_masks')

        self.train_labels = tf.placeholder(tf.int32, shape=[None, 1], name='train_labels')
        self.valid_dataset = tf.constant(self.valid_samples, dtype=tf.int32, name='valid_samples')

        # ------------ other ---------
        self.output_class = 3  # 0 for contradiction, 1 for neural and 2 for entailment
        self.batch_size = tf.shape(self.train_inputs)[0]
        self.code_len = tf.shape(self.train_inputs)[1]

        # context codes
        self.context_codes = self.train_inputs[:, :, 0]

        # mask for padding codes are all 0, actual codes are 1
        self.context_mask = tf.cast(self.context_codes, tf.bool)

        # time interval between context code and label code
        self.context_delta = self.train_inputs[:, :, 1]

        #building model and other parts
        self.context_fusion, self.code_embeddings = self.build_network()
        self.loss, self.optimizer, self.nce_weights = self.build_loss_optimizer()
        self.final_embeddings, self.final_weights = self.build_embedding()
        self.final_emb_sim, self.final_wgt_sim = self.build_similarity()

    def build_loss_optimizer(self):

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        losses = tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=self.train_labels,
            inputs=self.context_fusion,
            num_sampled=self.num_samples,
            num_classes=self.vocabulary_size)

        # loss = tf.reduce_mean(losses, name='loss_mean')
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)

        optimizer = tf.train.AdamOptimizer().minimize(loss)
        return loss, optimizer, nce_weights

    def build_accuracy(self):
        pass

    def build_network(self):
        # Look up embeddings for inputs.
        with tf.name_scope('code_embeddings'):
            init_code_embed = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
            code_embeddings = tf.Variable(init_code_embed)
            context_embed = tf.nn.embedding_lookup(code_embeddings, self.context_codes)

        if self.model_type == 'tesa':
            with tf.name_scope(self.model_type):
                # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]
                init_date_embed = tf.random_uniform([self.dates_size, self.embedding_size], -1.0, 1.0)
                date_embeddings = tf.Variable(init_date_embed)

                if self.is_date_encoding:
                    date_embed = tf.nn.embedding_lookup(date_embeddings, self.context_dates)

                    # self_attention
                    cntxt_embed = temporal_date_sa_with_dense(rep_tensor=context_embed,
                                                              rep_mask=self.context_mask,
                                                              date_tensor=date_embed,
                                                              is_train=True,
                                                              activation=self.activation,
                                                              is_scale = self.is_scale)
                else:
                    date_embed = tf.nn.embedding_lookup(date_embeddings, self.train_masks)

                    # self_attention
                    cntxt_embed = temporal_delta_sa_with_dense(rep_tensor=context_embed,
                                                               rep_mask=self.context_mask,
                                                               delta_tensor = date_embed,
                                                               is_train=True,
                                                               activation=self.activation,
                                                               is_scale = self.is_scale)

                # Attention pooling
                context_fusion = multi_dimensional_attention(cntxt_embed,self.context_mask,is_train=True)

        elif self.model_type == 'delta':
            with tf.name_scope(self.model_type):
                #self_attention
                init_date_embed = tf.random_uniform([self.dates_size, self.embedding_size], -1.0, 1.0)
                date_embeddings = tf.Variable(init_date_embed)
                date_embed = tf.nn.embedding_lookup(date_embeddings, self.train_masks)
                cntxt_embed = delta_with_dense(rep_tensor=context_embed,
                                                        rep_mask=self.context_mask,
                                                        delta_tensor=date_embed,
                                                        is_train=True,
                                                        activation=self.activation,
                                                        is_scale = self.is_scale)

                # attention pooling
                context_fusion = multi_dimensional_attention(cntxt_embed,self.context_mask,is_train=True)

        elif self.model_type == 'sa':
            with tf.name_scope(self.model_type):
                #self_attention
                cntxt_embed = self_attention_with_dense(rep_tensor=context_embed,
                                                        rep_mask=self.context_mask,
                                                        is_train=True,
                                                        activation=self.activation,
                                                        is_scale = self.is_scale)

                # attention pooling
                context_fusion = multi_dimensional_attention(cntxt_embed,self.context_mask,is_train=True)

        elif self.model_type == 'normal':
            with tf.name_scope(self.model_type):
                #self_attention
                cntxt_embed = normal_attention(rep_tensor=context_embed,
                                                        rep_mask=self.context_mask,
                                                        is_train=True,
                                                        activation=self.activation)

                # attention pooling
                context_fusion = multi_dimensional_attention(cntxt_embed,self.context_mask,is_train=True)

        elif self.model_type == 'cbow':
            with tf.name_scope(self.model_type):
                cntxt_embed = mask_for_high_rank(context_embed, self.context_mask)# bs,sl,vec
                context_fusion = tf.reduce_mean(cntxt_embed, 1)

        elif self.model_type == 'ta_attn':
            context_fusion = time_aware_attention(self.train_inputs,context_embed,self.context_mask,self.embedding_size,k=100)

        elif self.model_type == 'fusion':
            with tf.name_scope(self.model_type):
                # self-attention
                code2code = self_attention_with_dense(rep_tensor=context_embed, rep_mask=self.context_mask,
                                                      is_train=True,activation=self.activation)
                # attention pooling
                source2code = multi_dimensional_attention(code2code,self.context_mask,is_train=True)
                # time-aware attention
                ta_attn_res = time_aware_attention(self.train_inputs,context_embed,self.context_mask,self.embedding_size,k=100)

                ivec = ta_attn_res.get_shape()[1]
                concat_context = tf.concat([source2code, ta_attn_res], 1)

                # context_fusion = fusion_gate(source2code,ta_attn_res,wd=0., keep_prob=1., is_train=True)
                context_fusion = bn_dense_layer(concat_context,ivec,True, 0., 'bn_dense_map', self.activation,
                                        False, wd=0., keep_prob=1., is_train=True)
        return context_fusion, code_embeddings

    def build_embedding(self):
        with tf.name_scope('build_embedding'):
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.code_embeddings), 1, keepdims =True))
            final_embeddings = self.code_embeddings / norm

            weights_norm = tf.sqrt(tf.reduce_sum(tf.square(self.nce_weights), 1, keepdims=True))
            final_weights = self.nce_weights / weights_norm
        return final_embeddings, final_weights

    def build_similarity(self):
        with tf.name_scope('build_similarity'):
            valid_embeddings = tf.nn.embedding_lookup(self.final_embeddings, self.valid_dataset)
            final_emb_sim = tf.matmul(valid_embeddings, self.final_embeddings, transpose_b=True)

            valid_embeddings = tf.nn.embedding_lookup(self.final_weights, self.valid_dataset)
            final_wgt_sim = tf.matmul(valid_embeddings, self.final_weights, transpose_b=True)

        return final_emb_sim, final_wgt_sim

