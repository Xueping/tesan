import tensorflow as tf
import numpy as np
from os.path import join
from src.utils.configs import cfg
from src.embedding.concept.evaluation import ConceptEvaluation as Evaluator
from src.embedding.concept.model import ConceptModel as Model

from src.utils.graph_handler import GraphHandler
from src.utils.record_log import RecordLog
from src.embedding.concept.dataset import  ConceptDataset as CDataset
from src.embedding.concept.datasetEncodeDate import ConceptAndDateDataset as CDDataset

import warnings
warnings.filterwarnings('ignore')

logging = RecordLog()


def train():

    if cfg.gpu_mem is None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem, allow_growth=True)
        graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem)
        graph_config = tf.ConfigProto(gpu_options=gpu_options)

    num_steps = cfg.num_steps

    if cfg.is_date_encoding:
        data_set = CDDataset()
        data_set.prepare_data(cfg.visit_threshold)
        data_set.build_dictionary()
        data_set.build_dict4date()
        data_set.load_data()
        sample_batches = data_set.generate_batch(num_steps)
    else:
        data_set = CDataset()
        data_set.prepare_data(cfg.visit_threshold)
        data_set.build_dictionary()
        data_set.load_data()
        sample_batches = data_set.generate_batch(num_steps)
        print(data_set.train_size)
        batch_num = data_set.train_size / data_set.batch_size
        print(batch_num)

    sess = tf.Session(config=graph_config)
    with tf.variable_scope('concept_embedding') as scope:
        model = Model(scope.name,data_set)

    graph_handler = GraphHandler(model,logging)
    graph_handler.initialize(sess)
    evaluator = Evaluator(model,logging)

    global_step = 0
    total_loss = 0

    epoch_loss = 0
    tmp_epoch = 0
    tmp_cur_batch = 0

    logging.add()
    logging.add('Begin training...')
    for batch in sample_batches:
        if num_steps is not None: # run based on step number
            if cfg.is_date_encoding:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[1]}
            else:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[2], model.train_masks: batch[1]}
            _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            total_loss += loss_val
            global_step += 1

            if global_step % 5000 == 0:
                avg_loss = total_loss / 1000
                log_str = "Average loss at step %s: %s " % (global_step, avg_loss)
                logging.add(log_str)
                total_loss = 0

                icd_nns = evaluator.get_nns_p_at_top_k(sess, 'ICD')
                icd_weigh_scores = evaluator.get_clustering_nmi(sess, 'ICD')
                ccs_nns = evaluator.get_nns_p_at_top_k(sess, 'CCS')
                ccs_weigh_scores = evaluator.get_clustering_nmi(sess, 'CCS')

                logging.add('validating the embedding performance .....')
                log_str = "weight: %s %s %s %s" % (icd_weigh_scores, ccs_weigh_scores, icd_nns, ccs_nns)
                logging.add(log_str)
        else: # run based on epoch number
            if cfg.is_date_encoding:
                batch_num, current_epoch, current_batch = batch[2], batch[3], batch[4]
            else:
                batch_num, current_epoch, current_batch = batch[3], batch[4], batch[5]

            if tmp_epoch != current_epoch:
                epoch_loss /= tmp_cur_batch
                log_str = "Average loss at epoch %s: %s " % (tmp_epoch, epoch_loss)
                logging.add(log_str)
                epoch_loss = 0
                tmp_epoch = current_epoch

                icd_nns = evaluator.get_nns_p_at_top_k(sess, 'ICD')
                icd_weigh_scores = evaluator.get_clustering_nmi(sess, 'ICD')
                ccs_nns = evaluator.get_nns_p_at_top_k(sess, 'CCS')
                ccs_weigh_scores = evaluator.get_clustering_nmi(sess, 'CCS')

                print('validating the embedding performance .....')
                log_str = "weight: %s %s %s %s" % (icd_weigh_scores, ccs_weigh_scores, icd_nns, ccs_nns)
                logging.add(log_str)

            else:
                tmp_cur_batch = current_batch

            if current_epoch == model.max_epoch:
                embeddings = sess.run(model.final_weights)

                path = cfg.data_source + '_model_' + cfg.model + '_epoch_' + \
                       str(cfg.max_epoch) + '_sk_' + str(cfg.skip_window) + '.vect'
                np.savetxt(join(cfg.saved_vect_dir, path), embeddings, delimiter=',')
                break

            if cfg.is_date_encoding:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[1]}
            else:
                feed_dict = {model.train_inputs: batch[0], model.train_labels: batch[2], model.train_masks: batch[1]}

            _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            epoch_loss += loss_val

    logging.done()


def test():
    pass


def main(_):
    if cfg.mode == 'train':
        train()
    elif cfg.mode == 'test':
        test()
    else:
        raise RuntimeError('no running mode named as %s' % cfg.mode)


def output_model_params():
    logging.add()
    logging.add('==>model_title: ' + cfg.model_name[1:])
    logging.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            logging.add('%s: %s' % (key, value))


if __name__ == '__main__':
    tf.app.run()
    # --data_source mimic3 --model delta --gpu 2 --max_epoch 30 --num_steps 10000 --train_batch_size 64 --num_samples 10 --reduced_window True --skip_window 6 --verbose True --is_scale False --is_date_encoding False --task embedding
# --data_source mimic3 --model sa --gpu 1 --max_epoch 30 --train_batch_size 64 --num_samples 10 --reduced_window True --skip_window 6 --verbose True --is_scale False --is_date_encoding False --task embedding --visit_threshold 1
