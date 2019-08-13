import argparse
import os
from os.path import join


class Configs(object):
    def __init__(self):
        root_dir, _ = os.path.split(os.path.abspath(__file__))
        root_dir = os.path.dirname(root_dir)
        root_dir = os.path.dirname(root_dir)
        self.project_dir = root_dir
        self.icd_file = join(self.project_dir, 'src/utils/ontologies/D_ICD_DIAGNOSES.csv')
        self.ccs_file = join(self.project_dir, 'src/utils/ontologies/SingleDX-edit.txt')

        self.dataset_dir = join(self.project_dir, 'dataset', 'processed')
        self.standby_log_dir = self.mkdir(self.project_dir, 'logs')
        self.result_dir = self.mkdir(self.project_dir, 'outputs')
        self.all_model_dir = self.mkdir(self.result_dir, 'tasks')

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))

        # @ ----- control ----
        parser.add_argument('--debug', type='bool', default=False, help='whether run as debug mode')
        parser.add_argument('--mode', type=str, default='train', help='train, dev, or test')
        parser.add_argument('--model', type=str, default='tesa', help='tesa, vanila_sa, or cbow')
        parser.add_argument('--network_type', type=str, default='cbow', help='cbow or sg')
        parser.add_argument('--data_source', type=str, default='mimic3', help='mimic3 or cms')
        parser.add_argument('--gpu', type=int, default=0, help='eval_period')
        parser.add_argument('--gpu_mem', type=float, default=None, help='eval_period')
        parser.add_argument('--save_model', type='bool', default=True, help='save_model')
        parser.add_argument('--verbose', type='bool', default=False, help='print ...')
        parser.add_argument('--load_model', type='bool', default=False, help='load_model')
        parser.add_argument('--task', type=str, default='prediction', help='embedding or prediction')

        # @ ------------------RNN------------------
        parser.add_argument('--cell_type', type=str, default='gru', help='cell unit')
        parser.add_argument('--hn', type=int, default=100, help='number of hidden units')
        parser.add_argument('--time_steps', type=int, default=174, help='length of sequence')
        parser.add_argument('--num_per_step', type=int, default=200, help='number of per step')

        # @ ----------------Hierarchical TeSa----------------------
        parser.add_argument('--is_plus_date', type='bool', default=True, help='add temporal interval')
        parser.add_argument('--is_plus_sa', type='bool', default=True, help='add multi-dim self-attention')
        parser.add_argument('--task_type', type=str, default='none', help='type1: dx and readmission in future visit;type2: los and death in current visit')
        parser.add_argument('--predict_type', type=str, default='dx', help='dx:diagnosis; re:readmission,death: mortality, los: length of stay')

        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=20, help='Max Epoch Number')
        parser.add_argument('--train_batch_size', type=int, default=128, help='Train Batch Size')
        parser.add_argument('--skip_window', type=int, default=9, help='Skip window Size')
        parser.add_argument('--num_samples', type=int, default=5, help='Number of negative examples to sample') 
        parser.add_argument('--reduced_window', type='bool', default=True, help='reduced spik window')
        parser.add_argument('--is_date_encoding', type='bool', default=False, help='To control date encoding')
        parser.add_argument('--activation', type=str, default='relu', help='activation function')
        parser.add_argument('--is_scale', type='bool', default=True, help='to scale the attention facts')
        
        parser.add_argument('--num_steps', type=int, default=None, help='num_steps')
        parser.add_argument('--test_batch_size', type=int, default=100, help='Test Batch Size')
        parser.add_argument('--optimizer', type=str, default='adadelta', help='Test Batch Size')
        parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate')
        parser.add_argument('--dropout', type=float, default=0.75, help='')
        parser.add_argument('--wd', type=float, default=5e-5, help='weight decay factor')
        parser.add_argument('--var_decay', type=float, default=0.999, help='Learning rate')  # ema
        parser.add_argument('--decay', type=float, default=0.9, help='Learning rate')  # ema

        # @ ----- code Processing ----
        parser.add_argument('--sample_flag', type='bool', default=True, help='sample_flag')
        parser.add_argument('--sample_rate', type=float, default=1e-3, help='sample_rate')
        parser.add_argument('--embedding_size', type=int, default=100, help='code embedding size')
        parser.add_argument('--only_dx_flag', type='bool', default=True, help='only_dx_flag')
        parser.add_argument('--visit_threshold', type=int, default=4, help='visit_threshold')
        parser.add_argument('--min_cut_freq', type=int, default=5, help='min code frequency')

        # @ ------validatation-----
        parser.add_argument('--valid_size', type=int, default=1000, help='evaluate similarity size')
        parser.add_argument('--top_k', type=int, default=1, help='number of nearest neighbors')

        # -------Hierarchical Self-Attention for prediction ----------
        parser.add_argument('--hierarchical', type='bool', default=True, help='hierarchical attention')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))

        # ------- dataset --------
        self.valid_examples = list(range(1,self.valid_size+1))  # Only pick dev samples in the head of the distribution.

        #------------------path-------------------------------
        self.model_dir = self.mkdir(self.all_model_dir, self.task, self.model)
        self.summary_dir = self.mkdir(self.model_dir, 'summary')
        self.ckpt_dir = self.mkdir(self.model_dir, 'ckpt')
        self.log_dir = self.mkdir(self.model_dir, 'log_files')
        self.saved_vect_dir = self.mkdir(self.model_dir, 'vects')
        self.dict_dir = self.mkdir(self.result_dir, 'dict')
        self.processed_dir = self.mkdir(self.result_dir, 'processed_data')
        self.processed_task_dir = self.mkdir(self.processed_dir, self.task)

        self.processed_name = '_'.join([self.data_source, str(self.skip_window), self.task, self.task_type]) + '.pickle'
        if self.is_date_encoding:
            self.processed_name = '_'.join([self.data_source,str(self.skip_window),'date_encoding'])+'.pickle'
        self.processed_path = join(self.processed_task_dir, self.processed_name)
        # self.dict_path = join(self.dict_dir, self.dict_name)
        # self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        # self.log_name = self.get_params_str(['data_source', 'model',
        #                  'max_epoch', 'train_batch_size',
        #                  'skip_window', 'num_samples',
        #                  'activation', 'is_scale',
        #                  'is_date_encoding', 'reduced_window'])
        self.log_name = self.get_params_str(['data_source', 'model', 'task_type',
                                             'max_epoch', 'train_batch_size', 'predict_type'])

    def get_params_str(self, params):
        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '_' + str(eval('self.args.' + paramsStr))
        return model_params_str

    def mkdir(self, *args):
        dir_path = join(*args)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def get_file_name_from_path(self, path):
        assert isinstance(path, str)
        file_name = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return file_name


cfg = Configs()
