import math
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split

from src.template.dataset import DatasetTemplate
from src.utils.file import save_file,load_file
from src.utils.configs import cfg
import json


class MortalityDataset(DatasetTemplate):

    def __init__(self):
        super(MortalityDataset, self).__init__()
        self.train_patients = None
        self.train_labels = None
        self.all_patients = None
        self.all_labels = None
        self.test_patients = None
        self.test_labels = None
        self.train_size = 0

    def process_data(self):

        batches = []
        patient_dict = {}
        index = 0
        for patient in self.patients:
            # get patient's visits
            patient_dict['pid_'+patient['pid']] = index
            index += 1
            visits = patient['visits']
            # sorting visits by admission date
            sorted_visits = sorted(visits, key=lambda visit: visit['admsn_dt'])

            # number of visits
            no_visits = len(visits)

            # generating batch sample: list of visits including concept codes,
            # label of last visit mortality
            ls_visits = []
            label = [int(sorted_visits[no_visits-1]['Death'])]
            for visit in sorted_visits:
                codes = visit['DXs']
                if not self.dx_only:
                    codes.extend(visit['CPTs'])

                code_size = len(codes)
                # code padding
                if code_size < self.max_len_visit:
                    list_zeros = [0] * (self.max_len_visit - code_size)
                    codes.extend(list_zeros)
                ls_visits.append(codes)

            # visit padding
            if no_visits < self.max_visits:
                for i in range(self.max_visits - no_visits):
                    list_zeros = [0] * self.max_len_visit
                    ls_visits.append(list_zeros)
            # print(len(ls_visits))
            batches.append([np.array(ls_visits, dtype=np.int32), np.array(label, dtype=np.int32)])

        b_patients = []
        b_label = []
        for batch in batches:
            b_patients.append(batch[0])
            b_label.append(batch[1])

        save_file({'patient': b_patients, 'label': b_label},
                  join(cfg.dataset_dir, 'mortality_'+cfg.data_source+'.pickle'))

        dict_file = join(cfg.dataset_dir, 'mimic3_patient_dict')
        print('patient dict file location: ', dict_file)
        with open(dict_file + '.json', 'w') as fp:
            json.dump(patient_dict, fp)

        return b_patients, b_label

    def load_data(self):

        processed_file = join(cfg.dataset_dir, 'mortality_'+cfg.data_source+'.pickle')
        is_load, data = load_file(processed_file, 'processed data', 'pickle')

        if not is_load:
            data = self.process_data()
            patients = data[0]
            labels = data[1]
            self.all_patients = patients
            self.all_labels = labels
        else:
            patients = data['patient']
            labels = data['label']
            self.all_patients = patients
            self.all_labels = labels

        self.train_patients, self.test_patients, self.train_labels, self.test_labels\
            = train_test_split(patients, labels, test_size=0.1, random_state=42)

        self.train_size = len(self.train_patients)
        #
        # self.dev_patients, self.test_patients, self.dev_labels, self.test_labels \
        #     = train_test_split(vt_patients, vt_labels, test_size=0.5, random_state=42)

    def generate_batch(self, num_steps):

        def data_queue(train_patients, train_labels, batch_size):
            assert len(train_patients) >= batch_size
            data_ptr = 0
            data_round = 0
            idx_b = 0
            step = 0
            while True:
                if data_ptr + batch_size <= len(train_patients):
                    batch_patients = train_patients[data_ptr:data_ptr + batch_size]
                    batch_labels = train_labels[data_ptr:data_ptr + batch_size]

                    yield np.array(batch_patients, dtype=np.int32), \
                          np.array(batch_labels, dtype=np.int32), \
                          data_round, idx_b
                    data_ptr += batch_size
                    idx_b += 1
                    step += 1
                elif data_ptr + batch_size > len(train_patients):
                    offset = data_ptr + batch_size - len(train_patients)

                    batch_patients = train_patients[data_ptr:]
                    batch_patients += train_patients[:offset]
                    batch_labels = train_labels[data_ptr:]
                    batch_labels += train_labels[:offset]

                    data_ptr = offset
                    data_round += 1

                    yield np.array(batch_patients, dtype=np.int32),\
                          np.array(batch_labels,dtype=np.int32), data_round, 0
                    idx_b = 1
                    step += 1
                if step >= num_steps:
                    break

        batch_num = math.ceil(self.train_size / self.batch_size)
        for patients, labels, data_round, idx_b in data_queue(self.train_patients,
                                                              self.train_labels,
                                                              self.batch_size):
            yield patients, labels, batch_num, data_round, idx_b

    def generate_batch_sample_iter(self):

        batch_num = math.ceil(len(self.test_patients) / self.batch_size)

        def data_queue():
            assert len(self.test_patients) >= self.batch_size
            data_ptr = 0
            data_round = 0
            idx_b = 0
            step = 0
            while True:
                if data_ptr + self.batch_size <= len(self.test_patients):
                    batch_patients = self.test_patients[data_ptr:data_ptr + self.batch_size]
                    batch_labels = self.test_labels[data_ptr:data_ptr + self.batch_size]

                    yield np.array(batch_patients, dtype=np.int32), \
                          np.array(batch_labels, dtype=np.int32), \
                          data_round, idx_b
                    data_ptr += self.batch_size
                    idx_b += 1
                    step += 1
                elif data_ptr + self.batch_size > len(self.test_patients):
                    offset = data_ptr + self.batch_size - len(self.test_patients)

                    batch_patients = self.test_patients[data_ptr:]
                    batch_patients += self.test_patients[:offset]
                    batch_labels = self.test_labels[data_ptr:]
                    batch_labels += self.test_labels[:offset]

                    data_ptr = offset
                    data_round += 1

                    yield np.array(batch_patients, dtype=np.int32),\
                          np.array(batch_labels,dtype=np.int32), data_round, 0
                    idx_b = 1
                    step += 1
                if step >= batch_num:
                    break

        for patients, labels, data_round, idx_b in data_queue():
            yield patients, labels, batch_num, data_round, idx_b

    def generate_batch_sample_all(self):

        batch_num = math.ceil(len(self.all_patients) / self.batch_size)

        def data_queue():
            assert len(self.all_patients) >= self.batch_size
            data_ptr = 0
            data_round = 0
            idx_b = 0
            step = 0
            while True:
                if data_ptr + self.batch_size <= len(self.all_patients):
                    batch_patients = self.all_patients[data_ptr:data_ptr + self.batch_size]
                    batch_labels = self.all_labels[data_ptr:data_ptr + self.batch_size]

                    yield np.array(batch_patients, dtype=np.int32), \
                          np.array(batch_labels, dtype=np.int32), \
                          data_round, idx_b, step
                    data_ptr += self.batch_size
                    idx_b += 1
                    step += 1
                elif data_ptr + self.batch_size > len(self.all_patients):
                    offset = data_ptr + self.batch_size - len(self.all_patients)

                    batch_patients = self.all_patients[data_ptr:]
                    # batch_patients += self.all_patients[:offset]
                    batch_labels = self.all_labels[data_ptr:]
                    # batch_labels += self.all_labels[:offset]

                    data_ptr = offset
                    data_round += 1

                    yield np.array(batch_patients, dtype=np.int32),\
                          np.array(batch_labels,dtype=np.int32), data_round, 0, step
                    idx_b = 1
                    step += 1
                if step >= batch_num:
                    break

        for patients, labels, data_round, idx_b, step in data_queue():
            yield patients, labels, step, batch_num, data_round, idx_b