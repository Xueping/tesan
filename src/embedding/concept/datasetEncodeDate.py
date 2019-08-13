import collections
import math
import datetime
import numpy as np
from numpy import random
import json
from src.embedding.concept.dataset import ConceptDataset
from src.utils.file import save_file,load_file
from src.utils.configs import cfg


class ConceptAndDateDataset(ConceptDataset):
    def __init__(self):
        super(ConceptAndDateDataset, self).__init__()
        self.date_dict = dict()
        self.date_reverse_dict = dict()

    def build_dict4date(self):
        all_dates = []# store all datetime

        for patient in self.patients:
            for visit in patient['visits']:
                time = visit['admsn_dt']
                all_dates.append(time)

        # store all times and corresponding counts
        count_dates = []
        count_dates.extend(collections.Counter(all_dates).most_common())

        # add padding for time
        self.date_dict['PAD'] = 0
        for date, cnt in count_dates:
            index = len(self.date_dict)
            self.date_dict[date] = index

        # encoding patient using index
        for patient in self.patients:
            visits = patient['visits']
            for visit in visits:
                visit['date_index'] = self.date_dict[visit['admsn_dt']]

        self.date_reverse_dict = dict(zip(self.date_dict.values(), self.date_dict.keys()))
        self.days_size = len(self.date_dict)

        with open(self.dict_file + '_date.json', 'w') as fp:
            json.dump(self.date_reverse_dict, fp)

    def process_data(self):

        batches = []
        for patient in self.patients:
            visits = patient['visits']

            selected_visits = sorted(visits, key=lambda visit: visit['admsn_dt'])
            selected_codes = []

            # reduced window
            if self.is_reduced_window:
                reduced_window = random.RandomState(1).randint(self.skip_window)
            else:
                reduced_window = 0
            # actual window
            actual_window = self.skip_window - reduced_window

            # concat all codes together for one patient, and the format is [code, date, date_index]
            for s_visit in selected_visits:
                dt_index = s_visit['date_index']
                dt = datetime.datetime.strptime(s_visit['admsn_dt'], "%Y%m%d")
                codes = s_visit['DXs']
                if not self.dx_only:
                    codes.extend(s_visit['CPTs'])
                for code in codes:
                    selected_codes.append([code, dt, dt_index])

            # sampling codes based on their frequncy in dataset
            if self.is_sample:
                sampled_codes = [code for code in selected_codes if self.word_sample[code[0]] > random.rand() * 2 ** 32]
            else:
                sampled_codes = selected_codes

            # generate batch samples
            for pos, word in enumerate(sampled_codes):

                # now go over all words from the actual window, predicting each one in turn
                start = max(0, pos - actual_window)
                window_pos = enumerate(sampled_codes[start:(pos + actual_window + 1)], start)

                context_indices = [[word2[0], (word2[1] - word[1]).days, word2[2]] for pos2, word2
                                   in window_pos if (word2[0] is not None and pos2 != pos)]

                context_len = len(context_indices)
                if context_len > 0:
                    # if context length is less than two times of actual window, and padding
                    if context_len < 2 * actual_window:
                        for i in range(2 * actual_window - context_len):
                            context_indices.append([0, 0, 0])
                    batches.append([np.array(context_indices, dtype=np.int32), np.array([word[0]], dtype=np.int32)])

        contexts = []
        labels = []
        for batch in batches:
            contexts.append(batch[0])
            labels.append(batch[1])

        save_file({'contexts': contexts,'labels': labels},cfg.processed_path)
        return contexts, labels

    def load_data(self):
        is_load, data = load_file(cfg.processed_path, 'processed data', 'pickle')
        if not is_load:
            data = self.process_data()
            self.contexts = data[0]
            self.labels = data[1]
        else:
            self.contexts = data['contexts']
            self.labels = data['labels']
        self.train_size = len(self.contexts)

    def generate_batch(self,num_steps=None):

        def data_queue():
            assert self.train_size >= self.batch_size
            data_ptr = 0
            dataRound = 0
            idx_b = 0
            step = 0
            while True:
                if data_ptr + self.batch_size <= self.train_size:
                    context = self.contexts[data_ptr:data_ptr + self.batch_size]
                    labels = self.labels[data_ptr:data_ptr + self.batch_size]
                    yield np.array(context,dtype=np.int32),np.array(labels,dtype=np.int32), dataRound, idx_b
                    data_ptr += self.batch_size
                    idx_b += 1
                    step += 1
                elif data_ptr + self.batch_size > self.train_size:
                    offset = data_ptr + self.batch_size - self.train_size
                    context = self.contexts[data_ptr:]
                    context += self.contexts[:offset]
                    labels = self.labels[data_ptr:]
                    labels += self.labels[:offset]

                    data_ptr = offset
                    dataRound += 1
                    yield np.array(context,dtype=np.int32),np.array(labels,dtype=np.int32), dataRound, 0
                    idx_b = 1
                    step += 1
                if num_steps is not None and  step >= num_steps:
                    break

        batch_num = math.ceil(self.train_size / self.batch_size)
        for context,labels, data_round, idx_b in data_queue():
            yield context, labels, batch_num, data_round, idx_b