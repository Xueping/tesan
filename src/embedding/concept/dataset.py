import math
import datetime
import numpy as np
from numpy import random
from src.template.dataset import DatasetTemplate
from src.utils.file import save_file,load_file
from src.utils.configs import cfg


class ConceptDataset(DatasetTemplate):
    def __init__(self):
        super(ConceptDataset, self).__init__()
        self.contexts = None
        self.intervals = None
        self.labels = None

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

            # concat all codes togeter for one patient, and the format is [code, date]
            for s_visit in selected_visits:
                dt = datetime.datetime.strptime(s_visit['admsn_dt'], "%Y%m%d")
                codes = s_visit['DXs']
                if not self.dx_only:
                    codes.extend(s_visit['CPTs'])
                for code in codes:
                    selected_codes.append([code, dt])

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

                context_indices = [[word2[0], (word2[1] - word[1]).days] for pos2, word2
                                   in window_pos if (word2[0] is not None and pos2 != pos)]

                context_len = len(context_indices)
                if context_len > 0:
                    # if context lenth is less than two times actual window, and padding
                    if context_len < 2 * actual_window:
                        for i in range(2 * actual_window - context_len):
                            context_indices.append([0, 0])

                    intervals = np.zeros((2 * actual_window, 2 * actual_window))
                    for i in range(2 * actual_window):
                        for j in range(2 * actual_window):
                            if i > j:
                                code_i = context_indices[i][0]
                                code_j = context_indices[j][0]
                                interval_i = context_indices[i][1]
                                interval_j = context_indices[j][1]
                                if code_i > 0 and code_j > 0:
                                    intervals[i, j] = np.abs(interval_i - interval_j) + 1
                                    intervals[j, i] = np.abs(interval_i - interval_j) + 1

                    batches.append([np.array(context_indices, dtype=np.int32),
                                    intervals, np.array([word[0]], dtype=np.int32)])

        contexts = []
        intervals = []
        labels = []
        for batch in batches:
            contexts.append(batch[0])
            intervals.append(batch[1])
            labels.append(batch[2])

        save_file({'contexts': contexts, 'intervals': intervals, 'labels': labels}, cfg.processed_path)
        return contexts, intervals, labels

    def load_data(self):
        is_load, data = load_file(cfg.processed_path, 'processed data', 'pickle')
        if not is_load:
            data = self.process_data()
            self.contexts = data[0]
            self.intervals = data[1]
            self.labels = data[2]
        else:
            self.contexts = data['contexts']
            self.intervals = data['intervals']
            self.labels = data['labels']
        self.train_size = len(self.contexts)

    def generate_batch(self, num_steps=None):

        def data_queue():
            assert self.train_size >= self.batch_size
            data_ptr = 0
            dataRound = 0
            idx_b = 0
            step = 0
            while True:
                if data_ptr + self.batch_size <= self.train_size:
                    context = self.contexts[data_ptr:data_ptr + self.batch_size]
                    time_mask = self.intervals[data_ptr:data_ptr + self.batch_size]
                    labels = self.labels[data_ptr:data_ptr + self.batch_size]
                    yield np.array(context,dtype=np.int32),\
                          np.array(time_mask,dtype=np.int32),\
                          np.array(labels,dtype=np.int32), dataRound, idx_b
                    data_ptr += self.batch_size
                    idx_b += 1
                    step += 1
                elif data_ptr + self.batch_size > self.train_size:
                    offset = data_ptr + self.batch_size - self.train_size
                    context = self.contexts[data_ptr:]
                    context += self.contexts[:offset]
                    time_mask = self.intervals[data_ptr:]
                    time_mask += self.intervals[:offset]
                    labels = self.labels[data_ptr:]
                    labels += self.labels[:offset]

                    data_ptr = offset
                    dataRound += 1

                    yield np.array(context,dtype=np.int32),\
                          np.array(time_mask,dtype=np.int32),\
                          np.array(labels,dtype=np.int32), dataRound, 0
                    idx_b = 1
                    step += 1

                if num_steps is not None and step >= num_steps:
                    break

        batch_num = math.ceil(len(self.patients) / self.batch_size)
        for context,time_mask, labels, data_round, idx_b in data_queue():
            yield context,time_mask,labels, batch_num, data_round, idx_b
