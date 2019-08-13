from src.utils.icd9_ontology import ICD_Ontology as icd
from src.utils.icd9_ontology import CCS_Ontology as ccs
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from src.utils.configs import cfg

from src.template.evaluation import EvaluationTemplate


class ConceptEvaluation(EvaluationTemplate):
    def __init__(self, model, logging):
        self.icd_file = cfg.icd_file
        self.ccs_file = cfg.ccs_file
        super(ConceptEvaluation, self).__init__(model, logging)

    def get_clustering_nmi(self,sess, ground_truth):

        embeddings = sess.run(self.model.final_weights)

        code_list = list(self.dictionary.keys())[1:]
        dx_codes = list()
        tx_codes = list()
        for code in code_list:
            if code.startswith('D_'):
                dx_codes.append(code)
            elif code.startswith('T_'):
                tx_codes.append(code)

        if ground_truth == 'ICD':
            icd9 = icd(self.icd_file, True)
        else:
            icd9 = ccs(self.ccs_file)

        dx_weights = np.empty((len(dx_codes),embeddings.shape[1]),float)
        dx_labels  = np.empty((len(dx_codes)), int)

        dx_index = 0
        for dx in dx_codes:
            dx_labels[dx_index] = icd9.getRootLevel(dx[2:])
            dx_weights[dx_index] = embeddings[self.dictionary[dx],:]
            dx_index += 1

        dx_uni_labels = np.unique(dx_labels).shape[0]
        kmeans = KMeans(n_clusters=dx_uni_labels, random_state=42).fit(dx_weights)

        nmi = metrics.normalized_mutual_info_score(dx_labels,kmeans.labels_)
        nmi_round = round(nmi, 4)

        if self.verbose:
            if ground_truth == 'ICD':
                log_str = "number of dx_labels in ICD9: %s" % dx_uni_labels
                self.logging.add(log_str)
                log_str = "ICD, NMI Score:%s" % nmi_round
                self.logging.add(log_str)
            else:
                log_str = "number of dx_labels in CCS: %s" % dx_uni_labels
                self.logging.add(log_str)
                log_str = "CCS, NMI Score:%s" % nmi_round
                self.logging.add(log_str)

        return nmi_round

    def get_nns_p_at_top_k(self,sess, ground_truth):

        similarity = sess.run(self.model.final_wgt_sim)
        if ground_truth == 'ICD':
            icd9 = icd(self.icd_file, True)
        else:
            icd9 = ccs(self.ccs_file)

        total_precision = 0.
        valid_size = len(self.valid_samples)
        for i in range(valid_size):
            valid_word = self.reverse_dict[self.valid_samples[i]]
            valid_label = icd9.getRootLevel(valid_word[2:])
            nearest = (-similarity[i, :]).argsort()[1:self.top_k+1]
            no_same_cat = 0.
            actual_k = 0
            for k in range(self.top_k):
                close_word = self.reverse_dict[nearest[k]]
                if close_word != "PAD":
                    actual_k += 1
                    close_label = icd9.getRootLevel(close_word[2:])
                    if valid_label == close_label:
                        no_same_cat += 1
            if actual_k > 0:
                total_precision += no_same_cat/actual_k

        evg_precision = round(total_precision/valid_size,4)
        if self.verbose:
            if ground_truth == 'ICD':
                log_str = "ICD NNS P@%s Score:%s" % (self.top_k, evg_precision)
            else:
                log_str = "CCS NNS P@%s Score:%s" % (self.top_k, evg_precision)
            self.logging.add(log_str)

        return evg_precision

    def get_nns_pairs_count(self,ground_truth):

        code_list = list(self.reverse_dict.values())[1:]
        dx_codes = list()
        tx_codes = list()
        for code in code_list:
            if code.startswith('D_'):
                dx_codes.append(code)
            else:
                tx_codes.append(code)

        if ground_truth == 'ICD':
            icd9 = icd(self.icd_file, True)
        else:
            icd9 = ccs(self.ccs_file)

        label_cnt_dict = {}
        for dx in dx_codes:
            label = icd9.getRootLevel(dx[2:])
            if label in label_cnt_dict:
                label_cnt_dict[label] += 1
            else:
                label_cnt_dict[label] = 1

        no_cat = len(label_cnt_dict)
        total_pairs = 0
        for k, v in label_cnt_dict.items():
            total_pairs += v * (v-1)/2

        return no_cat, total_pairs
