from abc import ABCMeta, abstractmethod
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, auc

class EvaluationTemplate(metaclass=ABCMeta):

    def __init__(self, model, logging):

        self.model = model
        self.logging = logging
        self.reverse_dict = model.reverse_dict
        self.dictionary = dict(zip(model.reverse_dict.values(), model.reverse_dict.keys()))
        self.verbose = model.verbose
        self.valid_samples = model.valid_samples
        self.top_k = model.top_k

    @abstractmethod
    def get_clustering_nmi(self,sess, ground_truth):
        pass

    @abstractmethod
    def get_nns_p_at_top_k(self,sess, ground_truth):
        pass

    @abstractmethod
    def get_nns_pairs_count(self, ground_truth):
        pass

    @staticmethod
    def metric_pred(y_true, probs, y_pred):
        [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        # print(TN, FP, FN, TP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specificity = TN / (FP + TN)
        precision = TP / (TP + FP)
        sensitivity = recall = TP / (TP + FN)
        f_score = 2 * TP / (2 * TP + FP + FN)

        # calculate AUC
        # roc_auc = roc_auc_score(y_true, probs)
        # print('roc_auc: %.4f' % roc_auc)
        # calculate roc curve
        # fpr, tpr, thresholds = roc_curve(y_true, probs)

        # calculate precision-recall curve
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, probs)

        # calculate F1 score
        f1 = f1_score(y_true, y_pred)
        # calculate precision-recall AUC
        pr_auc = auc(recall_curve, precision_curve)

        return [accuracy, precision, sensitivity, specificity, f_score, pr_auc, f1]

        # return [accuracy, precision, sensitivity, specificity, f_score, roc_auc, pr_auc, f1]


