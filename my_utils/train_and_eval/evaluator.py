import numpy as np
from scipy.stats.mstats import gmean
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import copy

# base class
class Evaluator():
    # the prototype of Evaluator
    def __init__(self, model, data_loader, measure):
        self.model = model
        self.data_loader = data_loader
        self.measure = measure
        self.record = []

    def evaluate(self, record=False):
        return

# Evaluator for classifier
class EvaluatorC(Evaluator):
    def __init__(self, model, data_loader, measure='accuracy'):
        super().__init__(model, data_loader, measure)
        self.go_up = True

    # override
    def evaluate(self, record=False):
        ys_pred = []
        ys_true = []
        for inputs, y_true in self.data_loader:
            ys_pred.append(self.model.predict(inputs))
            ys_true.append(y_true)
        ys_pred = np.concatenate(ys_pred)
        ys_true = np.concatenate(ys_true)

        if self.measure == 'accuracy': value = accuracy_score(ys_true, ys_pred)
        elif self.measure == 'F': value = f1_score(ys_true, ys_pred)
        elif self.measure == 'macroF': value = f1_score(ys_true, ys_pred, average='macro')
        elif self.measure == 'microF': value = f1_score(ys_true, ys_pred, average='micro')
        else:
            raise ValueError('measure: [accuray, F, macroF, microF]')

        self.record.append(value)
        return value


class EvaluatorLM(Evaluator):
    def __init__(self, model, data_loader, measure='perplexity'):
        super().__init__(model, data_loader, measure)
        self.go_up = False

    def evaluate(self, record=False):
        prob = np.array([])

        for inputs, labels in self.data_loader:
            predicted = self.model.predict(inputs)

            if type(predicted) is list:
                # if predicted == [[label, label, ...], [...], ...]
                #    labels = [[label, label, ...], [...], ...]
                predicted = np.concatenate(predicted)
                labels = np.concatenate(labels)
            predicted = predicted[[range(len(labels)), labels]]
            if type(predicted) is not np.ndarray:
                predicted = predicted.cpu().numpy()
            prob = np.concatenate((prob, predicted))
        perplexity = gmean(1/prob)
        if record:
            self.record.append(perplexity)
        return perplexity


class EvaluatorTE(Evaluator):
    def __init__(self, model, testdata, measure='meanrank', predict_rel=False):
        """
        testdata: an array of tuples (feature, label)
        """
        super(EvaluatorTE, self).__init__(model, testdata, measure)
        self.predict_rel = predict_rel
        if self.measure == 'meanrank':
            self.go_up = False
            self.best_score = np.inf

    # override
    def evaluate(self, record=False):
        if self.measure == 'accuracy': func = self.hits_at_best
        elif self.measure == 'hits@10': func = self.hits_at_ten
        elif self.measure == 'meanrank': func = self.get_rank
        else:
            raise ValueError('measure: [accuray, hits@10, meanrank]')

        value = np.array([func(triple) for triple in self.testdata]).mean()
        if record:
            self.record.append(value)
        return value

    def get_rank(self, triple):
        idx = int(self.predict_rel)
        target_id = triple[idx]
        ds = self.model.get_collapted_distance(triple, predict_rel=self.predict_rel)
        target_d = ds[target_id]
        rank = (ds < target_d).sum()
        return rank

    def hits_at_best(self, triple):
        rank = self.get_rank(triple)
        return rank == 0

    def hits_at_ten(self, triple):
        rank = self.get_rank(triple)
        return rank < 10
