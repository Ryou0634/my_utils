import numpy as np
from nltk import bleu_score
from scipy.stats.mstats import gmean
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import copy

class Evaluator():
    """
    Base class for Evaluators

    Args
    ----
    model :
        A model to be evaluated.
    data_loader : my_utils.DataLoader
        A data-loader which contains test/dev dataset.
    measure : str
        Specify how to evaluate the dataset.
    """
    def __init__(self, model, data_loader, measure):
        self.model = model
        self.data_loader = data_loader
        self.measure = measure

    def evaluate(self):
        return

class EvaluatorLoss(Evaluator):
    def __init__(self, model, data_loader):
        super().__init__(model, data_loader, measure='loss')

    def evaluate(self):
        loss_sum = 0
        for inputs, targets in self.data_loader:
            loss_sum += self.model.fit(inputs, targets, optimizer=None)
        loss_sum = loss_sum/self.data_loader.n_batches
        return loss_sum


class EvaluatorC(Evaluator):
    """Evaluator for classification"""
    def __init__(self, model, data_loader, measure='accuracy'):
        super().__init__(model, data_loader, measure)

    # override
    def evaluate(self):
        ys_pred = []
        ys_true = []
        for inputs, targets in self.data_loader:
            ys_pred.append(self.model.predict(inputs))
            ys_true.append(targets)
        ys_pred = np.concatenate(ys_pred)
        ys_true = np.concatenate(ys_true)

        if self.measure == 'accuracy': value = accuracy_score(ys_true, ys_pred)
        elif self.measure == 'F': value = f1_score(ys_true, ys_pred)
        elif self.measure == 'macroF': value = f1_score(ys_true, ys_pred, average='macro')
        elif self.measure == 'microF': value = f1_score(ys_true, ys_pred, average='micro')
        else:
            raise ValueError('measure: [accuray, F, macroF, microF]')

        return value

# Evaluator for sequence generation
class EvaluatorSeq(Evaluator):
    def __init__(self, model, data_loader, measure='accuracy'):
        super().__init__(model, data_loader, measure)

    def evaluate(self):
        ys_pred = []
        ys_true = []
        if self.measure == 'accuracy':
            for inputs, targets in self.data_loader:
                predicted = self.model.predict(inputs)
                predicted, targets = self._pad(predicted, targets) # when over- or under-generate
                ys_pred.append(np.concatenate(predicted))
                ys_true.append(np.concatenate(targets))
            ys_pred = np.concatenate(ys_pred)
            ys_true = np.concatenate(ys_true)
            value = accuracy_score(ys_true, ys_pred)
        elif self.measure == 'BLEU':
            for inputs, targets in self.data_loader:
                ys_pred += self.model.predict(inputs)
                ys_true += [np.array(t)[np.newaxis, :] for t in targets]
            value = bleu_score.corpus_bleu(ys_true, ys_pred)
        else:
            raise ValueError("measure: ['accuray', 'BLEU']")
        return value

    def _pad(self, predicted, targets):
        for i in range(len(predicted)):
            offset = len(predicted[i])-len(targets[i])
            if offset > 0:
                targets[i] = np.concatenate((targets[i], [-1 for _ in range(offset)]))
            else:
                predicted[i] = np.concatenate((predicted[i], [-1 for _ in range(-offset)]))
        return predicted, targets

class EvaluatorLM(Evaluator):
    def __init__(self, model, data_loader, measure='perplexity'):
        super().__init__(model, data_loader, measure)

    def evaluate(self):
        prob = np.array([])

        for inputs, targets in self.data_loader:
            predicted = self.model.predict(inputs)

            if type(predicted) is list:
                # if predicted == [[label, label, ...], [...], ...]
                #    targets = [[label, label, ...], [...], ...]
                predicted = np.concatenate(predicted)
                targets = np.concatenate(targets)
            predicted = predicted[[range(len(targets)), targets]]
            if type(predicted) is not np.ndarray:
                predicted = predicted.cpu().numpy()
            prob = np.concatenate((prob, predicted))
        perplexity = gmean(1/prob)
        return perplexity


class EvaluatorTE(Evaluator):
    def __init__(self, model, testdata, measure='meanrank', predict_rel=False):
        """
        testdata: an array of tuples (feature, label)
        """
        super(EvaluatorTE, self).__init__(model, testdata, measure)
        self.predict_rel = predict_rel


    # override
    def evaluate(self):
        if self.measure == 'accuracy': func = self.hits_at_best
        elif self.measure == 'hits@10': func = self.hits_at_ten
        elif self.measure == 'meanrank': func = self.get_rank
        else:
            raise ValueError('measure: [accuray, hits@10, meanrank]')

        value = np.array([func(triple) for triple in self.testdata]).mean()
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
