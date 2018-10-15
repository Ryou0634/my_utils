import random
import numpy as np
from .trainer import Trainer

class CrossValidator():

    def __init__(self, dataset, train_loader, test_loader, fold=5):
        self.dataset = dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.fold = fold

    def evaluate(self, model, evaluator, optimizer=None, show_log=False):
        random.shuffle(self.dataset)
        splitted = self._get_splitted()
        scores = [0 for _ in range(self.fold)]
        for epoch in range(self.fold):
            if show_log:
                print('========== fold {} =========='.format(epoch))
            # split to train and test
            train = []
            test = []
            for i in range(self.fold):
                if i == epoch:
                    test += splitted[i]
                else:
                    train += splitted[i]
            self.train_loader.dataset = train
            self.test_loader.dataset = test
            # training
            model.initialize()
            evaluator.initialize(model, self.test_loader)
            best_score = self._train_and_evaluate(model, evaluator, optimizer, show_log)
            scores[epoch] = best_score
        mean = np.mean(scores)
        std = np.std(scores)
        if show_log:
            print('{:.2f}±{:.2f}'.format(mean, std))
        return mean, std

    def _train_and_evaluate(self, model, evaluator, optimizer, show_log):
        return

    def _get_splitted(self):
        data_size = len(self.dataset)
        a = data_size // self.fold
        b = data_size % self.fold
        return [self.dataset[i*a + (i if i < b else b):(i+1)*a + (i+1 if i < b else b)] for i in range(self.fold)]

class CV_Neural(CrossValidator):
    def __init__(self, dataset, train_loader, test_loader, fold=5):
        super().__init__(dataset, train_loader, test_loader, fold)

    def _train_and_evaluate(self, model, evaluator, optimizer, show_log):
        trainer = Trainer(model)
        trainer.train_epoch(self.train_loader, optimizer, max_epoch=100, evaluator=evaluator, show_log=show_log)
        return evaluator.best_score

class CV_Simple(CrossValidator):
    def __init__(self, dataset, train_loader, test_loader, fold=5):
        super().__init__(dataset, train_loader, test_loader, fold)

    def _train_and_evaluate(self, model, evaluator, optimizer, show_log):
        model.fit(self.train_loader.dataset)
        return evaluator.evaluate()
