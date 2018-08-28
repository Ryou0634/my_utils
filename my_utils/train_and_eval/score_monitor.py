import copy
import numpy as np

class ScoreMonitor():
    '''
    Monitor the change of scores(loss, accuracy on testdata, etc...),
    and send a signal to stop traning.

    Attributes
    ----------

    threshold : int
        If score doesn't get better for 'threshold' times,
        the instance will send a stop signal.
    go_up : boolean
        If True, we expect score goes up (e.g. accuracy, F1-score) and
        stop training when the score doesn't go above the best score.
        Set False, if score is loss, perplexity, rank, etc.
    stop_count : int
        Count of time when score didn't improve.
    best_score : int or float
        The best score so far.
    best_model : model
        The best model so far.
    '''

    def __init__(self, threshold=1, go_up=True):

        self.threshold = threshold
        self.go_up = go_up

        self.stop_count = 0
        self.best_score = -np.inf if go_up else np.inf

        self.best_model = None

    def update_best(self, current_score, current_model=None):
        sign = 1 if self.go_up else -1
        if sign*self.best_score < sign*current_score:
            self.best_score = current_score
            self.best_model = copy.deepcopy(current_model)
            self.stop_count = 0
            return True
        else:
            self.stop_count += 1
            return False

    def check_stop(self):
        return self.stop_count >= self.threshold
