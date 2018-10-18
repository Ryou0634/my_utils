import copy
from my_utils.misc.logging import logger

class Trainer():
    '''
    Trainer for mini-batch stochastic gradient decent.

    Args
    ----------
    model :
        A model to train.
    train_loader : my_utils.DataLoader
        DataLoader with training dataset.
    '''
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.total_epoch = 0
        self.total_steps = 0

    def train_epoch(self, optimizer, max_epoch=1,
              evaluator=None, score_monitor=None, model_saver=None):
        """
        Train the model for the specidied number of epochs.

        Parameters
        ----------
        optimizer :
            Something to control optimization process. This is used in model.fit().
        max_epoch : int
            The maximum number of epoches.
        evaluator : my_utils.Evaluator
            Evaluator for test or validation dataset.
        score_monitor : my_utils.ScoreMonitor
            Used for early-stopping.
        model_saver :
            An object which contains the model and has save() method.
        """
        for epoch in range(max_epoch):
            loss_sum = 0
            for inputs, labels in self.train_loader:
                loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
                self.total_steps += 1
            loss_sum /= self.train_loader.n_batches
            self.total_epoch += 1
            logger.info("epoch [{}/{}]\tloss: {}\t".format(self.total_epoch, max_epoch, loss_sum))
            stop_flag = self._evaluate(evaluator, score_monitor, model_saver)
            if stop_flag: break
        return

    def train_step(self, optimizer, checkpoint_steps=5000, max_steps=100000,
                   evaluator=None, score_monitor=None, model_saver=None):
        """Train the model for the specidied number of steps instead of epochs."""
        loss_sum = 0
        stop_flag = False
        while True:
            for inputs, labels in self.train_loader:
                loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
                self.total_steps += 1
                if self.total_steps%checkpoint_steps == 0:
                    loss_sum /= checkpoint_steps
                    logger.info("steps [{}/{}]\tloss: {}\t".format(self.total_steps, max_steps, loss_sum))
                    loss_sum = 0
                    stop_flag = self._evaluate(evaluator, score_monitor, model_saver)
                if stop_flag or self.total_steps >= max_steps: return

    def _evaluate(self, evaluator, score_monitor, model_saver):
        if evaluator:
            current_eval = evaluator.evaluate()
            logger.info("Evaluator {}: {}\t".format(evaluator.measure, current_eval))
        if model_saver:
            name_suffix = '_step_{}_{}_{}'.format(self.total_steps, evaluator.measure, current_eval)
            model_saver.save(name_suffix)
        if score_monitor:
            score_monitor.update_best(current_eval, self.model)
            stop_flag = score_monitor.check_stop()
            if stop_flag:
                logger.info('Dev score saturated.')
            return stop_flag
