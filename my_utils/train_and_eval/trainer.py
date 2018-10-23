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
        self.total_steps = 0


    def train_epoch(self, optimizer, max_epoch=1,
              evaluator=None, score_monitor=None, model_saver=None):
        checkpoint_steps = self.train_loader.n_batches
        for _ in range(max_epoch):
            stop_flag = self.train_step(optimizer, checkpoint_steps, checkpoint_steps,
                                       evaluator, score_monitor, model_saver)
            if stop_flag: return

    def train_step(self, optimizer, checkpoint_steps=5000, max_steps=100000,
                   evaluator=None, score_monitor=None, model_saver=None):
        """Train the model for the specified number of steps instead of epochs."""
        loss_sum = 0
        stop_flag = False
        self.train_loader.train = True

        for i, (inputs, labels) in enumerate(self.train_loader, 1):
            loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
            self.total_steps += 1
            if i%checkpoint_steps == 0:
                loss_sum /= checkpoint_steps
                logger.info("steps [{}/{}]\tloss: {}\t".format(i, max_steps, loss_sum))
                loss_sum = 0
                stop_flag = self._evaluate(evaluator, score_monitor, model_saver)
            if stop_flag or i >= max_steps: return stop_flag

    def _evaluate(self, evaluator, score_monitor, model_saver):
        if evaluator:
            current_eval = evaluator.evaluate()
            logger.info("Evaluator {}: {}\t".format(evaluator.measure, current_eval))
        if model_saver:
            name_suffix = '_step_{}'.format(self.total_steps)
            if evaluator:
                name_suffix += '_{}_{}'.format(evaluator.measure, current_eval)
            model_saver.save(name_suffix)
        if score_monitor:
            score_monitor.update_best(current_eval, self.model)
            stop_flag = score_monitor.check_stop()
            if stop_flag:
                logger.info('Dev score saturated.')
            return stop_flag
        return False
