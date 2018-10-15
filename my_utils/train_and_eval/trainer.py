import copy
from my_utils.misc.logging import logger

class Trainer():
    '''
    Data loader for mini-batch stochastic gradient decent models.
    Shuffle dataset and feed it into a model. Typically used with Trainer.

    Attributes
    ----------
    model :
        A model to train.
    train_loader : my_utils.DataLoader
        DataLoader with training dataset.
    total_epoch : int
        The number of epoches elasped.
    '''
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.total_epoch = 0
        self.total_steps = 0

    def train_epoch(self, optimizer, max_epoch=1,
              evaluator=None, score_monitor=None, model_saver=None):
        """
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
        hook_func : function
            Do whatever you want to during each epoch by this function.
            (e.g. save the model, print some other information, etc.)
        """
        for epoch in range(max_epoch):
            loss_sum = 0
            for inputs, labels in self.train_loader:
                loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
                self.total_steps += 1
            loss_sum /= self.train_loader.n_batch
            self.total_epoch += 1
            logger.info("epoch [{}/{}]\tloss: {}\t".format(self.total_epoch, max_epoch, loss_sum))
            stop_flag = self.evaluate(evaluator, score_monitor, model_saver)
            if stop_flag: break
        return

    def train_step(self, optimizer, checkpoint_steps=5000, max_steps=100000,
                   evaluator=None, score_monitor=None, model_saver=None):
        loss_sum = 0
        stop_flag = False
        while True:
            for inputs, labels in self.train_loader:
                loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
                self.total_steps += 1
                if self.total_steps%checkpoint_steps == 0:
                    loss_sum = loss_sum/checkpoint_steps
                    logger.info("steps [{}/{}]\tloss: {}\t".format(self.total_steps, max_steps, loss_sum))
                    loss_sum = 0
                    stop_flag = self.evaluate(evaluator, score_monitor, model_saver)
                if stop_flag or self.total_steps >= max_steps: return

    def evaluate(self, evaluator, score_monitor, model_saver):
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



# class MultiSwithTrainer(Trainer):
#     def __init__(self, models, train_loaders, n_fit_per_switch):
#         assert len(models) == len(train_loaders)
#         self.n_models = len(models)
#         super().__init__(models, train_loaders)
#         self.n_fit_per_switch = n_fit_per_switch
#
#     def _fit_n_times(self, model, train_loader, optimizer):
#         end_of_epoch = False
#         for _ in range(self.n_fit_per_switch):
#             try:
#                 inputs, labels = next(train_loader)
#                 loss = model.fit(inputs, labels, optimizer)
#             except StopIteration:
#                 iter(train_loader)
#                 end_of_epoch = True
#         return loss, end_of_epoch
#
#     def _train_epoch(self, optimizers):
#         for tl in self.train_loader:
#             iter(tl)
#         losses = [0 for _ in range(self.n_models)]
#         n_batch = 0
#         while True:
#             for i, (ml, tl, optim) in enumerate(zip(self.model, self.train_loader, optimizers)):
#                 loss, end_of_epoch = self._fit_n_times(ml, tl, optim)
#                 losses[i] += loss
#             n_batch += 1
#             if end_of_epoch: break
#         self.total_epoch += 1
#         log = "epoch {:<3}\t".format(self.total_epoch)
#         for i, loss in enumerate(losses):
#             log += 'loss{}: {}\t'.format(i, loss/n_batch)
#         return log
