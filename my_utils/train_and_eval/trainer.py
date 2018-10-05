import copy

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

    def _train_epoch(self, optimizer):
        loss_sum = 0
        for inputs, labels in self.train_loader:
            loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
        loss_sum = loss_sum/self.train_loader.n_batch
        log = "epoch {:<3}\tloss: {}\t".format(self.total_epoch, loss_sum)
        self.total_epoch += 1
        return log

    def train(self, optimizer, max_epoch=1,
              evaluator=None, score_monitor=None, show_log=False, hook_func=None):
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
        show_log : bool
            If True, print log during training.
        hook_func : function
            Do whatever you want to during each epoch by this function.
            (e.g. save the model, print some other information, etc.)
        """

        for epoch in range(max_epoch):
            log = self._train_epoch(optimizer)
            self.misc(evaluator, score_monitor, hook_func, show_log, log)
        return

    def train_iter(self, optimizer, n_iter=5000,
                   evaluator=None, score_monitor=None, show_log=False, hook_func=None):
        loss_sum = 0
        count = 0
        while True:
            for inputs, labels in self.train_loader:
                loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
                count += 1
                if count%n_iter == 0:
                    loss_sum = loss_sum/n_iter
                    log = "iter {:<3}\tloss: {}\t".format(count, loss_sum)
                    loss_sum = 0
                    self.misc(evaluator, score_monitor, hook_func, show_log, log)
        return

    def misc(self, evaluator, score_monitor, hook_func, show_log, log):
        if evaluator:
            current_eval = evaluator.evaluate(record=True)
            log += "{}: {:.4}\t".format(evaluator.measure, current_eval)
        if show_log:
            print(log)
        if score_monitor:
            score_monitor.update_best(current_eval, self.model)
        if hook_func:
            hook_func()



class MultiSwithTrainer(Trainer):
    def __init__(self, models, train_loaders, n_fit_per_switch):
        assert len(models) == len(train_loaders)
        self.n_models = len(models)
        super().__init__(models, train_loaders)
        self.n_fit_per_switch = n_fit_per_switch

    def _fit_n_times(self, model, train_loader, optimizer):
        end_of_epoch = False
        for _ in range(self.n_fit_per_switch):
            try:
                inputs, labels = next(train_loader)
                loss = model.fit(inputs, labels, optimizer)
            except StopIteration:
                iter(train_loader)
                end_of_epoch = True
        return loss, end_of_epoch

    def _train_epoch(self, optimizers):
        for tl in self.train_loader:
            iter(tl)
        losses = [0 for _ in range(self.n_models)]
        n_batch = 0
        while True:
            for i, (ml, tl, optim) in enumerate(zip(self.model, self.train_loader, optimizers)):
                loss, end_of_epoch = self._fit_n_times(ml, tl, optim)
                losses[i] += loss
            n_batch += 1
            if end_of_epoch: break
        self.total_epoch += 1
        log = "epoch {:<3}\t".format(self.total_epoch)
        for i, loss in enumerate(losses):
            log += 'loss{}: {:.4}\t'.format(i, loss/n_batch)
        return log
