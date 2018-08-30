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
    loss_record : List
        A list of loss values for each epoch.
    '''
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.total_epoch = 0
        self.loss_record = []

    def train(self, optimizer, max_epoch=1,
              evaluator=None, score_monitor=None, loss_monitor=None,
              show_log=False, hook_func=None):

        """
        Parameters
        ----------
        optimizer :
            Something to control optimization. This is used in model.fit().
        max_epoch : int
            The maximum number of epoches.
        evaluator : my_utils.Evaluator
            Evaluator for test or validation dataset.
        score_monitor : my_utils.ScoreMonitor
            Used for early-stopping.
        loss_monitor : my_utils.ScoreMonitor
            Used for early-stopping.
        show_log : bool
            If True, print log during training.
        hook_func : function
            Do whatever you want to during each epoch by this function.
            (e.g. save the model, print some other information, etc.)

        Returns
        --------
        best_model : a copy of model
            The model with the highst dev score.
        """

        best_model = None
        for epoch in range(max_epoch):
            loss_sum = 0
            for inputs, labels in self.train_loader:
                loss_sum += self.model.fit(inputs, labels, optimizer=optimizer)
            loss_sum = loss_sum/self.train_loader.n_batch
            log = "epoch {:<3}\tloss: {}\t".format(self.total_epoch, loss_sum)
            self.loss_record.append(loss_sum)
            self.total_epoch += 1

            if evaluator:
                current_eval = evaluator.evaluate(record=True)
                log += "{}: {:.4}\t".format(evaluator.measure, current_eval)
            if show_log:
                print(log)
            if hook_func:
                hook_func()
            if score_monitor:
                score_monitor.update_best(current_eval, self.model)
                if score_monitor.check_stop(): break
            if loss_monitor:
                loss_monitor.update_best(loss_sum, self.model)
                if loss_monitor.check_stop(): break
        return
