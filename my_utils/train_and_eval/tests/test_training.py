from my_utils import Trainer, EvaluatorC, ScoreMonitor, DataLoader

class PseudoModel():
    def forward(self, inputs):
        return 0

    def fit(self, inputs, labels, optimizer):
        return 0

    def predict(self, inputs):
        return inputs

train = [(0, 0) for _ in range(10)]
test = [(0, 0) for _ in range(3)] + [(0 , 1) for _ in range(3)]

def test_training():
    threshold = 3

    train_loader = DataLoader(train, batch_size=2)
    test_loader = DataLoader(test, batch_size=2)

    model = PseudoModel()

    optimizer = None
    monitor = ScoreMonitor(threshold = threshold)
    assert monitor.best_model is None
    trainer = Trainer(model, train_loader)
    evaluator = EvaluatorC(model, test_loader)
    trainer.train(optimizer, max_epoch=10, evaluator=evaluator, score_monitor=monitor)
    assert 0.5 == evaluator.evaluate()
    assert 1 + threshold == trainer.total_epoch
    assert monitor.best_model is not None
