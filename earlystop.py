import numpy as np

class EarlyStopping:
    """
    Research-grade early stopping for RTFL: tracks best model params by validation loss/metric.
    Usage: create an instance, call .step(val_metric, model_params) after each epoch/round.
    """
    def __init__(self, mode='min', patience=3, restore_best=True):
        assert mode in ['min', 'max']
        self.mode = mode
        self.patience = patience
        self.restore_best = restore_best
        self.best_metric = np.inf if mode == 'min' else -np.inf
        self.best_params = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_metric, model_params):
        improved = (val_metric < self.best_metric) if self.mode == 'min' else (val_metric > self.best_metric)
        if improved:
            self.best_metric = val_metric
            self.best_params = {k: np.copy(v) for k, v in model_params.items()}
            self.counter = 0
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def get_best(self):
        return self.best_params

    def reset(self):
        self.best_metric = np.inf if self.mode == 'min' else -np.inf
        self.best_params = None
        self.counter = 0
        self.early_stop = False
