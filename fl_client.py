import random
from sklearn.linear_model import SGDClassifier
import numpy as np
from dss import DifferentialStateSynchronizer
from zkip import ZeroKnowledgeIntegrityProofs
from ebcd import EntropyBasedCorruptionDetection
from earlystop import EarlyStopping

class FLClient:
    def __init__(self, client_id, X_train, y_train, num_features, learning_rate=0.01,
                 dp_epsilon=1.0, dp_delta=1e-5, dp_l2_norm_clip=1.0, random_state=None,
                 X_val=None, y_val=None, earlystop_patience=3):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=learning_rate, random_state=random_state, warm_start=True)
        self.num_features = num_features
        if self.X_train.shape[0] > 0 and self.X_train.shape[1] > 0 :
            self.model.coef_ = np.zeros((1, num_features))
            self.model.intercept_ = np.array([0.0])
            if len(np.unique(self.y_train)) >= 2:
                 self.model.partial_fit(self.X_train[:1], self.y_train[:1], classes=np.array([0, 1]))
            elif len(self.y_train) > 0 : 
                 self.model.partial_fit(self.X_train[:1], self.y_train[:1], classes=np.unique(self.y_train))
        self.dss = DifferentialStateSynchronizer() 
        self.zkip = ZeroKnowledgeIntegrityProofs() 
        self.ebcd = EntropyBasedCorruptionDetection() 
        self.is_faulty = False
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_l2_norm_clip = dp_l2_norm_clip
        self.X_val = X_val
        self.y_val = y_val
        self.earlystop_patience = earlystop_patience

    def set_global_model_parameters(self, global_params):
        current_params = {}
        if global_params and 'coef_' in global_params and 'intercept_' in global_params:
            current_params = {
                'coef_': np.copy(global_params['coef_']).reshape(1, self.num_features),
                'intercept_': np.copy(global_params['intercept_'])
            }
        else:
            current_params = {
                'coef_': np.zeros((1, self.num_features)),
                'intercept_': np.array([0.0])
            }
        self.model.coef_ = current_params['coef_']
        self.model.intercept_ = current_params['intercept_']
        self.dss.set_base_model_parameters(current_params)

    def _apply_differential_privacy(self, delta_params):
        noisy_delta_params = {}
        total_norm = 0.0
        for key in delta_params:
            total_norm += np.linalg.norm(delta_params[key].flatten())**2
        total_norm = np.sqrt(total_norm)
        clip_factor = min(1.0, self.dp_l2_norm_clip / (total_norm + 1e-6))
        noise_stddev = (self.dp_l2_norm_clip * np.sqrt(2 * np.log(1.25 / self.dp_delta))) / self.dp_epsilon
        if self.dp_epsilon == 0:
             noise_stddev = 0
        for key in delta_params:
            clipped_delta = delta_params[key] * clip_factor
            noise = np.random.normal(0, noise_stddev, size=delta_params[key].shape)
            noisy_delta_params[key] = clipped_delta + noise
        return noisy_delta_params

    def train(self, epochs):
        if self.is_faulty or self.X_train.shape[0] == 0 or len(np.unique(self.y_train)) < 2:
            return None, None 
        if not hasattr(self.model, 'classes_') and len(np.unique(self.y_train)) >= 2:
            self.model.partial_fit(self.X_train[:1], self.y_train[:1], classes=np.array([0,1]))
        elif not hasattr(self.model, 'classes_') and len(self.y_train) > 0:
            self.model.partial_fit(self.X_train[:1], self.y_train[:1], classes=np.unique(self.y_train))
        earlystop = EarlyStopping(mode='min', patience=self.earlystop_patience)
        for epoch in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train)
            # Early stopping: check validation loss if validation data is provided
            if self.X_val is not None and self.y_val is not None and len(np.unique(self.y_val)) >= 2:
                try:
                    val_pred = self.model.predict_proba(self.X_val)[:, 1]
                    val_loss = -np.mean(self.y_val * np.log(val_pred + 1e-8) + (1 - self.y_val) * np.log(1 - val_pred + 1e-8))
                except Exception:
                    val_loss = float('inf')
                if earlystop.step(val_loss, {'coef_': np.copy(self.model.coef_), 'intercept_': np.copy(self.model.intercept_)}):
                    break
        # Restore best params if early stopped
        best_params = earlystop.get_best()
        if best_params is not None:
            self.model.coef_ = best_params['coef_']
            self.model.intercept_ = best_params['intercept_']
        current_params = {'coef_': self.model.coef_, 'intercept_': self.model.intercept_}
        try:
            delta_params = self.dss.compute_delta(current_params)
        except ValueError as e:
            self.dss.set_base_model_parameters({'coef_': np.zeros_like(current_params['coef_']), 
                                                'intercept_': np.zeros_like(current_params['intercept_'])})
            delta_params = self.dss.compute_delta(current_params)
        if self.dp_epsilon > 0:
            noisy_delta_params = self._apply_differential_privacy(delta_params)
        else:
            noisy_delta_params = delta_params
        proof = self.zkip.generate_proof(noisy_delta_params)
        return noisy_delta_params, proof

    def simulate_failure(self, probability=0.1):
        if random.random() < probability:
            self.is_faulty = True
        else:
            self.is_faulty = False
        return self.is_faulty

    def model_parameters(self):
        return {'coef_': self.model.coef_, 'intercept_': self.model.intercept_}
