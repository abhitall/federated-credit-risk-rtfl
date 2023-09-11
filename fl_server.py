from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from tcm import TemporalCheckpointManifold
from zkip import ZeroKnowledgeIntegrityProofs
from arrp import AdaptiveRoleReassignmentProtocol
from ebcd import EntropyBasedCorruptionDetection
from dss import DifferentialStateSynchronizer
from sklearn.linear_model import SGDClassifier
from earlystop import EarlyStopping

class FLServer:
    def __init__(self, server_id, client_ids, num_features, X_val=None, y_val=None, earlystop_patience=3):
        self.server_id = server_id
        self.global_model_parameters = {
            'coef_': np.zeros((1, num_features)),
            'intercept_': np.array([0.0])
        }
        self.num_features = num_features
        self.tcm = TemporalCheckpointManifold() 
        self.zkip = ZeroKnowledgeIntegrityProofs() 
        self.arrp = AdaptiveRoleReassignmentProtocol(server_id, client_ids) 
        self.ebcd = EntropyBasedCorruptionDetection() 
        self.dss_aggregator = DifferentialStateSynchronizer() 
        self.dss_aggregator.set_base_model_parameters(self.global_model_parameters)
        self.X_val = X_val
        self.y_val = y_val
        self.earlystop_patience = earlystop_patience
        self.earlystop = EarlyStopping(mode='max', patience=earlystop_patience)

    def aggregate_model_deltas(self, client_deltas_with_proofs, client_data_sizes):
        valid_deltas = []
        weights = [] 
        active_client_ids_in_aggregation = []
        for i, (delta, proof, client_id) in enumerate(client_deltas_with_proofs):
            if delta is None: continue
            if self.zkip.verify_proof(delta, proof):
                valid_deltas.append(delta)
                weights.append(client_data_sizes[i] if client_data_sizes else 1) 
                active_client_ids_in_aggregation.append(client_id)
            else:
                print(f"Server: ZKIP verification FAILED for update from client {client_id}. Discarding.")
        if not valid_deltas:
            return False, active_client_ids_in_aggregation 
        total_weight = sum(weights)
        if total_weight == 0: 
            return False, active_client_ids_in_aggregation
        aggregated_delta = {k: np.zeros_like(v) for k, v in valid_deltas[0].items()}
        for i, delta in enumerate(valid_deltas):
            for key in aggregated_delta:
                if key in delta:
                     aggregated_delta[key] += delta[key] * (weights[i] / total_weight)
        self.dss_aggregator.set_base_model_parameters(self.global_model_parameters) 
        self.global_model_parameters = self.dss_aggregator.apply_delta_to_base(aggregated_delta)
        if self.ebcd.check_for_corruption(self.global_model_parameters):
            print("Server: EBCD detected potential corruption/high noise in global model POST-aggregation.")
        # Early stopping: check validation metric (accuracy)
        if self.X_val is not None and self.y_val is not None and self.X_val.shape[0] > 0:
            eval_model = SGDClassifier(loss='log_loss')
            eval_model.coef_ = np.zeros((1, self.num_features))
            eval_model.intercept_ = np.array([0.0])
            unique_y_val = np.unique(self.y_val)
            if len(unique_y_val) >=2 :
                eval_model.partial_fit(self.X_val[:1], self.y_val[:1], classes=np.array([0,1]))
            elif len(unique_y_val) == 1:
                eval_model.partial_fit(self.X_val[:1], self.y_val[:1], classes=unique_y_val)
            eval_model.coef_ = np.copy(self.global_model_parameters['coef_'])
            eval_model.intercept_ = np.copy(self.global_model_parameters['intercept_'])
            try:
                val_pred = eval_model.predict(self.X_val)
                val_acc = accuracy_score(self.y_val, val_pred)
            except Exception:
                val_acc = 0.0
            if self.earlystop.step(val_acc, self.global_model_parameters):
                print("Server: Early stopping triggered. Restoring best global model parameters.")
                best_params = self.earlystop.get_best()
                if best_params is not None:
                    self.global_model_parameters = {k: np.copy(v) for k, v in best_params.items()}
        return True, active_client_ids_in_aggregation

    def get_global_model_parameters_for_clients(self):
        self.arrp.check_server_status() 
        coordinator_id = self.arrp.get_current_coordinator_id()
        if coordinator_id == self.server_id and self.arrp.is_original_server_active():
            return {k: np.copy(v) for k, v in self.global_model_parameters.items()}
        elif coordinator_id != self.server_id: 
            return {k: np.copy(v) for k, v in self.global_model_parameters.items()}
        else: 
            return None

    def evaluate_global_model(self, X_test, y_test, round_num):
        if X_test.shape[0] == 0: return {}
        eval_model = SGDClassifier(loss='log_loss')
        eval_model.coef_ = np.zeros((1, self.num_features))
        eval_model.intercept_ = np.array([0.0])
        if X_test.shape[0] > 0 :
            unique_y_test = np.unique(y_test)
            if len(unique_y_test) >=2 :
                eval_model.partial_fit(X_test[:1], y_test[:1], classes=np.array([0,1]))
            elif len(unique_y_test) == 1:
                 eval_model.partial_fit(X_test[:1], y_test[:1], classes=unique_y_test)
        eval_model.coef_ = np.copy(self.global_model_parameters['coef_'])
        eval_model.intercept_ = np.copy(self.global_model_parameters['intercept_'])
        try:
            predictions = eval_model.predict(X_test)
            pred_classes = np.unique(predictions)
            y_classes = np.unique(y_test)
            print(f"[Round {round_num}] y_test classes: {y_classes}, predictions classes: {pred_classes}")
            # F1 and AUC only meaningful if >1 class in y_test and predictions
            if len(y_classes) < 2 or len(pred_classes) < 2:
                f1 = np.nan
                auc_roc_val = np.nan
            else:
                f1 = f1_score(y_test, predictions, zero_division=0)
                if hasattr(eval_model, "predict_proba") and hasattr(eval_model, "classes_") and len(eval_model.classes_) >=2 :
                    probas = eval_model.predict_proba(X_test)[:, 1]
                    auc_roc_val = roc_auc_score(y_test, probas)
                else:
                    auc_roc_val = np.nan
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'f1_score': f1,
                'auc_roc': auc_roc_val
            }
            return metrics
        except Exception as e:
            print(f"Error during evaluation (Round {round_num}): {e}")
            return {'accuracy': 0, 'f1_score': np.nan, 'auc_roc': np.nan}
