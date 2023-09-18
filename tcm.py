import time
import json
import hashlib
import numpy as np
import pickle

class TemporalCheckpointManifold:
    def __init__(self):
        self.manifold_log = [] 
        self.detailed_states = {} 

    def _hash_data(self, data):
        serialized_data = json.dumps(data, sort_keys=True, default=lambda o: str(o) if isinstance(o, np.ndarray) else '<not_serializable>').encode('utf-8')
        return hashlib.sha256(serialized_data).hexdigest()

    def record_state(self, round_num, model_parameters, server_state_details, client_updates_summary):
        timestamp = time.time()
        serializable_model_params = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model_parameters.items()}
        model_params_hash = self._hash_data(serializable_model_params)
        server_state_hash = self._hash_data(server_state_details)
        client_updates_hash = self._hash_data(client_updates_summary)
        self.manifold_log.append((timestamp, round_num, model_params_hash, server_state_hash, client_updates_hash))
        if model_params_hash not in self.detailed_states:
            self.detailed_states[model_params_hash] = {k: np.copy(v) for k,v in model_parameters.items()}

    def get_latest_state_info(self):
        if not self.manifold_log:
            return None
        return self.manifold_log[-1]

    def recover_state_by_round(self, target_round):
        for entry in reversed(self.manifold_log):
            _, round_num, model_params_hash, _, _ = entry
            if round_num == target_round:
                if model_params_hash in self.detailed_states:
                    return self.detailed_states[model_params_hash], entry
                else:
                    return None, None
        return None, None

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'manifold_log': self.manifold_log, 'detailed_states': self.detailed_states}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.manifold_log = data.get('manifold_log', [])
            self.detailed_states = data.get('detailed_states', {})
