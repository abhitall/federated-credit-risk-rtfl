import numpy as np

class DifferentialStateSynchronizer:
    def __init__(self):
        self.base_model_parameters = None 

    def set_base_model_parameters(self, model_parameters):
        self.base_model_parameters = {k: np.copy(v) for k, v in model_parameters.items()}

    def compute_delta(self, new_model_parameters):
        if self.base_model_parameters is None:
            raise ValueError("Base model parameters not set for DSS.")
        delta = {}
        for key in new_model_parameters:
            if key in self.base_model_parameters:
                delta[key] = new_model_parameters[key] - self.base_model_parameters[key]
            else: 
                delta[key] = new_model_parameters[key]
        return delta

    def apply_delta_to_base(self, delta):
        if self.base_model_parameters is None:
            raise ValueError("Base model parameters not set for DSS.")
        reconstructed_params = {}
        for key in self.base_model_parameters: 
            reconstructed_params[key] = np.copy(self.base_model_parameters[key])
        for key in delta: 
            if key in reconstructed_params:
                reconstructed_params[key] += delta[key]
            else: 
                reconstructed_params[key] = np.copy(delta[key])
        return reconstructed_params
