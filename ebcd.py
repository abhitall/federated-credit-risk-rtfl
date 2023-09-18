import numpy as np
from scipy.stats import kurtosis, skew
import pickle

class EntropyBasedCorruptionDetection:
    def __init__(self, baseline_stats=None, tolerance_factor=10.0):
        self.baseline_stats = baseline_stats if baseline_stats else {}
        self.tolerance_factor = tolerance_factor 

    def _calculate_param_stats(self, params_array):
        if not isinstance(params_array, np.ndarray) or params_array.ndim == 0:
             return 0,0,0 
        if params_array.size == 0:
            return 0,0,0
        flat_params = params_array.flatten()
        var = np.var(flat_params)
        kurt = kurtosis(flat_params, fisher=True) 
        skw = skew(flat_params)
        return var, kurt, skw

    def save_baseline(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.baseline_stats, f)

    def load_baseline(self, filepath):
        with open(filepath, 'rb') as f:
            self.baseline_stats = pickle.load(f)

    def establish_baseline(self, model_parameters_list, tolerance_factor=None):
        if tolerance_factor is not None:
            self.tolerance_factor = tolerance_factor
        else:
            self.tolerance_factor = 10.0
        all_vars, all_kurts, all_skews = [], [], []
        for params_dict in model_parameters_list:
            for key, params_array in params_dict.items(): 
                if isinstance(params_array, np.ndarray) and params_array.size > 1: 
                    var, kurt, skw = self._calculate_param_stats(params_array)
                    all_vars.append(var)
                    all_kurts.append(kurt)
                    all_skews.append(skw)
        if all_vars: self.baseline_stats['mean_variance'] = np.mean(all_vars)
        if all_vars: self.baseline_stats['std_variance'] = np.std(all_vars)
        if all_kurts: self.baseline_stats['mean_kurtosis'] = np.mean(all_kurts)
        if all_kurts: self.baseline_stats['std_kurtosis'] = np.std(all_kurts)
        if all_skews: self.baseline_stats['mean_skewness'] = np.mean(all_skews)
        if all_skews: self.baseline_stats['std_skewness'] = np.std(all_skews)

    def check_for_corruption(self, model_parameters):
        if not self.baseline_stats or 'mean_variance' not in self.baseline_stats: 
            return False 
        is_corrupted = False
        for key, params_array in model_parameters.items():
            if isinstance(params_array, np.ndarray) and params_array.size > 1:
                var, kurt, skw = self._calculate_param_stats(params_array)
                var_out = False
                kurt_out = False
                if 'mean_variance' in self.baseline_stats and 'std_variance' in self.baseline_stats:
                    mean_v, std_v = self.baseline_stats['mean_variance'], self.baseline_stats['std_variance']
                    if std_v == 0: std_v = 0.001 
                    if abs(var - mean_v) > self.tolerance_factor * std_v:
                        var_out = True
                if 'mean_kurtosis' in self.baseline_stats and 'std_kurtosis' in self.baseline_stats:
                    mean_k, std_k = self.baseline_stats['mean_kurtosis'], self.baseline_stats['std_kurtosis']
                    if std_k == 0: std_k = 0.001
                    if abs(kurt - mean_k) > self.tolerance_factor * std_k:
                        kurt_out = True
                # Less sensitive: require BOTH variance and kurtosis to be out of bounds
                if var_out and kurt_out:
                    is_corrupted = True
        return is_corrupted
