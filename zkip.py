import hashlib
import json
import numpy as np
import os

class ZeroKnowledgeIntegrityProofs: 
    def __init__(self, shared_secret=None): 
        if shared_secret is None:
            shared_secret = os.environ.get('RTFL_ZKIP_SECRET', 'default_rtfl_secret')
        self.shared_secret = shared_secret.encode('utf-8')

    def _serialize_data(self, data):
        if isinstance(data, dict):
            serializable_data = {}
            for k, v in sorted(data.items()): 
                if isinstance(v, np.ndarray):
                    serializable_data[k] = v.tolist() 
                else:
                    serializable_data[k] = v
            return json.dumps(serializable_data, sort_keys=True)
        return json.dumps(data, sort_keys=True)

    def generate_proof(self, data_to_prove):
        serialized_data = self._serialize_data(data_to_prove).encode('utf-8')
        proof = hashlib.sha256(self.shared_secret + serialized_data).hexdigest()
        return proof

    def verify_proof(self, data_to_verify, proof):
        expected_proof = self.generate_proof(data_to_verify)
        return expected_proof == proof
