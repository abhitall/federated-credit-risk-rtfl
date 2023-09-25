import random
from enum import Enum

class ServerStatus(Enum):
    NORMAL = 1
    SERVER_DOWN = 2
    COORDINATOR_ELECTED = 3

class AdaptiveRoleReassignmentProtocol:
    def __init__(self, server_id, client_ids, random_seed=None):
        self.server_id = server_id
        self.client_ids = list(client_ids) 
        self.active_clients = list(client_ids)
        self.status = ServerStatus.NORMAL
        self.current_coordinator_id = server_id
        self.failure_detection_probability = 0.05 
        self.recovery_probability = 0.3 
        if random_seed is not None:
            random.seed(random_seed)

    def update_active_clients(self, active_client_ids):
        self.active_clients = list(active_client_ids)

    def check_server_status(self):
        if self.status == ServerStatus.NORMAL:
            if random.random() < self.failure_detection_probability:
                self.status = ServerStatus.SERVER_DOWN
                self._elect_new_coordinator()
        elif self.status == ServerStatus.COORDINATOR_ELECTED or self.status == ServerStatus.SERVER_DOWN:
            if self.current_coordinator_id == self.server_id: 
                if random.random() < self.recovery_probability:
                    self.status = ServerStatus.NORMAL
                    self.current_coordinator_id = self.server_id
            elif self.current_coordinator_id in self.active_clients: 
                if random.random() < self.recovery_probability:
                    self.status = ServerStatus.NORMAL
                    self.current_coordinator_id = self.server_id
            else: 
                self.status = ServerStatus.SERVER_DOWN 
                self._elect_new_coordinator()

    def _elect_new_coordinator(self):
        if not self.active_clients:
            self.current_coordinator_id = self.server_id 
            self.status = ServerStatus.SERVER_DOWN 
            return
        self.active_clients.sort() 
        new_coordinator_id = self.active_clients[0]
        self.current_coordinator_id = new_coordinator_id
        self.status = ServerStatus.COORDINATOR_ELECTED

    def get_current_coordinator_id(self):
        return self.current_coordinator_id

    def is_original_server_active(self):
        return self.status == ServerStatus.NORMAL and self.current_coordinator_id == self.server_id
