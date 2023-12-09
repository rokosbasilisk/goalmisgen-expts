import torch

class BaseAgent(object):
    

    def __init__(self, env, policy, logger, storage, device, num_checkpoints, env_valid=None, storage_valid=None):
        
        self.env = env
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.device = device
        self.num_checkpoints = num_checkpoints
        self.env_valid = env_valid
        self.storage_valid = storage_valid
        self.t = 0

    def predict(self, obs):
        
        pass

    def update_policy(self):
        
        pass

    def train(self, num_timesteps):
        
        pass

    def evaluate(self):
        
        pass
