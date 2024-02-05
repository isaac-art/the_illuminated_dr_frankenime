import json
import torch
import torch.nn as nn
from typing import Optional
import safetensors.torch as sf

class BaseTrainer():
    def __init__(self, 
            device: str = "cpu", 
            job_id: Optional[str] = None, 
            model: Optional[nn.Module] = None, 
            model_format: str = "pth") -> None:
        super(BaseTrainer, self).__init__()
        self.device = device
        self.job_id = job_id
        self.model = model
        self.model_format = model_format

    def prepare_data(self,  *args, **kwargs):
        return ""
    
    def train_model(self,  *args, **kwargs):
        return ""
    
    def train_step(self,  *args, **kwargs):
        return ""
    
    def val_step(self,  *args, **kwargs):
        return ""
    
    def save_config(self,  *args, **kwargs) -> str:
        f_name = f'archive/{self.job_id}.json'
        with open(f_name, 'w') as f:
            json.dump(kwargs, f)
        return f_name

    def save_model(self) -> str:
        if self.model_format == "safe_tensor":
            path = f'archive/{self.job_id}.bin'
            model_state_dict = self.model.state_dict()
            safe_tensor_dict = {k: sf.serialize(v) for k, v in model_state_dict.items()}
            sf.save(safe_tensor_dict, path)
        else:
            path = f'archive/{self.job_id}.pth'
            torch.save(self.model.state_dict(), path)
        return path
        
    def load_model(self, path=None) -> str:
        if self.model_format == "safe_tensor":
            if path is None:
                path = f'archive/{self.job_id}.bin'
            safe_tensor_dict = sf.load(path)
            self.model.load_state_dict(sf.deserialize(safe_tensor_dict))
        else:
            if path is None:
                path = f'archive/{self.job_id}.pth'
            self.model.load_state_dict(path)
        return path
