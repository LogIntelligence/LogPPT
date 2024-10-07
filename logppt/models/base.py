from typing import Any, Iterator, Tuple
from transformers.utils.dummy_pt_objects import PreTrainedModel
from abc import ABC, abstractmethod
from torch import nn
from torch.nn.parameter import Parameter

class ModelBase(nn.Module):
    def __init__(self, 
                 model_path: str,
                 num_label_tokens: int = 1):
        super().__init__()
        self.model_path = model_path
        self.num_label_tokens = num_label_tokens

    @abstractmethod
    def add_label_token(self, num_tokens:int):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def save_pretrained(self, output_dir):
        self.plm.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
