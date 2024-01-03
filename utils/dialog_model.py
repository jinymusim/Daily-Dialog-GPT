from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from .dialog_utils import Tokens
import torch

class OrcaModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = LlamaConfig(vocab_size=16000, hidden_size=1024, intermediate_size=1024, num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=4096)
        self.model = LlamaForCausalLM._from_config(self.config)
        self.max_model_len = 4096
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        return {'loss': outputs.loss, "model_output": outputs}
        
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path, safe_serialization=False)

class DialogModel(torch.nn.Module):
    def __init__(self, pretrained_model,max_len:int = 1024, resize_now=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model,output_hidden_states=True)
        if resize_now:
            self.model.resize_token_embeddings(50257 + len([Tokens.USER, Tokens.SYSTEM]))
        self.max_model_len = max_len
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        return {'loss': outputs.loss, "model_output": outputs}
        
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path, safe_serialization=False)