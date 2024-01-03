import os
from transformers import PreTrainedTokenizer

import json
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from .dialog_utils import Tokens

class OrcaDataset(Dataset):
    def __init__(self, dataset_name: str = "Open-Orca/OpenOrca", cache_dir = "./") -> None:
        super().__init__()
        self.data = []
        if os.path.isfile(os.path.join(cache_dir, "orca_train_data.json")):
            self.data = json.load(open(os.path.join(cache_dir, "orca_train_data.json"), 'r'))
        else:
            dataset = load_dataset(dataset_name)
            self.__create_data(dataset)
            json.dump(self.data, open(os.path.join(cache_dir, "orca_train_data.json"), 'w+'), indent=6)
            
            
    def __create_data(self, raw_dataset):
        for datum in raw_dataset['train']:
                    
                self.data.append({
                        'input_ids' : f"{Tokens.PERSONALITY} {datum['system_prompt']} {Tokens.USER} {datum['question']} {Tokens.SYSTEM} {datum['response']}"
                    })
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]        
    
    
    @staticmethod
    def collate(batch,  tokenizer):
        tokenized = tokenizer([datum['input_ids'] for datum in batch], return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        return {
            "input_ids" : input_ids,
            "labels": input_ids.type(torch.LongTensor),
            "attention_mask" : attention
        }
        

class DialogDataset(Dataset):
    
    def __init__(self, dataset_name: str = "daily_dialog", cache_dir = "./") -> None:
        super().__init__()
        self.data = []
        if os.path.isfile(os.path.join(cache_dir, "train_data.json")):
            self.data = json.load(open(os.path.join(cache_dir, "train_data.json"), 'r'))
        else:
            dataset = load_dataset(dataset_name)
            self.__create_data(dataset)
            json.dump(self.data, open(os.path.join(cache_dir, "train_data.json"), 'w+'), indent=6)
            
    def __create_data(self, raw_dataset):
        for datum in raw_dataset['train']:
            # Only 1 on 1 conversation
            if  len(set(datum['act'])) != 2:
                continue
            USER_INDEX = datum['act'][0]
            curr_index = USER_INDEX
            players = set(datum['act'])
            players.discard(USER_INDEX)
            SYSTEM_INDEX = list(players)[0]
            
            Token_hash = {USER_INDEX : Tokens.USER, SYSTEM_INDEX: Tokens.SYSTEM }
            
            input_ids = f"{Token_hash.get(curr_index)} "
            i = 0
            
            index_swapped = False
            while i < len(datum['act']):
                if datum['act'][i] == curr_index:
                    pass
                else:
                    curr_index = USER_INDEX if curr_index != USER_INDEX else SYSTEM_INDEX
                    index_swapped = True
                    input_ids += f"{Token_hash.get(curr_index)} "
                input_ids += f"{datum['dialog'][i]} "
                
                if index_swapped:
                    
                    self.data.append({
                        'input_ids' : input_ids.strip()
                    })
                i+=1
                    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]        
    
    
    @staticmethod
    def collate(batch,  tokenizer):
        tokenized = tokenizer([datum['input_ids'] + tokenizer.eos_token for datum in batch], return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        return {
            "input_ids" : input_ids,
            "labels": input_ids.type(torch.LongTensor),
            "attention_mask" : attention
        }
        