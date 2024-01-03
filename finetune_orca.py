import argparse
import torch
from accelerate import Accelerator
from functools import partial
from transformers import AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast, IntervalStrategy
from utils.dialog_model import OrcaModel
from utils.dataset import OrcaDataset
from utils.dialog_utils import Tokens
import os

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname("__file__"), "orca_model_e16")), type=str, help="Model path")
parser.add_argument("--tokenizer", default=os.path.abspath(os.path.join(os.path.dirname("__file__"), "dialog_model_e16")), type=str, help="Model path")
parser.add_argument("--dataset", default="Open-Orca/OpenOrca", type=str, help="Dialog Dataset to use for finetune")

def main(args: argparse.Namespace):
    model = OrcaModel()
    
    # Parallel Plugin
    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare(model)  
    

    dataset = OrcaDataset(args.dataset)    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    if tokenizer.pad_token == None:
        tokenizer.pad_token_id = 0
    tokenizer.model_max_length = model.max_model_len
    
    collate = partial(OrcaDataset.collate, tokenizer=tokenizer)
    
    
    training_args = TrainingArguments(
                                  save_strategy  = IntervalStrategy.EPOCH,
                                  save_total_limit=1,
                                  warmup_steps = 0,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = 16,
                                  learning_rate = 3e-4,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="cosine",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = 16)

    trainer = Trainer(model = model,
                  args = training_args,
                  train_dataset= dataset,
                  data_collator=collate).train()

    
    # Save Model
    model.save_LM(args.model_path)
    tokenizer.save_pretrained(args.model_path)
              
        
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
