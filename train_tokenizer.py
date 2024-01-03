import os
from utils.dataset import OrcaDataset
from tokenizers import Tokenizer

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as BytePre
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteDec

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer( vocab_size = 16000, min_frequency=2,)
        
tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
tokenizer.decoder = ByteDec()

data = OrcaDataset()
tokenizer.train_from_iterator(data.data, trainer=trainer)
tokenizer.save(os.path.join(os.path.dirname(__file__), 'utils', 'tokenizer', 'tokenizer.json'))