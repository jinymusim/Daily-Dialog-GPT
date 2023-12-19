from transformers import  AutoTokenizer, AutoModelForCausalLM, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, PreTrainedTokenizer, PreTrainedModel
from diffusers import StableDiffusionPipeline
from utils.dialog_utils import Tokens
from playsound import playsound
import datasets
import torch
import argparse
import os
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser()

parser.add_argument("--max_token_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--tokenizer_hf_model", default="jinymusim/dialogmodel", type=str, help="Default huggingface model path")
parser.add_argument("--lm_model_path", default="jinymusim/dialogmodel", type=str, help="Model path")
parser.add_argument("--voice_preprocess_path", default="microsoft/speecht5_tts", type=str, help="Model path")
parser.add_argument("--voice_model_path", default="microsoft/speecht5_tts", type=str, help="Model path")
parser.add_argument("--voice_vocoder_path", default="microsoft/speecht5_hifigan", type=str, help="Model path")
parser.add_argument("--use_voice", default=True, type=bool, help="If to use voice output")
parser.add_argument("--stable_diff_model", default='runwayml/stable-diffusion-v1-5', type=str, help="Stable diffusion Model")
parser.add_argument("--use_image", default=False, type=bool, help="If to use image output")

class DialogSystem:
    
    def __init__(self, tokenizer: PreTrainedTokenizer, lm_model: PreTrainedModel, voice_preprocess = None, voice_model = None, voice_vocoder = None, 
                 speaker_embed = None, diff_model = None) -> None:
        self.tokenizer = tokenizer
        self.lm_model = lm_model
        self.speaker_embed = speaker_embed
        self.voice_preprocess = voice_preprocess
        self.voice_model = voice_model
        self.voice_vocoder = voice_vocoder
        self.diff_model = diff_model
        self.dialog_state = ''
        
        self.sys_token_count = 0
        
    def interact(self):
        conversation = ""
        while True:
            conversation += f"{Tokens.USER} " 
            user_utterance = input('USER> ')
            user_utterance = user_utterance.strip()
            if user_utterance.lower() in ['stop', 'end', 'break']:
                break
            conversation += user_utterance
            response = self.generate(conversation)
            print(f'SYSTEM> {response}')
            
            conversation += f" {Tokens.SYSTEM} {response} "

        
        
    def generate(self, conversation: str):
        token_size = True
        while token_size:
            tokenized_context: torch.Tensor = self.tokenizer.encode(conversation, return_tensors='pt')
            if tokenized_context.size(-1) > 1024:
                conv_index = min(conversation.index(Tokens.USER) + len(Tokens.USER), conversation.index(Tokens.SYSTEM) + len(Tokens.SYSTEM))
                # Deletion of System Token
                if conversation.index(Tokens.USER) > conversation.index(Tokens.SYSTEM):
                    self.sys_token_count -= 1
                conversation = conversation[conv_index:]
            else:
                token_size = False
                
        out_response = self.lm_model.generate(tokenized_context, 
                                              max_length=min(1024 - tokenized_context.size(-1), 50),
                                              do_sample=True,
                                              top_k=50,
                                              early_stopping=True,
                                              eos_token_id=self.tokenizer.eos_token_id,
                                              pad_token_id=self.tokenizer.pad_token_id)
        
        
        decoded_response = self.tokenizer.decode(out_response[0], skip_special_tokens=False)
        decoded_response = decoded_response.split(Tokens.SYSTEM)[-1].split(self.tokenizer.eos_token)[0].strip() if decoded_response.count(Tokens.SYSTEM) <= self.sys_token_count else decoded_response.split(Tokens.SYSTEM)[self.sys_token_count + 1].split(self.tokenizer.eos_token)[0].strip()
        decoded_response = decoded_response.split(Tokens.USER)[0].strip() if Tokens.USER in decoded_response else decoded_response
        
        if self.voice_model != None:
            input_voc = self.voice_preprocess(text=decoded_response, return_tensors='pt')
            speech = self.voice_model.generate_speech(input_voc["input_ids"],self.speaker_embed, vocoder=self.voice_vocoder)
        
            sf.write("ds_test.wav", speech.numpy(), samplerate=16000)
            playsound("ds_test.wav")
            os.remove("ds_test.wav")
            
        if self.diff_model != None:
            image = self.diff_model(decoded_response).images[0]
            plt.imshow(image)
            plt.show()  
            
            
        self.sys_token_count +=1
        
        return decoded_response
    
def main(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_hf_model)
    tokenizer.model_max_length = args.max_token_len
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path)
    voice_pre, voice_model, voice_vocoder,speaker_embeddings = None, None, None, None
    if args.use_voice:
        voice_pre =  SpeechT5Processor.from_pretrained(args.voice_preprocess_path)
        voice_model =  SpeechT5ForTextToSpeech.from_pretrained(args.voice_model_path)
        voice_vocoder = SpeechT5HifiGan.from_pretrained(args.voice_vocoder_path)
    
        embeddings_dataset = datasets.load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[1000]["xvector"]).unsqueeze(0)
    stable_diff  = None
    if args.use_image: 
        stable_diff = StableDiffusionPipeline.from_pretrained(args.stable_diff_model, torch_dtype=torch.float16)
    
    DS = DialogSystem(tokenizer, lm_model, voice_pre, voice_model, voice_vocoder,speaker_embeddings,stable_diff)
    DS.interact()
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)