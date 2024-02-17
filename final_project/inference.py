import sys
import json
import pandas as pd
sys.path.append("mixtral-offloading")
import torch
from torch.nn import functional as F
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from IPython.display import clear_output
from tqdm.auto import trange
from transformers import AutoConfig, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.utils import logging as hf_logging
import torch
from typing import List
import os
from tqdm import tqdm, trange
from src.build_model import OffloadConfig, QuantConfig, build_model

hf_logging.disable_progress_bar()
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

config = AutoConfig.from_pretrained(quantized_model_name)
state_path = snapshot_download(quantized_model_name)

device = torch.device("cuda:0")

##### Change this to 5 if you have only 12 GB of GPU VRAM #####
offload_per_layer = 4
# offload_per_layer = 5
###############################################################

num_experts = config.num_local_experts

offload_config = OffloadConfig(
    main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
    offload_size=config.num_hidden_layers * offload_per_layer,
    buffer_size=4,
    offload_per_layer=offload_per_layer,
)


attn_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=True,
)
attn_config["scale_quant_params"]["group_size"] = 256


ffn_config = BaseQuantizeConfig(
    nbits=2,
    group_size=16,
    quant_zero=True,
    quant_scale=True,
)
quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


model = build_model(
    device=device,
    quant_config=quant_config,
    offload_config=offload_config,
    state_path=state_path,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.add_eos_token = True
tokenizer.add_bos_token=True
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_bos_token, tokenizer.add_eos_token

def formatting_text(data):
    # prompt = '''Below is an English context with  user's question, situational background information, and my response. Your task is to judge whether my response to the user's question is appropriate or not only according to the situational background information provided. The user's question, situational background information and my response are specified with [User's question], [Background] and [My response] in the context respectively. If you think my response is appropriate, please answer "Yes"; if you think my response is inappropriate, please answer "No".'''
    # prompt = '''In this task, you are asked to provide your answer on "whether my response to the user's question is appropriate based on the 12 situational background statements", and you need to hightlight your answer(Yes/No) with <answer></answer>. Please think step by step first, and then answer the question.'''
    # prompt = '''Please review the 12 situational background statements provided. For each statement, briefly analyze its relevance to the user's question and how it impacts the appropriateness of my response. After this step-by-step analysis, conclude with a clear 'Yes' or 'No' to indicate whether my response is appropriate in the context of these statements by enclosing it within '<answer></answer>' tags. For example, <answer>Yes</answer> or <answer>No</answer>.'''
    prompt = '''
    ###Instructions###
    You are tasked with evaluating my response to a user's question. This evaluation is based on 12 situational background statements provided to you.
    Your task is to determine whether my response is appropriate in the context of these statements and the user's question.
    Please organize your thoughts into four steps : Step 1: Identify the user's request and the context information relevant to the question and the response. Step 2: Evaluate the relevance of the response to the user's request. Step 3: Evaluate the appropriateness of the response in the context of the background information. Step 4: Provide a final answer.
    Clearly indicate your final answer "Yes" or "No" by enclosing it within '<answer></answer>' tags. For example, <answer>Yes</answer> or <answer>No</answer>.
    I'm going to tip you if your answer is correct !
    '''
    background = "".join(data['s'])
    question = data['u']
    response = data['r']
    
    backgorund_info = f"###Background###\n"
    for i,(type, bg) in enumerate(zip(data['s.type'],data['s'])):
        # backgorund_info += f"\n {i+1}. [{type}] {bg} [/{type}]"
        backgorund_info +=  f"\n {i+1}. {bg}"
    # backgorund_info +=f"\n[/Background]"
    # backgorund_info = f"[Background]: {background} [/Background]"
    text= f"{prompt}\n\n{backgorund_info}\n\n###User's question###\n{question}\n\n###My response###\n{response}"
    return text
def read_json(mode):
    with open(f"dataset/{mode}.json", 'r') as file:
        data = json.load(file)
    data = pd.DataFrame(data)
    return data
test_data = read_json("test")
test_data['text'] = test_data.apply(formatting_text,axis=1)
print(test_data['text'].tolist()[0])

class StopWords(StoppingCriteria):
    def __init__(self, tk, stop_words: list[str]):
        self.tk = tk
        self.stop_tokens = stop_words

    def __call__(self, input_ids, *_) -> bool:
        s = self.tk.batch_decode(input_ids)[0]
        for t in self.stop_tokens:
            if s.endswith(t):
                return True
        return False
sw = StopWords(tokenizer, ["</answer>"])
scl = StoppingCriteriaList([sw])

# def batch_inference(texts: List[str], tokenizer, model, batch_size: int):
#     predictions = []
#     all_outputs = []
#     for i in trange(0, len(texts), batch_size):
#         batch_texts = texts[i:i+batch_size]
#         tokenizer.padding_side = 'left'
#         model_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        
#         with torch.no_grad():
#             batch_outputs = model.generate(**model_inputs, max_new_tokens=5000,stopping_criteria=scl,pad_token_id=tokenizer.eos_token_id)
#             batch_decoded = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        
#         # Post-process each item in the batch
#         for j, output in enumerate(batch_decoded):
#             clean_output = output.replace(batch_texts[j], "").strip()
#             predictions.append(clean_output)
#         if not os.path.exists("predictions.txt"):
#             with open("predictions.txt","w") as f:
#                 for pre in predictions:
#                     f.write(str(pre)+"\n")
#         else:
#             with open("predictions.txt","a") as f:
#                 for pre in predictions:
#                     f.write(str(pre)+"\n")
#         break
#     pd.DataFrame(predictions, columns = ['result']).to_csv("predictions.csv",index=False)
#     return predictions

# # Example usage
# batch_size = 5  # Set batch size based on your GPU memory constraints
# predictions = batch_inference(test_data['text'].tolist(), tokenizer, model, batch_size)
predictions = []
with torch.no_grad():
    for i in trange(len(test_data)):
        # if i==5:
        #     break
        eval_prompt = test_data['text'][i]
        tokenizer.padding_side = 'left'
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        pred = tokenizer.decode(model.generate(**model_input, max_new_tokens=500,pad_token_id = tokenizer.eos_token_id,stopping_criteria=scl)[0], skip_special_tokens=True)
        pred = pred.replace(eval_prompt,"")
        print(pred)
        predictions.append(pred)
        if not os.path.exists("predictions.txt"):
            with open("predictions.txt","w") as f:
                f.write(str(pred)+"\n")
        else:
            with open("predictions.txt","a") as f:
                f.write(str(pred)+"\n")
pd.DataFrame(predictions, columns = ['result']).to_csv("predictions.csv",index=False)