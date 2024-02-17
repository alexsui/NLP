import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
import json
import re
from tqdm import tqdm, trange
import os
from transformers import StoppingCriteriaList,StoppingCriteria

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.add_eos_token = True
tokenizer.add_bos_token=True
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_bos_token, tokenizer.add_eos_token
def formatting_text(data):
    # prompt = """[INST]根據所提供的文章内容，為每個问题选择一个合理且正确的答案。回答时请使用以下格式：在 '<answer>' 和 '</answer>' 之间填写答案，格式为 '問題題號 : 选项数字.正确答案'。範例 :<answer>問題1: 1.答案</answer> """
    # prompt = """
    # ###系统指令###
    # 请仔细阅读所提供的文章内容。根据文章的信息，为每个问题选择最合理且正确的答案。在回答每个问题时，你需要“严格遵守”以下的格式规定：

    # - 每个回答都应该放在 '<answer>' 和 '</answer>' 标签之间。
    # - 每个回答的格式应该是 '问题编号 : 选项数字.正确答案'。
    # - 确保每个问题的答案都是清晰并且独立的。

    # 例如，如果文章中的问题1的正确答案是选项1，问题2的正确答案是选项2，你应该这样回答：

    # <answer>问题1: 1.正确答案</answer>
    # <answer>问题2: 2.正确答案</answer>

    # 请确保按照这种格式准确回答每个问题，以便答案可以被清楚地识别和整理。
    # """
    prompt = """###系统指令###
    你是一个专业的文章问题回答人员，你的任务是根据所提供的文章内容，为每个问题选择一个符合文意且正确合理的答案。如果你的回答是正確的，我会给你小费!"""
    paragraph = data['paragraph']
    all_questions = ""
    for i,q in enumerate(data["questions"]):
        question = f"问题{i+1}:\n{q['question']}"
        options = " ".join([f"{i+1}.{c}"for i,c in enumerate(q['choice'])])
        all_questions+= f"{question}\n选项:{options}\n"
    text= f"{prompt} \n\n###文章内容###\n{paragraph}\n###问题###\n{all_questions}[/INST]"
    return text
def read_json(mode):
    with open(f"dataset/{mode}.json", 'r') as file:
        data = json.load(file)
    data = pd.DataFrame(data)
    return data

ft_model = PeftModel.from_pretrained(base_model, "./models/chat_model_finetune2/checkpoint-1400")
test_data = read_json("test")
test_data['text'] = test_data.apply(formatting_text,axis=1)
print(test_data['text'].tolist()[0])
ft_model.eval()
predictions = []
with torch.no_grad():
    for i in trange(len(test_data)):
        eval_prompt = test_data['text'][i]
        tokenizer.padding_side = 'left'
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        pred = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=200,pad_token_id = tokenizer.eos_token_id)[0], skip_special_tokens=True)
        pred = pred.replace(eval_prompt,"")
        predictions.append(pred)
        # if not os.path.exists("predictions_8x7b.txt"):
        #     with open("predictions_8x7b.txt","w") as f:
        #         f.write(str(pred)+"\n")
        # else:
        #     with open("predictions_8x7b.txt","a") as f:
        #         f.write(str(pred)+"\n")
pd.DataFrame(predictions, columns = ['result']).to_csv("predictions_finetune.csv",index=False)
