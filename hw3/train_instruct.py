import pandas as pd 
import torch
import torch.nn as nn
import transformers
import json
from datasets import load_dataset
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging,Trainer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch
from datasets import load_dataset
from datasets import load_metric
from trl import SFTTrainer
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
model_name = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(  
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= True,
)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)
model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = 'left'
tokenizer.add_eos_token = True
tokenizer.add_bos_token=True
tokenizer.pad_token = tokenizer.eos_token

def formatting_text(data):
    # prompt = """[INST]根據所提供的文章内容，為每個问题选择一个符合文意且正确合理的答案。回答时请使用以下格式：在 '<answer>' 和 '</answer>' 之间填写答案，格式为 '問題題號 : 选项数字.正确答案'。範例 :<answer>問題1: 1.答案</answer> """
    prompt = """###系统指令###
    你是一个专业的文章问题回答人员，你的任务是根据所提供的文章内容，为每个问题选择一个符合文意且正确合理的答案。如果你的回答是正確的，我会给你小费!"""
    paragraph = data['paragraph']
    all_questions = ""
    for i,q in enumerate(data["questions"]):
        question = f"问题{i+1}:\n{q['question']}"
        options = " ".join([f"{i+1}.{c}"for i,c in enumerate(q['choice'])])
        all_questions+= f"{question}\n选项:{options}\n"
    all_answers = ""
    for i,q in enumerate(data["questions"]):
        a = q['choice']
        answer = f"{a.index(q['answer'])+1}. {q['answer']}"
        all_answers+= f"<answer>问题{i+1}: {answer}</answer>\n"
    text= f"{prompt} \n\n###文章内容###\n{paragraph}\n###问题###\n{all_questions}"
    return text
def formatting_label(data):
    all_answers = ""
    for i,q in enumerate(data["questions"]):
        a = q['choice']
        question = f"问题{i+1}:"
        answer = f"{a.index(q['answer'])+1}.{q['answer']}"
        all_answers+= f"{question} {answer}\n"
    text = f"\n###答案###\n{all_answers}"
    return text
def format_prompt(data):
    prompt = f"{data['text']}\n{data['label']}"
    return prompt
def read_json(mode):
    with open(f"dataset/{mode}.json", 'r') as file:
        data = json.load(file)
    data = pd.DataFrame(data)
    return data
train_data = read_json("train")
valid_data = read_json("val")
test_data = read_json("test")
train_data['text'] = train_data.apply(formatting_text,axis=1)
train_data['label'] = train_data.apply(formatting_label,axis=1)
train_data['prompt'] = train_data.apply(format_prompt,axis=1)
train_data = train_data[['text','label','prompt']]
valid_data['text'] = valid_data.apply(formatting_text,axis=1)
valid_data['label'] = valid_data.apply(formatting_label,axis=1)
valid_data['prompt'] = valid_data.apply(format_prompt,axis=1)
valid_data = valid_data[['text','label','prompt']]

print(train_data['prompt'][0])

def preprocess(data): #full prompt
    model_inputs = tokenizer(data['prompt'], padding="max_length", max_length=1930,truncation=True, return_tensors='pt')
    # label = tokenizer(data['label'], padding="max_length", max_length=50,truncation=True, return_tensors='pt')
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs
new_train_data = datasets.Dataset.from_pandas(train_data,split='train')
new_valid_data = datasets.Dataset.from_pandas(valid_data,split='train')
tokenized_train_data = new_train_data.map(preprocess,remove_columns = new_train_data.column_names,batched=True)
tokenized_valid_data = new_valid_data.map(preprocess,remove_columns = new_valid_data.column_names,batched=True)
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[ "q_proj",
        "k_proj",""
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",]
)
model = get_peft_model(model, peft_config)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(model)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare_model(model)
def preprocess_logits_for_metrics(logits, labels):
    if type(logits)==tuple:
        logits = logits[0]
    logits = logits.argmax(axis=-1)
    return logits
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = preprocess_logits_for_metrics(logits,labels)
    predictions = logits.argmax(axis=-1)
    accuracy_metric = load_metric("accuracy")
    return accuracy_metric.compute(predictions=predictions, references=labels)
training_arguments = TrainingArguments(
    output_dir="./models/chat_model_finetune2", #baseline_with_complete_prompt
    logging_dir="./logs/chat_model_finetune2",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=100,
    save_total_limit = 4,
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    do_eval=True,
    lr_scheduler_type="constant",
    metric_for_best_model = "eval_loss",
)
# trainer = Trainer(
#     model=model,
#     train_dataset=tokenized_train_data,
#     eval_dataset = tokenized_valid_data,
#     tokenizer=tokenizer,
#     args=training_arguments,
#     data_collator=data_collator,
# )
trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_data,
    eval_dataset = tokenized_valid_data,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=data_collator,

)
trainer.train()
print(trainer.state.best_model_checkpoint)