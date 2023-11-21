import optuna
import plotly as plot
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trainer import Trainer
from optimizer import Lion
import numpy as np
import random
from dataset import ConversationDataset
from model import RoBERTaClassifier
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tokenizer initialization




def objective(trial):
    # Generate the hyperparameters to be tuned
    # lr = 10**trial.suggest_int('logval', -5, -2)
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    model_name = trial.suggest_categorical('model_name', ["voidism/diffcse-roberta-base-trans"])
    # max_len = trial.suggest_categorical('max_len', [20, 25, 50])
    max_len = trial.suggest_int('max_len', 20, 60, step=5)
    k = trial.suggest_int('k', 1, 3,step=1)
    # dropout = trial.suggest_float('dropout', 0.1, 0.4,step=0.1)
    # is_attention = trial.suggest_categorical('is_attention', [True, False])
    # bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    # freeze = trial.suggest_categorical('freeze', [True, False])
    # embedding_table_lr = trial.suggest_loguniform('embedding_table_lr', 1e-8, 1e-4)
    lambda_ = trial.suggest_loguniform('lambda_', 1e-6, 1e-2)
    threshold = trial.suggest_float('threshold', 0.3, 0.6, step=0.05)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 384, 768])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ConversationDataset('dataset/train.tsv', tokenizer, max_len=max_len, k=k, mode="train")
    val_dataset = ConversationDataset('dataset/val.tsv', tokenizer, max_len=max_len, mode="valid")
    # DataLoader initialization
    train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size= batch_size, shuffle=False)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(train_dataset.mlb.classes_))
    model = RoBERTaClassifier(model_name, hidden_size, len(train_dataset.mlb.classes_))
    model.to(device)
    optimizer = Lion(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, torch.device("cuda"), lambda_=lambda_, threshold=threshold)
    res = trainer.train(epochs=40, train_dataloader=train_dataloader, val_dataloader=val_dataloader, patience=5, model_name="test", scheduler=None)
    return max(res['val_acc'])
# Optimize the hyperparameters using Optuna
study = optuna.create_study(direction='maximize')  # or 'minimize' based on your goal
study.optimize(objective, n_trials=500)

# Get the best parameters
best_params = study.best_params
best_value = study.best_value
print("best_params:", best_params)
print("best_value:", best_value)