import optuna
import plotly as plot
import torch
from dataset import myDataset
from torch.utils.data import Dataset, DataLoader
from model import BaselineModel, ELMO, KaggleModel
from trainer import Trainer
from optimizer import Lion
import numpy as np
import random
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_path = "./HW1_dataset/train.json"
val_data_path = "./HW1_dataset/val.json"
test_data_path = "./HW1_dataset/test.json"
vocab = np.load("./glove/vocab.npy")
embs = np.load("./glove/embeddings.100d.npy")
train_dataset = myDataset(train_data_path, vocab, max_seq_length= 55,pad_token='<pad>', unk_token='<unk>',kind="elmo_new",  mode ="train")
val_dataset = myDataset(val_data_path, vocab, max_seq_length= 55,pad_token='<pad>', unk_token='<unk>',kind="elmo_new", mode ="valid")
test_dataset = myDataset(test_data_path, vocab, max_seq_length= 55,pad_token='<pad>', unk_token='<unk>', kind="elmo_new", mode ="test")

def objective(trial):
    # Generate the hyperparameters to be tuned
    # lr = 10**trial.suggest_int('logval', -5, -2)
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64,128,256])
    num_layers = trial.suggest_int('num_layers', 1, 3, step=1) 
    hidden_size= trial.suggest_int('hidden_size', 64, 512, step=64)
    dropout = trial.suggest_float('dropout', 0.1, 0.4,step=0.1)
    # is_attention = trial.suggest_categorical('is_attention', [True, False])
    bidirectional = trial.suggest_categorical('bidirectional', [True, False])
    # freeze = trial.suggest_categorical('freeze', [True, False])
    # embedding_table_lr = trial.suggest_loguniform('embedding_table_lr', 1e-8, 1e-4)
    lambda_ = trial.suggest_loguniform('lambda_', 1e-6, 1e-2)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = KaggleModel(embedding_dim=1024, hidden_dim=hidden_size, num_layers=num_layers, num_classes=12,
                      dropout=dropout, bidirectional=bidirectional)
    model.to(device)
    EPOCHs = 100
    opt = Lion(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, verbose=True)
    trainer = Trainer(model,opt=opt, device = device, lambda_ = lambda_)
    res = trainer.train(epochs= EPOCHs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, patience=5, model_name="trial", scheduler=None)
    return max(res['val_acc'])
# Optimize the hyperparameters using Optuna
study = optuna.create_study(direction='maximize')  # or 'minimize' based on your goal
study.optimize(objective, n_trials=350)

# Get the best parameters
best_params = study.best_params
best_value = study.best_value
print("best_params:", best_params)
print("best_value:", best_value)