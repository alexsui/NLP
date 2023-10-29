from tqdm import tqdm
from pathlib import Path
import ipdb
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
import ipdb
from sklearn.metrics import f1_score
class Trainer(object):
    def __init__(self, model, opt, device, lambda_=0.0001):
        self.model = model
        self.device = device
        self.opt = opt
        self.train_loss_function = nn.BCEWithLogitsLoss()
        self.test_loss_function = nn.BCEWithLogitsLoss()
        self.classes = list(pd.read_csv("./HW1_dataset/sample_submission.csv").columns[1:])
        self.lambda_ = lambda_
    def train_step(self, train_dataloader):
        train_acc = 0
        train_loss = 0
        self.model.train()
        for x,y in train_dataloader:
            x,y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.train_loss_function(pred, y)
            l2_reg = torch.tensor(0.).to(self.device)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss += self.lambda_ * l2_reg
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            train_loss += loss.item()
            
            y_prob = torch.sigmoid(pred)>0.5
            true_labels = y.cpu().detach().numpy()
            predicted_labels = y_prob.int().cpu().detach().numpy()
            # Compute the macro F1 score
            macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
            train_acc += macro_f1
        return train_loss/len(train_dataloader), train_acc/len(train_dataloader)

    def val_step(self, val_dataloader): 
        test_acc = 0
        test_loss = 0
        self.model.eval()
        with torch.inference_mode():
            for x,y in val_dataloader:
                x,y = x.to(self.device), y.to(self.device)
                pred_y = self.model(x)
                loss = self.test_loss_function(pred_y,y) 
                test_loss += loss.item()
                y_prob = torch.sigmoid(pred_y)>0.5
                true_labels = y.cpu().detach().numpy()
                predicted_labels = y_prob.int().cpu().detach().numpy()
                macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
                test_acc += macro_f1
        return test_loss/len(val_dataloader), test_acc/len(val_dataloader)
    def test_step(self, test_dataloader): 
        self.model.eval()
        all_data =np.zeros((len(test_dataloader),12))
        with torch.inference_mode():
            for i,x in enumerate(test_dataloader):
                x = x.to(self.device)
                pred_y = self.model(x)
                y_prob = (torch.sigmoid(pred_y)>0.5).int().cpu().detach().numpy()
                all_data[i] = np.expand_dims(y_prob,0)
        df =pd.DataFrame(all_data, columns= self.classes)
        return df
    def train(self, epochs, train_dataloader, val_dataloader, patience, model_name, scheduler):
        last_loss = float("inf")
        cur = 0
        results ={
            "train_loss":[],
            "train_acc":[],
            "val_loss":[],
            "val_acc":[]
        }
        for epoch in range(epochs):
            train_loss, train_acc = self.train_step(train_dataloader)
            test_loss, test_acc = self.val_step(val_dataloader)
            if scheduler:    
                scheduler.step(test_loss)
            if (epoch+1)%5 == 0:
                if test_loss > last_loss:
                    cur += 1
                    print('trigger times:', cur)
                    if cur >= patience:
                        print("early stop !")
                        return results
                else:
                    cur = 0
            last_loss = test_loss
            print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {test_loss:.4f} | "
            f"val_acc: {test_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(test_loss)
            results["val_acc"].append(test_acc)
            
            if (epoch+1)%10 == 0:
                MODEL_PATH = Path("models/"+model_name)
                MODEL_PATH.mkdir(parents=True, 
                                exist_ok=True
                )

                MODEL_NAME = f"model_{epoch+1}.pth"
                MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
                print(f"Saving model to: {MODEL_SAVE_PATH}")
                torch.save(obj=self.model.state_dict(),
                        f=MODEL_SAVE_PATH)
        MODEL_PATH = Path("models/"+model_name)
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        torch.save(obj=self.model.state_dict(),
                        f=MODEL_PATH/f"best_model_E{epoch}.pth")
        return results