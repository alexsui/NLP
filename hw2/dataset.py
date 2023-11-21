from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import torch
from torch.utils.data import Dataset
class ConversationDataset(Dataset):
    def __init__(self, filename: str, tokenizer, max_len: int = 25, k=3, mode='train'):
        self.data = pd.read_csv(filename, sep='\t')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.classes_list  = pd.read_csv('dataset/sample_submission.csv').columns[1:].tolist()
        self.k = k
        if mode != "test":
            self.classes = self.data['classes'].apply(lambda x: x.split(','))
            self.mlb = MultiLabelBinarizer(classes=self.classes_list)
            self.mlb.fit(self.classes)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if self.mode !="train":
            utterance = self.data.iloc[idx]['utterance']
            tokens = self.tokenizer(utterance, max_length=self.max_len, padding='max_length', truncation=True)
            if self.mode == "test":
                return {
                    'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(tokens['attention_mask'], dtype=torch.long),
                }
            elif self.mode =="valid":
                label = self.mlb.transform([self.classes.iloc[idx]])[0]
                return {
                    'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
                    'attention_mask': torch.tensor(tokens['attention_mask'], dtype=torch.long),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
        else: # train mode 
            if idx<2:
                utterance = self.data.iloc[idx]['utterance']
                label = self.mlb.transform([self.classes.iloc[idx]])[0]
            else:
                utterance = ' '.join([self.data.iloc[max(0, idx - i)]['utterance'] for i in range(self.k)][::-1])
                label = [self.classes.iloc[max(0, idx - i)][0] for i in range(self.k)]
                label = self.mlb.transform([label])[0]
            # print("utterance:",utterance)
            # print("original labels: ", label)
            tokens = self.tokenizer(utterance, max_length=self.max_len, padding='max_length', truncation=True)
            return {
                'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(tokens['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        
           