import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel,RobertaModel
class AttentionWeightedAverage(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWeightedAverage, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Compute attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)
        # Weighted sum of the inputs based on attention weights
        weighted_average = torch.sum(x * attention_weights, dim=1)
        return weighted_average

class Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout=0.2):
        super(Classifier, self).__init__()
        self.spatial_dropout = nn.Dropout1d(dropout)
        self.dropout = nn.Dropout(dropout)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.attention = AttentionWeightedAverage(embedding_dim)

        self.dense1 = nn.Linear(4*embedding_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        # Concatenate the pooling layers and attention
        x = self.spatial_dropout(x.permute(0,2,1)).permute((0,2,1))
        max_pool = self.global_max_pool(x.permute(0,2,1)).squeeze(2)
        avg_pool = self.global_avg_pool(x.permute(0,2,1)).squeeze(2)
        attention_pool = self.attention(x)
        last = x[:, -1, :]
        # print(last.shape, max_pool.shape, avg_pool.shape, attention_pool.shape) #torch.Size([128, 256]) torch.Size([128, 256]) torch.Size([128, 256]) torch.Size([128, 256])
        concatenated = torch.cat([last, max_pool, avg_pool, attention_pool], dim=1)
        concatenated = self.dropout(concatenated)
        x = F.relu(self.dense1(concatenated))
        x = self.dense2(x)
        
        return x
class RoBERTaClassifier(nn.Module):
    def __init__(self, pretrained_model_name,hidden_size, num_classes):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = AutoModel.from_pretrained(pretrained_model_name)
        self.classifier = Classifier(self.roberta.config.hidden_size, hidden_size, num_classes)
        self.loss_function = nn.BCEWithLogitsLoss()
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        roberta_embedding = outputs.last_hidden_state
        logits = self.classifier(roberta_embedding)
        if labels is not None:
            loss = self.loss_function(logits, labels)
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}