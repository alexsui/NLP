import torch
import torch.nn as nn
import torch.nn.functional as F
class BaselineModel(nn.Module):
    def __init__(self, embed_size, pretrain_embeddings, hidden_size, num_layers, num_classes, dropout, is_attention = False, bidirectional = False, freeze = False):
        super(BaselineModel, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.from_numpy(pretrain_embeddings).float(), freeze=freeze)
        self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout = dropout, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(2*hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        self.is_attention = is_attention
        self.dropout = nn.Dropout(dropout)
        if is_attention:
            if bidirectional:
                self.attention = nn.Linear(2*hidden_size, 1)
            else:
                self.attention = nn.Linear(hidden_size, 1) 
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        if self.is_attention:
            attention_weights = torch.softmax(self.attention(x), dim=1)  # Attention weights
            x = torch.sum(attention_weights * x, dim=1)
            x = self.dropout(x)
            x = self.fc(x)
        else:
            x = self.dropout(x)
            x = self.fc(x[:, -1, :])
        return x

class ELMO(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, num_classes, dropout, is_attention = False, bidirectional = False, freeze = False):
        super(ELMO, self).__init__()
        self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout = dropout, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(2*hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        self.is_attention = is_attention
        self.dropout = nn.Dropout(dropout)
        if is_attention:
            if bidirectional:
                self.attention = nn.Linear(2*hidden_size, 1)
            else:
                self.attention = nn.Linear(hidden_size, 1) 
    def forward(self, x):
        x, _ = self.lstm(x)
        if self.is_attention:
            attention_weights = torch.softmax(self.attention(x), dim=1)  # Attention weights
            x = torch.sum(attention_weights * x, dim=1)
            x = self.dropout(x)
            x = self.fc(torch.relu(x))
        else:
            x = self.dropout(x)
            x = self.fc(torch.relu(x[:, -1, :]))
        return x

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

class KaggleModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,num_layers, num_classes, dropout,bidirectional):
        super(KaggleModel, self).__init__()
        # Dropout
        self.spatial_dropout = nn.Dropout1d(dropout)
        self.dropout = nn.Dropout(dropout)
        # Bidirectional GRU layers
        self.bi_gru = nn.GRU(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        
        
        # Pooling Layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Attention Layer
        if bidirectional:
            self.attention = AttentionWeightedAverage(2*hidden_dim)
        else:
            self.attention = AttentionWeightedAverage(hidden_dim)
        
        # Dense Layers
        if bidirectional:
            self.dense1 = nn.Linear(8*hidden_dim, hidden_dim)
        else:
            self.dense1 = nn.Linear(4*hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.spatial_dropout(x.permute(0,2,1)).permute((0,2,1))  # Apply dropout
        
        x, _ = self.bi_gru(x)
        # x, _ = self.bi_gru_2(x)
        
        # Concatenate the pooling layers and attention
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
