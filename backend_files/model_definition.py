import torch
import torch.nn as nn

class ActigraphyEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=5):
        super(ActigraphyEncoder, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.embedding_head = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, return_embedding=False):
        feats = self.feature_extractor(x)
        feats = feats.permute(0, 2, 1)
        
        lstm_out, (hn, _) = self.lstm(feats)
        
        hidden_cat = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        embedding = self.embedding_head(hidden_cat)
        
        if return_embedding:
            return embedding
            
        logits = self.classifier(embedding)
        return logits, embedding