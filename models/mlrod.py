#From Deep convolutional neural networks for Raman spectrum recognition: A unified solution with slight modification based on the MLROD paper.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLROD_model(nn.Module):
    def __init__(self,sp_size=1024,num_classes=15):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,21)
        self.conv2 = nn.Conv1d(16,32,11)
        self.conv3 = nn.Conv1d(32,64,5)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.norm1 = nn.BatchNorm1d(16)
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2,2)
        l = (((((sp_size-20)//2)-10)//2)-4)//2

        self.dense = nn.Linear(64*l,2048)
        self.head = nn.Linear(2048,num_classes)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.55)
        self.norm4 = nn.BatchNorm1d(2048)
        self.norm5 = nn.BatchNorm1d(num_classes)
    
    def forward(self,x):
        b = x.shape[0]
        x = self.relu(self.norm1(self.conv1(x))) #of shape (b,16,1004)        
        x = self.pool(x) #of shape (b,16,502)

        x = self.relu(self.norm2(self.conv2(x))) #of shape (b,32,492)
        x = self.pool(x) #of shape (b,32,246)

        x = self.relu(self.norm3(self.conv3(x))) #of shape (b,64,242)
        x = self.pool(x) #of shape (b,64,121)

        x = x.view(b,-1)
        x = self.tanh(self.norm4(self.dense(x))) #of shape (b,2048)
        x = self.drop(x)
        out = self.norm5(self.head(x))

        return out #of shape (b,15)


