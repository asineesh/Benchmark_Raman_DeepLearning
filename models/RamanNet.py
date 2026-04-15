#From RamanNet: A generalized neural network architecture for Raman Spectrum Analysis
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RamanNet(nn.Module):
    def __init__(self,len_window=50,step_size=25,sp_size=1024,num_classes=15):
        """
        len_window is an integer representing the length of the window
        step_size is an integer representing the stride
        sp_size is an integer representing the length of the spectra
        num_classes is the number of classes 
        """
        super().__init__()
        self.len_window = len_window
        self.step_size = step_size
        self.sp_size = sp_size

        num_dense_blocks = math.floor((sp_size-len_window)/step_size)+1
        extra_dense_blocks = math.ceil(sp_size/step_size)-num_dense_blocks
        self.dense_blocks = nn.ModuleList()

        for _ in range(num_dense_blocks):            
            self.dense_blocks.append(nn.Sequential(nn.Linear(len_window,step_size),nn.BatchNorm1d(step_size),nn.LeakyReLU()))
        
        for i in range(extra_dense_blocks):
            self.dense_blocks.append(nn.Sequential(nn.Linear(sp_size-num_dense_blocks*step_size-i*step_size,step_size),nn.BatchNorm1d(step_size),nn.LeakyReLU()))

        self.dropout1 = nn.Dropout(0.5)
        self.dense = nn.Linear((num_dense_blocks+extra_dense_blocks)*step_size,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.4)
        self.emb = nn.Linear(512,256)

        self.bn2 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.25)

        self.head = nn.Linear(256,num_classes)
    
    def forward(self,x):
        """
        x is of shape (b,1,sp_size)
        """
        x = x.squeeze(1) #of shape (b,sp_size)
        feats = []

        for i in range(len(self.dense_blocks)):
            feats.append(self.dense_blocks[i](x[:,i*self.step_size:min(i*self.step_size+self.len_window,self.sp_size)]))

        feats = torch.cat(feats,1) #of shape (b,num_blocks*sp_size)
        feats = self.dropout1(feats)

        feats = self.relu(self.bn1(self.dense(feats))) #of shape (b,512)
        feats = self.dropout2(feats)

        feats = self.emb(feats) #of shape (b,256)
        emb = F.normalize(feats,dim=1)

        out = self.relu(self.bn2(feats))
        out = self.dropout3(out)

        out = self.head(out) #of shape (b,num_classes)

        return out,emb



