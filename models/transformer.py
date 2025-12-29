import torch
import torch.nn as nn
import torch.nn.functional as F

#From Classification of deep‑sea cold seep bacteria by transformer combined with Raman spectroscopy

class Patch_Embedding(nn.Module):
    def __init__(self,in_channels=1,embedding_size=768,patch_size=64,sp_size=1024):
        #Assuming that the spectra size is perfectly divisible by patch_size
        super().__init__()
        self.embedding_size = embedding_size
        self.num_patches = sp_size//patch_size
        self.conv = nn.Conv1d(in_channels,embedding_size,patch_size,stride=patch_size) #Divides up the spectra into patches
        self.cls = nn.Parameter(torch.randn((1,embedding_size))) #The cls token
        self.pos = nn.Parameter(torch.randn((self.num_patches+1,embedding_size))) #Position embedding

    def forward(self,x):
        # x is of shape (b,c,h)
        batch_size = x.shape[0]
        y = self.conv(x) #y is of shape (b,embedding_size,h/patch_size)
        out = y.view(-1,self.embedding_size,self.num_patches) #out is of shape (b,embedding_size,h/patch_size)
        out = torch.permute(out,(0,2,1)) #out is of shape (b,num_patches,embedding_size)

        #Appending the cls token
        cout = torch.concat((out,self.cls.repeat(batch_size,1,1)),1) #cout is of shape (b,num_patches+1,embedding_size)

        #Adding the position embedding
        pcout = cout+self.pos.repeat(batch_size,1,1)  #pcout is of shape (b,num_patches+1,embedding_size)

        return pcout


class ClassificationHead(nn.Module):
    def __init__(self,embedding_size=768,num_classes=15):
        super().__init__()
        self.fc = nn.Linear(embedding_size,num_classes)

    def forward(self,x):
        #x is of shape (b,num_patches,embedding_size)
        x = x[:,-1,:] #Taking the output of the last patch
        out = self.fc(x)
        return out #(b,num_classes)

class MLP(nn.Module):
    #The MLP block within the transformer
    def __init__(self,embedding_size=768,p=0.0,expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size,expansion*embedding_size)
        self.fc2 = nn.Linear(expansion*embedding_size,embedding_size)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self,x):
        #x is of shape (b,num_patches,embedding_size)
        x = self.gelu(self.fc1(x))
        out = self.fc2(self.drop(x))
        return out #(b,num_patches,embedding_size)

class Transformer(nn.Module):
    #The overall Transformer block
    def __init__(self,embedding_size=768,p=0.0,expansion=4,num_heads=12):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_size)
        self.MAH = nn.MultiheadAttention(embedding_size, num_heads, p,batch_first=True)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.mlp = MLP(embedding_size,p,expansion)

    def forward(self,x):
        #x is of shape (batch_size,num_patches,embedding_size)
        identity1 = x
        x = self.ln1(x)
        x,_ = self.MAH(x,x,x)
        identity2 = x + identity1 #skip connection
        out = self.mlp(self.ln2(identity2))
        out = out + identity2 #skip connection
        return out #of shape (batch_size,num_patches,embedding_size) 

#Overall ViT implementation
class ViT(nn.Module):
    def __init__(self,embedding_size=768,p=0.1,expansion=4,num_heads=12,in_channels=1,patch_size=64,sp_size=1024,num_classes=15,depth=12):
        super().__init__()

        self.embedding = Patch_Embedding(in_channels,embedding_size,patch_size,sp_size)

        Tlayers = [] #A ViT would have multiple (depth) Transformer blocks
        for i in range(depth):
            Tlayers.append(Transformer(embedding_size,p,expansion,num_heads))

        self.Tlayers = nn.Sequential(*Tlayers)

        self.head = ClassificationHead(embedding_size,num_classes)

    def forward(self,x):
        #Getting the embeddings of each patch of all the batch spectra
        x = self.embedding(x)

        #Passing them through "depth" Transformer blocks
        x = self.Tlayers(x)

        #Passing the output through classification head
        out = self.head(x)
        return out