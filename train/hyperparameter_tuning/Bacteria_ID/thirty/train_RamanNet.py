import torch
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader, random_split
from torchvision import transforms

from tqdm import tqdm
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from models.RamanNet import RamanNet
from triplet_loss import TripletLoss
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import copy

class Bacteria_Dataset(Dataset):
    def __init__(self,X_path,y_path):
        """
        X_path is a string containing the path to the spectra
        y_path is a string containing the path to the labels for the spectra
        """
        super().__init__()
        self.X = np.load(X_path)
        self.y = np.load(y_path)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        data = torch.Tensor(self.X[index]).unsqueeze(0) #of shape (1,1000)
        data = (data-data.min())/(data.max()-data.min())
        label = self.y[index]
        return data,int(label)

def train(model,device,train_dataloader,criterion1,criterion2,optimizer):
    model.train()
    loop = tqdm(train_dataloader)
    total_loss = 0
    correct = 0

    for i,(X,y) in enumerate(loop):
        X = X.to(device)
        y = y.to(device)

        y_pred,y_emb = model(X) #of shape (b,15)

        #Applying cross-entropy loss
        ce_loss = criterion1(y_pred,y)

        #Applying triplet loss
        tr_loss = criterion2(y_emb,y)

        loss = 0.5*ce_loss+0.5*tr_loss

        total_loss += loss.item()
        loop.set_postfix(loss=total_loss/(i+1))

        prediction = y_pred.argmax(dim=1) #of shape (b)
        correct += sum(prediction==y).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model,device,test_dataloader,criterion):
    model.eval()
    loop = tqdm(test_dataloader)
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, (X,y) in enumerate(loop):
            X = X.to(device)
            y = y.to(device)

            y_pred,_ = model(X) #of shape (b,15)
            prediction = y_pred.argmax(dim=1) #of shape (b)
            correct += sum(prediction==y).item()

            loss = criterion(y_pred,y)
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/(i+1))
    
    print(f'Classification accuracy: {round(100*correct/len(test_dataloader.dataset),2)}')
    return 100*correct/len(test_dataloader.dataset)

def test_f1(model,device,test_dataloader,criterion):
    model.eval()
    loop = tqdm(test_dataloader)

    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (X, y) in enumerate(loop):
            X = X.to(device)
            y = y.to(device)

            y_pred,_ = model(X)#of shape (b,num_classes)                  
            preds = y_pred.argmax(dim=1) #of shape (b)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

            loss = criterion(y_pred, y)
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (i + 1))

    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy() #of shape (num_samples,)
    all_targets = torch.cat(all_targets).numpy() #of shape (num_samples,)

    accuracy = 100.0 * (all_preds == all_targets).mean()
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f'Classification accuracy: {round(accuracy,2)}')

    return accuracy, precision, recall, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/hyperparameter_tuning/Bacteria_ID/thirty/", exist_ok=True)
    os.makedirs("results/hyperparameter_tuning/Bacteria_ID/models/", exist_ok=True)
    os.makedirs("results/hyperparameter_tuning/trained_models/", exist_ok=True)
    
    filename = "results/hyperparameter_tuning/Bacteria_ID/thirty/results_RamanNet.txt"
    print(device)

    epochs = 40
    stopping_epochs = 10
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = TripletLoss(device)

    batch_sizes = [32,128,512]
    lrs = [0.001,0.0001,0.00001]
    all_best_val_acc = 0 #For validation
    all_best_test_acc = [0,0,0,0] #For test
    best_hyper = ""
    best_final_model_name = ""

    random.seed(42)

    for batch_size in batch_sizes:
        for lr in lrs:
            train_set = Bacteria_Dataset("datasets/Bacteria_ID/X_reference.npy","datasets/Bacteria_ID/y_reference.npy")
            fine_set = Bacteria_Dataset("datasets/Bacteria_ID/X_finetune.npy","datasets/Bacteria_ID/y_finetune.npy")
            test_set = Bacteria_Dataset("datasets/Bacteria_ID/X_test.npy","datasets/Bacteria_ID/y_test.npy")
            generator = torch.manual_seed(42)
            train_train_set, train_val_set = random_split(train_set,[0.8,0.2],generator=generator)
            fine_train_set, fine_val_set = random_split(fine_set,[0.8,0.2],generator=generator)

            train_train_loader = DataLoader(train_train_set, batch_size=batch_size, num_workers=8, shuffle=True)
            train_val_loader = DataLoader(train_val_set, batch_size=batch_size, num_workers=8, shuffle=True)
            fine_train_loader = DataLoader(fine_train_set, batch_size=batch_size, num_workers=8, shuffle=True)
            fine_val_loader = DataLoader(fine_val_set, batch_size=batch_size, num_workers=8, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8,shuffle=False)

            iteration = f"For Batch size {batch_size} and Learning Rate {lr}"

            with open(filename,"a", encoding="utf-8") as f:
                    f.write("\n"+iteration+"\n")
                    f.write("Pretraining \n")

            #Pretraining
            model = RamanNet(sp_size=1000,num_classes=30).to(device)
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)
            best_acc = 0
            best_model_name = ""
            best_epoch = -1
            print("Pretraining")
            for epoch in range(epochs):
                print(f"This is Epoch {epoch}")
                train(model,device,train_train_loader,criterion1,criterion2,optimizer)
                acc = test(model,device,train_val_loader,criterion1)

                with open(filename,"a", encoding="utf-8") as f:
                    f.write(f"This is epoch {epoch}\n")
                    f.write(f"The accuracy is {round(acc,2)}\n")
                    f.write("\n")

                if acc>best_acc:
                  #Deleting the previous best model
                    if len(best_model_name)!=0:
                      os.remove(best_model_name)

                    #Saving the current model
                    best_acc = acc
                    best_model_name = f"results/hyperparameter_tuning/Bacteria_ID/models/model_RamanNet_{epoch}_{round(acc,2)}_.pt"
                    torch.save(model.state_dict(),best_model_name)
                    best_epoch = epoch
                    continue

                if epoch-best_epoch>=stopping_epochs:
                  break

            print("Finetuning")
            with open(filename,"a", encoding="utf-8") as f:
                f.write("Finetuning \n")

            #Loading the pretrained model
            pretrained_model = RamanNet(sp_size=1000,num_classes=30).to(device)
            pretrained_model.load_state_dict(torch.load(best_model_name))
            os.remove(best_model_name)

            best_val_acc = 0
            best_test_acc = [0,0,0,0]
            best_epoch = -1
            optimizer = torch.optim.Adam(pretrained_model.parameters(),lr=lr/10)

            for epoch in range(epochs):
                print(f"This is Epoch {epoch}")
                train(pretrained_model,device,fine_train_loader,criterion1,criterion2,optimizer)
                acc = test(pretrained_model,device,fine_val_loader,criterion1)

                with open(filename,"a", encoding="utf-8") as f:
                    f.write(f"This is epoch {epoch}\n")
                    f.write(f"The accuracy is {round(acc,2)}\n")
                    f.write("\n")

                if acc>best_val_acc:
                    best_val_acc = acc
                    if best_val_acc>all_best_val_acc:
                        best_test_acc = test_f1(pretrained_model,device,test_loader,criterion1)
                        all_best_test_acc = copy.deepcopy(best_test_acc)
                        all_best_val_acc = best_val_acc
                        best_hyper=iteration

                        #Deleting the previous best model
                        if len(best_final_model_name)!=0:
                            os.remove(best_final_model_name)

                        #Saving the current model
                        best_final_model_name = f"results/hyperparameter_tuning/trained_models/Bacteria_thirty_RamanNet_{epoch}_{round(acc,2)}_.pt"
                        torch.save(pretrained_model.state_dict(),best_final_model_name)
                    
                    best_epoch = epoch
                    continue

                if epoch-best_epoch>=stopping_epochs:
                  break       

    with open(filename,"a", encoding="utf-8") as f:
        f.write(f"The best test accuracy is {round(all_best_test_acc[0],2)}\n")
        f.write(f"The best test Precision is {round(all_best_test_acc[1],4)}\n")
        f.write(f"The best test Recall is {round(all_best_test_acc[2],4)}\n")
        f.write(f"The best test F1_score is {round(all_best_test_acc[3],4)}\n")
        f.write(best_hyper)


if __name__=="__main__":
    main()