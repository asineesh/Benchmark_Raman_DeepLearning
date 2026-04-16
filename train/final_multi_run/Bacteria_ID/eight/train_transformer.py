import argparse
import torch
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader, random_split
from torchvision import transforms

from tqdm import tqdm
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from models.transformer import ViT
import math
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from datasets.Bacteria_ID.config import ATCC_GROUPINGS
import copy
import time

class Bacteria_Dataset(Dataset):
    def __init__(self,X_path,y_path,num_classes=30):
        """
        X_path is a string containing the path to the spectra
        y_path is a string containing the path to the labels for the spectra
        num_classes is an integer corresponding to an 8 class or 30 class problem 
        """
        super().__init__()
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        data = torch.Tensor(self.X[index]).unsqueeze(0) #of shape (1,1000)
        data = (data-data.min())/(data.max()-data.min())
        label = self.y[index]
        if self.num_classes==8:
            label = ATCC_GROUPINGS[label]
        return data,int(label)

def train(model,device,train_dataloader,criterion,optimizer):
    model.train()
    loop = tqdm(train_dataloader)
    total_loss = 0
    correct = 0

    for i,(X,y) in enumerate(loop):
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X) #of shape (b,num_classes)
        loss = criterion(y_pred,y)
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

            y_pred = model(X) #of shape (b,num_classes)
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

            y_pred = model(X)#of shape (b,num_classes)                  
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int,help="The batch size",default=32)
    parser.add_argument("--learning_rate", type=float, help="The learning rate",default=1e-05)
    parser.add_argument("--runs", type=int, help="The number of experiments",default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    os.makedirs("results/final_multi_run/Bacteria_ID/eight", exist_ok=True)
    os.makedirs("results/final_multi_run/trained_models/", exist_ok=True)

    filename = "results/final_multi_run/Bacteria_ID/eight/results_transformer.txt"

    with open(filename,"a",encoding="utf-8") as f:
        f.write(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Total number of runs: {args.runs} \n")

    epochs = 40
    stopping_epochs = 10
    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    lr = args.learning_rate
    all_test_acc = []
    per_epoch_time_pre = []
    per_epoch_time_fine = []
    per_run_time = []
    random.seed(42)

    for run in range(args.runs):
        start_run_time = time.time()
        generator = torch.manual_seed(43+run)
        train_set = Bacteria_Dataset("datasets/Bacteria_ID/X_reference.npy","datasets/Bacteria_ID/y_reference.npy",8)
        fine_set = Bacteria_Dataset("datasets/Bacteria_ID/X_finetune.npy","datasets/Bacteria_ID/y_finetune.npy",8)
        test_set = Bacteria_Dataset("datasets/Bacteria_ID/X_test.npy","datasets/Bacteria_ID/y_test.npy",8)

        train_train_set, train_val_set = random_split(train_set,[0.8,0.2],generator=generator)
        fine_train_set, fine_val_set = random_split(fine_set,[0.8,0.2],generator=generator)

        train_train_loader = DataLoader(train_train_set, batch_size=batch_size, num_workers=8, shuffle=True)
        train_val_loader = DataLoader(train_val_set, batch_size=batch_size, num_workers=8, shuffle=True)
        fine_train_loader = DataLoader(fine_train_set, batch_size=batch_size, num_workers=8, shuffle=True)
        fine_val_loader = DataLoader(fine_val_set, batch_size=batch_size, num_workers=8, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8,shuffle=False)

        with open(filename,"a", encoding="utf-8") as f:
            f.write(f"Run {run} \n")
            f.write("Pretraining \n")

        #Pretraining
        model = ViT(patch_size=125,sp_size=1000,num_classes=8,p=0.1).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)

        best_val_acc = 0
        best_epoch = -1
        best_final_model_name = ""
        print("Pretraining")
        for epoch in range(epochs):
            print(f"This is Epoch {epoch}")
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_epoch_time = time.time()

            train(model,device,train_train_loader,criterion,optimizer)

            if device.type == "cuda":
                torch.cuda.synchronize()
            per_epoch_time_pre.append(time.time()-start_epoch_time)
            
            acc = test(model,device,train_val_loader,criterion)

            with open(filename,"a", encoding="utf-8") as f:
                f.write(f"This is epoch {epoch}\n")
                f.write(f"The accuracy is {round(acc,2)}\n")
                f.write("\n")

            if acc>best_val_acc:
                best_val_acc = acc

                #Deleting the previous best model
                if len(best_final_model_name)!=0:
                    os.remove(best_final_model_name)

                #Saving the current model
                best_final_model_name = f"results/final_multi_run/trained_models/Bacteria_ID_eight_transformer_pre_{run}_{epoch}_{round(acc,2)}_.pt"
                torch.save(model.state_dict(),best_final_model_name)
            
                best_epoch = epoch
                continue

            if epoch-best_epoch>=stopping_epochs:
                break  
        
        print("Finetuning")
        with open(filename,"a", encoding="utf-8") as f:
            f.write("Finetuning \n")

        #Loading the pretrained model
        pretrained_model = ViT(patch_size=125,sp_size=1000,num_classes=8,p=0.1).to(device)
        pretrained_model.load_state_dict(torch.load(best_final_model_name))

        best_val_acc = 0
        best_test_acc = [0,0,0,0]
        best_epoch = -1
        best_final_model_name = ""
        optimizer = torch.optim.Adam(pretrained_model.parameters(),lr=lr/10)

        for epoch in range(epochs):
            print(f"This is Epoch {epoch}")
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_epoch_time = time.time()

            train(pretrained_model,device,fine_train_loader,criterion,optimizer)

            if device.type == "cuda":
                torch.cuda.synchronize()
            per_epoch_time_fine.append(time.time()-start_epoch_time)
            
            acc = test(pretrained_model,device,fine_val_loader,criterion)

            with open(filename,"a", encoding="utf-8") as f:
                f.write(f"This is epoch {epoch}\n")
                f.write(f"The accuracy is {round(acc,2)}\n")
                f.write("\n")

            if acc>best_val_acc:
                best_val_acc = acc
                best_test_acc = test_f1(pretrained_model,device,test_loader,criterion)

                #Deleting the previous best model
                if len(best_final_model_name)!=0:
                    os.remove(best_final_model_name)

                #Saving the current model
                best_final_model_name = f"results/final_multi_run/trained_models/Bacteria_ID_eight_transformer_fine_{run}_{epoch}_{round(acc,2)}_.pt"
                torch.save(pretrained_model.state_dict(),best_final_model_name)
            
                best_epoch = epoch
                continue

            if epoch-best_epoch>=stopping_epochs:
                break

        
        with open(filename,"a", encoding="utf-8") as f:
            f.write(f"The best test accuracy is {round(best_test_acc[0],2)}\n")
            f.write(f"The best test Precision is {round(best_test_acc[1],4)}\n")
            f.write(f"The best test Recall is {round(best_test_acc[2],4)}\n")
            f.write(f"The best test F1_score is {round(best_test_acc[3],4)}\n \n")

        all_test_acc.append(copy.deepcopy(best_test_acc))  

        per_run_time.append(time.time()-start_run_time)

    all_acc = np.array([x[0] for x in all_test_acc])
    all_pr = np.array([x[1] for x in all_test_acc])
    all_re = np.array([x[2] for x in all_test_acc])
    all_f1 = np.array([x[3] for x in all_test_acc])

    with open(filename,"a", encoding="utf-8") as f:
        f.write(f"The test accuracy is {round(all_acc.mean(),2)} \u00B1 {round(all_acc.std(),2)} \n")
        f.write(f"The test Precision {round(all_pr.mean(),4)} \u00B1 {round(all_pr.std(),4)} \n")
        f.write(f"The test Recall is {round(all_re.mean(),4)} \u00B1 {round(all_re.std(),4)} \n")
        f.write(f"The test F1_score is {round(all_f1.mean(),4)} \u00B1 {round(all_f1.std(),4)} \n \n")

    
    per_epoch_time_fine = np.array(per_epoch_time_fine)
    per_epoch_time_pre = np.array(per_epoch_time_pre)
    per_run_time = np.array(per_run_time)

    with open(filename,"a", encoding="utf-8") as f:
        f.write(f"The time for one epoch during Pretraining is {round(per_epoch_time_pre.mean(),2)} \u00B1 {round(per_epoch_time_pre.std(),2)} \n")
        for i in range(len(per_epoch_time_pre)):
            f.write(f"{per_epoch_time_pre[i].item()}, ")

        f.write(f"\nThe time for one epoch during Finetuning is {round(per_epoch_time_fine.mean(),2)} \u00B1 {round(per_epoch_time_fine.std(),2)} \n")
        for i in range(len(per_epoch_time_fine)):
            f.write(f"{per_epoch_time_fine[i].item()}, ")
        
        f.write(f"\nThe time for one run is {round(per_run_time.mean(),2)} \u00B1 {round(per_run_time.std(),2)} \n")
        for i in range(len(per_run_time)):
            f.write(f"{per_run_time[i].item()}, ")
        f.write("\n")
        

if __name__=="__main__":
    main()
