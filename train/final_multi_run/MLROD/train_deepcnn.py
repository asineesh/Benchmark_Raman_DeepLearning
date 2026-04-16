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
from models.DeepCNN import DeepCNN
import math
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import copy
import time

class MLROD_dataset(Dataset):
  def __init__(self,path):
    """
    path is a string containing the path to the pkl dataset
    """
    super().__init__()
    self.y, self.X = pickle.load(open(path, 'rb'))
    self.y = list(self.y) #y is a list with containing the name of the chemical corresponding to X
    self.X = list(self.X) #X is a list with each element of the list containing a 1024 time series data

    #To remove the Unknown samples from the dataset
    i = 0
    while i<len(self.y):
      if self.y[i]==15:
        self.y.pop(i)
        self.X.pop(i)

      else:
        i+=1

  def __len__(self):
    return len(self.y)

  def __getitem__(self,index):
    data = torch.Tensor(self.X[index]) #of shape (1,1024)
    data = (data-data.min())/(data.max()-data.min())
    label = self.y[index]
    return data,label

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
    parser.add_argument("--batch_size", type=int,help="The batch size",default=128)
    parser.add_argument("--learning_rate", type=float, help="The learning rate",default=0.001)
    parser.add_argument("--runs", type=int, help="The number of experiments",default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    os.makedirs("results/final_multi_run/MLROD", exist_ok=True)
    os.makedirs("results/final_multi_run/trained_models/", exist_ok=True)

    filename = "results/final_multi_run/MLROD/results_deepcnn.txt"

    with open(filename,"a",encoding="utf-8") as f:
        f.write(f"Batch size: {args.batch_size}, Learning rate: {args.learning_rate}, Total number of runs: {args.runs} \n")

    epochs = 40
    stopping_epochs = 10
    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    lr = args.learning_rate
    
    all_test_acc_overall = []
    all_test_acc_granite0 = []
    all_test_acc_granite50 = []
    all_test_acc_gabbro0 = []
    all_test_acc_gabbro50 = []

    per_epoch_time = []
    per_run_time = []
    random.seed(42)

    for run in range(args.runs):
        start_run_time = time.time()
        generator = torch.manual_seed(43+run)
        train_set = MLROD_dataset("datasets/MLROD/MLROD_train.pkl")
        train_train_set, train_val_set = random_split(train_set,[0.8,0.2],generator=generator)
        test_set = MLROD_dataset("datasets/MLROD/MLROD_test.pkl")

        train_loader = DataLoader(train_train_set, batch_size=batch_size, num_workers=8, shuffle=True)
        val_loader = DataLoader(train_val_set, batch_size=batch_size, num_workers=8, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8,shuffle=False)

        print(f"The number of elements in the train set is {len(train_loader.dataset)}")
        print(f"The number of elements in the val set is {len(val_loader.dataset)}")
        print(f"The number of elements in the test set is {len(test_loader.dataset)}")

        test_set_granite_0 = MLROD_dataset("datasets/MLROD/MLROD_test_granite_0.pkl")
        test_set_granite_50 = MLROD_dataset("datasets/MLROD/MLROD_test_granite_50.pkl")
        test_set_gabbro_0 = MLROD_dataset("datasets/MLROD/MLROD_test_gabbro_0.pkl")
        test_set_gabbro_50 = MLROD_dataset("datasets/MLROD/MLROD_test_gabbro_50.pkl")

        test_loader_granite_0 = DataLoader(test_set_granite_0, batch_size=batch_size, num_workers=8,shuffle=False)
        test_loader_granite_50 = DataLoader(test_set_granite_50, batch_size=batch_size, num_workers=8,shuffle=False)
        test_loader_gabbro_0 = DataLoader(test_set_gabbro_0, batch_size=batch_size, num_workers=8,shuffle=False)
        test_loader_gabbro_50 = DataLoader(test_set_gabbro_50, batch_size=batch_size, num_workers=8,shuffle=False)

        print(f"The number of elements in the test set is {len(test_loader_granite_0.dataset)}")
        print(f"The number of elements in the test set is {len(test_loader_granite_50.dataset)}")
        print(f"The number of elements in the test set is {len(test_loader_gabbro_0.dataset)}")
        print(f"The number of elements in the test set is {len(test_loader_gabbro_50.dataset)}")

        with open(filename,"a", encoding="utf-8") as f:
            f.write(f"Run {run} \n")

        model = DeepCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)

        best_val_acc = 0

        best_test_acc_overall = [0,0,0,0]
        best_test_acc_granite0 = [0,0,0,0]
        best_test_acc_granite50= [0,0,0,0]
        best_test_acc_gabbro0 = [0,0,0,0]
        best_test_acc_gabbro50  = [0,0,0,0]

        best_epoch = -1
        best_final_model_name = ""

        for epoch in range(epochs):
            print(f"This is Epoch {epoch}")
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_epoch_time = time.time()

            train(model,device,train_loader,criterion,optimizer)

            if device.type == "cuda":
                torch.cuda.synchronize()
            per_epoch_time.append(time.time()-start_epoch_time)
            
            acc = test(model,device,val_loader,criterion)

            with open(filename,"a", encoding="utf-8") as f:
                f.write(f"This is epoch {epoch}\n")
                f.write(f"The accuracy is {round(acc,2)}\n")
                f.write("\n")

            if acc>best_val_acc:
                best_val_acc = acc
                best_test_acc_granite0 = test_f1(model,device,test_loader_granite_0 ,criterion)
                best_test_acc_granite50= test_f1(model,device,test_loader_granite_50 ,criterion)
                best_test_acc_gabbro0 = test_f1(model,device,test_loader_gabbro_0 ,criterion)
                best_test_acc_gabbro50 = test_f1(model,device,test_loader_gabbro_50 ,criterion)
                best_test_acc_overall = test_f1(model,device,test_loader,criterion)

                #Deleting the previous best model
                if len(best_final_model_name)!=0:
                    os.remove(best_final_model_name)

                #Saving the current model
                best_final_model_name = f"results/final_multi_run/trained_models/MLROD_deepcnn_{run}_{epoch}_{round(acc,2)}_.pt"
                torch.save(model.state_dict(),best_final_model_name)
            
                best_epoch = epoch
                continue

            if epoch-best_epoch>=stopping_epochs:
                break  
        
        with open(filename,"a", encoding="utf-8") as f:
            f.write("For Granite 0 \n")
            f.write(f"The best test accuracy is {round(best_test_acc_granite0[0],2)}\n")
            f.write(f"The best test Precision is {round(best_test_acc_granite0[1],4)}\n")
            f.write(f"The best test Recall is {round(best_test_acc_granite0[2],4)}\n")
            f.write(f"The best test F1_score is {round(best_test_acc_granite0[3],4)}\n \n")

            f.write("For Granite 50 \n")
            f.write(f"The best test accuracy is {round(best_test_acc_granite50[0],2)}\n")
            f.write(f"The best test Precision is {round(best_test_acc_granite50[1],4)}\n")
            f.write(f"The best test Recall is {round(best_test_acc_granite50[2],4)}\n")
            f.write(f"The best test F1_score is {round(best_test_acc_granite50[3],4)}\n \n")

            f.write("For Gabbro 0 \n")
            f.write(f"The best test accuracy is {round(best_test_acc_gabbro0[0],2)}\n")
            f.write(f"The best test Precision is {round(best_test_acc_gabbro0[1],4)}\n")
            f.write(f"The best test Recall is {round(best_test_acc_gabbro0[2],4)}\n")
            f.write(f"The best test F1_score is {round(best_test_acc_gabbro0[3],4)}\n \n")

            f.write("For Gabbro 50 \n")
            f.write(f"The best test accuracy is {round(best_test_acc_gabbro50[0],2)}\n")
            f.write(f"The best test Precision is {round(best_test_acc_gabbro50[1],4)}\n")
            f.write(f"The best test Recall is {round(best_test_acc_gabbro50[2],4)}\n")
            f.write(f"The best test F1_score is {round(best_test_acc_gabbro50[3],4)}\n \n")

            f.write("For Overall \n")
            f.write(f"The best test accuracy is {round(best_test_acc_overall[0],2)}\n")
            f.write(f"The best test Precision is {round(best_test_acc_overall[1],4)}\n")
            f.write(f"The best test Recall is {round(best_test_acc_overall[2],4)}\n")
            f.write(f"The best test F1_score is {round(best_test_acc_overall[3],4)}\n \n")


        all_test_acc_overall.append(copy.deepcopy(best_test_acc_overall))  
        all_test_acc_granite0.append(copy.deepcopy(best_test_acc_granite0)) 
        all_test_acc_granite50.append(copy.deepcopy(best_test_acc_granite50)) 
        all_test_acc_gabbro0.append(copy.deepcopy(best_test_acc_gabbro0)) 
        all_test_acc_gabbro50.append(copy.deepcopy(best_test_acc_gabbro50)) 

        per_run_time.append(time.time()-start_run_time)

    all_acc_overall = np.array([x[0] for x in all_test_acc_overall])
    all_pr_overall = np.array([x[1] for x in all_test_acc_overall])
    all_re_overall = np.array([x[2] for x in all_test_acc_overall])
    all_f1_overall = np.array([x[3] for x in all_test_acc_overall])

    all_acc_granite0 = np.array([x[0] for x in all_test_acc_granite0])
    all_pr_granite0 = np.array([x[1] for x in all_test_acc_granite0])
    all_re_granite0 = np.array([x[2] for x in all_test_acc_granite0])
    all_f1_granite0 = np.array([x[3] for x in all_test_acc_granite0])

    all_acc_granite50 = np.array([x[0] for x in all_test_acc_granite50])
    all_pr_granite50 = np.array([x[1] for x in all_test_acc_granite50])
    all_re_granite50 = np.array([x[2] for x in all_test_acc_granite50])
    all_f1_granite50 = np.array([x[3] for x in all_test_acc_granite50])

    all_acc_gabbro0 = np.array([x[0] for x in all_test_acc_gabbro0])
    all_pr_gabbro0 = np.array([x[1] for x in all_test_acc_gabbro0])
    all_re_gabbro0 = np.array([x[2] for x in all_test_acc_gabbro0])
    all_f1_gabbro0 = np.array([x[3] for x in all_test_acc_gabbro0])

    all_acc_gabbro50 = np.array([x[0] for x in all_test_acc_gabbro50])
    all_pr_gabbro50 = np.array([x[1] for x in all_test_acc_gabbro50])
    all_re_gabbro50 = np.array([x[2] for x in all_test_acc_gabbro50])
    all_f1_gabbro50 = np.array([x[3] for x in all_test_acc_gabbro50])

    with open(filename,"a", encoding="utf-8") as f:
        f.write("For Overall \n")
        f.write(f"The test accuracy is {round(all_acc_overall.mean(),2)} \u00B1 {round(all_acc_overall.std(),2)} \n")
        f.write(f"The test Precision {round(all_pr_overall.mean(),4)} \u00B1 {round(all_pr_overall.std(),4)} \n")
        f.write(f"The test Recall is {round(all_re_overall.mean(),4)} \u00B1 {round(all_re_overall.std(),4)} \n")
        f.write(f"The test F1_score is {round(all_f1_overall.mean(),4)} \u00B1 {round(all_f1_overall.std(),4)} \n \n")

        f.write("For Granite 0 \n")
        f.write(f"The test accuracy is {round(all_acc_granite0.mean(),2)} \u00B1 {round(all_acc_granite0.std(),2)} \n")
        f.write(f"The test Precision {round(all_pr_granite0.mean(),4)} \u00B1 {round(all_pr_granite0.std(),4)} \n")
        f.write(f"The test Recall is {round(all_re_granite0.mean(),4)} \u00B1 {round(all_re_granite0.std(),4)} \n")
        f.write(f"The test F1_score is {round(all_f1_granite0.mean(),4)} \u00B1 {round(all_f1_granite0.std(),4)} \n \n")

        f.write("For Granite 50 \n")
        f.write(f"The test accuracy is {round(all_acc_granite50.mean(),2)} \u00B1 {round(all_acc_granite50.std(),2)} \n")
        f.write(f"The test Precision {round(all_pr_granite50.mean(),4)} \u00B1 {round(all_pr_granite50.std(),4)} \n")
        f.write(f"The test Recall is {round(all_re_granite50.mean(),4)} \u00B1 {round(all_re_granite50.std(),4)} \n")
        f.write(f"The test F1_score is {round(all_f1_granite50.mean(),4)} \u00B1 {round(all_f1_granite50.std(),4)} \n \n")

        f.write("For Gabbro 0 \n")
        f.write(f"The test accuracy is {round(all_acc_gabbro0.mean(),2)} \u00B1 {round(all_acc_gabbro0.std(),2)} \n")
        f.write(f"The test Precision {round(all_pr_gabbro0.mean(),4)} \u00B1 {round(all_pr_gabbro0.std(),4)} \n")
        f.write(f"The test Recall is {round(all_re_gabbro0.mean(),4)} \u00B1 {round(all_re_gabbro0.std(),4)} \n")
        f.write(f"The test F1_score is {round(all_f1_gabbro0.mean(),4)} \u00B1 {round(all_f1_gabbro0.std(),4)} \n \n")

        f.write("For Gabbro 50 \n")
        f.write(f"The test accuracy is {round(all_acc_gabbro50.mean(),2)} \u00B1 {round(all_acc_gabbro50.std(),2)} \n")
        f.write(f"The test Precision {round(all_pr_gabbro50.mean(),4)} \u00B1 {round(all_pr_gabbro50.std(),4)} \n")
        f.write(f"The test Recall is {round(all_re_gabbro50.mean(),4)} \u00B1 {round(all_re_gabbro50.std(),4)} \n")
        f.write(f"The test F1_score is {round(all_f1_gabbro50.mean(),4)} \u00B1 {round(all_f1_gabbro50.std(),4)} \n \n")

    
    per_epoch_time = np.array(per_epoch_time)
    per_run_time = np.array(per_run_time)

    with open(filename,"a", encoding="utf-8") as f:
        f.write(f"The time for one epoch is {round(per_epoch_time.mean(),2)} \u00B1 {round(per_epoch_time.std(),2)} \n")
        for i in range(len(per_epoch_time)):
            f.write(f"{per_epoch_time[i].item()}, ")
        
        f.write(f"\nThe time for one run is {round(per_run_time.mean(),2)} \u00B1 {round(per_run_time.std(),2)} \n")
        for i in range(len(per_run_time)):
            f.write(f"{per_run_time[i].item()}, ")
        f.write("\n")
        

if __name__=="__main__":
    main()
