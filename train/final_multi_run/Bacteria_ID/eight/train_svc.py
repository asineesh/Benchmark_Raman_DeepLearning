import argparse
import torch

from torch.utils.data import Dataset, random_split
from torchvision import transforms

import pickle
import random

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets.Bacteria_ID.config import ATCC_GROUPINGS
import os
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

def test_f1(model,test_set):
    X_test = [test_set[i][0].squeeze() for i in range(len(test_set))]
    y_test = [test_set[i][1] for i in range(len(test_set))]

    all_preds = model.predict(X_test)
    all_preds = np.array(all_preds)
    all_targets = np.array(y_test)

    accuracy = 100.0 * (all_preds == all_targets).mean()
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f'Classification accuracy: {round(accuracy,2)}')

    return accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float,help="Regularization coefficient",default=0.1)
    parser.add_argument("--kernel", type=str, help="Kernel used",default="linear")
    parser.add_argument("--runs", type=int, help="The number of experiments",default=5)

    args = parser.parse_args()

    os.makedirs("results/final_multi_run/Bacteria_ID/eight", exist_ok=True)
    os.makedirs("results/final_multi_run/trained_models/", exist_ok=True)

    filename = "results/final_multi_run/Bacteria_ID/eight/results_SVC.txt"

    with open(filename,"a",encoding="utf-8") as f:
        f.write(f"C: {args.C}, Kernel: {args.kernel}, Total number of runs: {args.runs} \n")

    all_test_acc = []
    per_run_time = []
    random.seed(42)

    for run in range(args.runs):
        start_run_time = time.time()
        generator = torch.manual_seed(43+run)
        fine_set = Bacteria_Dataset("datasets/Bacteria_ID/X_finetune.npy","datasets/Bacteria_ID/y_finetune.npy",8)
        test_set = Bacteria_Dataset("datasets/Bacteria_ID/X_test.npy","datasets/Bacteria_ID/y_test.npy",8)

        train_set, val_set = random_split(fine_set,[0.8,0.2],generator=generator)
        X_train = [train_set[i][0].squeeze() for i in range(len(train_set))]
        y_train = [train_set[i][1] for i in range(len(train_set))]

        with open(filename,"a", encoding="utf-8") as f:
            f.write(f"Run {run} \n")

        model = SVC(C=args.C, random_state=43+run, kernel=args.kernel)
        model.fit(X_train,y_train)
        accuracy,_,_,_ = test_f1(model,val_set)
        best_test_acc = test_f1(model,test_set)

        #Saving the current model
        best_final_model_name = f"results/final_multi_run/trained_models/Bacteria_ID_eight_SVC_{run}_{round(accuracy,2)}_.pkl"
        with open(best_final_model_name, 'wb') as file:
            pickle.dump(model, file)

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

    per_run_time = np.array(per_run_time)

    with open(filename,"a", encoding="utf-8") as f:       
        f.write(f"The time for one run is {round(per_run_time.mean(),2)} \u00B1 {round(per_run_time.std(),2)} \n")
        for i in range(len(per_run_time)):
            f.write(f"{per_run_time[i].item()}, ")
        f.write("\n")
        

if __name__=="__main__":
    main()
