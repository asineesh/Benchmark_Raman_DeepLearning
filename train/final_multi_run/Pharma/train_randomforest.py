import argparse
import torch

from torch.utils.data import Dataset, random_split
from torchvision import transforms

import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import copy
import time

class Pharma_dataset(Dataset):
  def __init__(self,path):
    """
    path is a string containing the path to the pkl dataset
    """
    super().__init__()   
    #X is a list with each element of the list containing a 1024 time series data 
    #y is a list with containing the name of the chemical corresponding to X
    self.y, self.X = pickle.load(open(path, 'rb'))
    names = sorted(list(set(self.y)))
    self.mapping = {names[i]:i for i in range(len(names))}   #Maps each material name to a number

  def __len__(self):
    return len(self.y)

  def __getitem__(self,index):
    data = torch.Tensor(self.X[index]) #of shape (1,1024)
    data = (data-data.min())/(data.max()-data.min())
    label = self.mapping[self.y[index]]
    return data,label

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
    parser.add_argument("--n_estimator", type=int,help="The number of estimators",default=100)
    parser.add_argument("--max_depth", type=int, help="The maximum depth of the tree",default=None)
    parser.add_argument("--runs", type=int, help="The number of experiments",default=5)

    args = parser.parse_args()

    os.makedirs("results/final_multi_run/Pharma", exist_ok=True)
    os.makedirs("results/final_multi_run/trained_models/", exist_ok=True)

    filename = "results/final_multi_run/Pharma/results_RandomForest.txt"

    with open(filename,"a",encoding="utf-8") as f:
        f.write(f"Number of estimators: {args.n_estimator}, Max Depth: {args.max_depth}, Total number of runs: {args.runs} \n")

    all_test_acc = []
    per_run_time = []
    random.seed(42)

    for run in range(args.runs):
        start_run_time = time.time()
        generator = torch.manual_seed(43+run)
        all_train_set = Pharma_dataset("datasets/Pharma/Pharma_train.pkl")
        test_set = Pharma_dataset("datasets/Pharma/Pharma_test.pkl")

        train_set, val_set = random_split(all_train_set,[0.8,0.2],generator=generator)
        X_train = [train_set[i][0].squeeze() for i in range(len(train_set))]
        y_train = [train_set[i][1] for i in range(len(train_set))]

        with open(filename,"a", encoding="utf-8") as f:
            f.write(f"Run {run} \n")

        model = RandomForestClassifier(max_depth=args.max_depth, random_state=43+run, n_estimators=args.n_estimator)
        model.fit(X_train,y_train)
        accuracy,_,_,_ = test_f1(model,val_set)
        best_test_acc = test_f1(model,test_set)

        #Saving the current model
        best_final_model_name = f"results/final_multi_run/trained_models/Pharma_RandomForest_{run}_{round(accuracy,2)}_.pkl"
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
