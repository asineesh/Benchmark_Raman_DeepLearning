import argparse
import torch

from torch.utils.data import Dataset, random_split
from torchvision import transforms

import pickle
import random

import numpy as np
from sklearn.svm import SVC
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
    parser.add_argument("--C", type=float,help="Regularization coefficient",default=10)
    parser.add_argument("--kernel", type=str, help="Kernel used",default="rbf")
    parser.add_argument("--runs", type=int, help="The number of experiments",default=5)

    args = parser.parse_args()

    os.makedirs("results/final_multi_run/MLROD", exist_ok=True)
    os.makedirs("results/final_multi_run/trained_models/", exist_ok=True)

    filename = "results/final_multi_run/MLROD/results_SVC.txt"

    with open(filename,"a",encoding="utf-8") as f:
        f.write(f"C: {args.C}, Kernel: {args.kernel}, Total number of runs: {args.runs} \n")

    all_test_acc_overall = []
    all_test_acc_granite0 = []
    all_test_acc_granite50 = []
    all_test_acc_gabbro0 = []
    all_test_acc_gabbro50 = []
    per_run_time = []
    random.seed(42)

    for run in range(args.runs):
        start_run_time = time.time()
        generator = torch.manual_seed(43+run)
        all_train_set = MLROD_dataset("datasets/MLROD/MLROD_train.pkl")
        train_set, val_set = random_split(all_train_set,[0.8,0.2],generator=generator)
        test_set = MLROD_dataset("datasets/MLROD/MLROD_test.pkl")

        test_set_granite_0 = MLROD_dataset("datasets/MLROD/MLROD_test_granite_0.pkl")
        test_set_granite_50 = MLROD_dataset("datasets/MLROD/MLROD_test_granite_50.pkl")
        test_set_gabbro_0 = MLROD_dataset("datasets/MLROD/MLROD_test_gabbro_0.pkl")
        test_set_gabbro_50 = MLROD_dataset("datasets/MLROD/MLROD_test_gabbro_50.pkl")

        X_train = [train_set[i][0].squeeze() for i in range(len(train_set))]
        y_train = [train_set[i][1] for i in range(len(train_set))]

        with open(filename,"a", encoding="utf-8") as f:
            f.write(f"Run {run} \n")

        model = SVC(C=args.C, random_state=43+run, kernel=args.kernel)
        model.fit(X_train,y_train)
        accuracy,_,_,_ = test_f1(model,val_set)
        best_test_acc_granite0 = test_f1(model,test_set_granite_0 )
        best_test_acc_granite50= test_f1(model,test_set_granite_50 )
        best_test_acc_gabbro0 = test_f1(model,test_set_gabbro_0 )
        best_test_acc_gabbro50 = test_f1(model,test_set_gabbro_50 )
        best_test_acc_overall = test_f1(model,test_set)

        #Saving the current model
        best_final_model_name = f"results/final_multi_run/trained_models/MLROD_SVC_{run}_{round(accuracy,2)}_.pkl"
        with open(best_final_model_name, 'wb') as file:
            pickle.dump(model, file)

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

    per_run_time = np.array(per_run_time)

    with open(filename,"a", encoding="utf-8") as f:       
        f.write(f"The time for one run is {round(per_run_time.mean(),2)} \u00B1 {round(per_run_time.std(),2)} \n")
        for i in range(len(per_run_time)):
            f.write(f"{per_run_time[i].item()}, ")
        f.write("\n")
        

if __name__=="__main__":
    main()
