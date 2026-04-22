import torch

from torch.utils.data import Dataset, random_split
from torchvision import transforms

import pickle
import random

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from sklearn.svm import SVC

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
    filename = "results/hyperparameter_tuning/MLROD/results_SVC.txt"
    os.makedirs("results/hyperparameter_tuning/MLROD", exist_ok=True)
    os.makedirs("results/hyperparameter_tuning/trained_models/", exist_ok=True)

    Cs = [0.1,1,10] 
    kernels = ['linear','rbf','sigmoid'] 

    generator = torch.manual_seed(42)
    random.seed(42)

    all_train_set = MLROD_dataset("datasets/MLROD/MLROD_train.pkl")
    test_set = MLROD_dataset("datasets/MLROD/MLROD_test.pkl")
    train_set, val_set = random_split(all_train_set,[0.8,0.2],generator=generator)

    test_set_granite_0 = MLROD_dataset("datasets/MLROD/MLROD_test_granite_0.pkl")
    test_set_granite_50 = MLROD_dataset("datasets/MLROD/MLROD_test_granite_50.pkl")
    test_set_gabbro_0 = MLROD_dataset("datasets/MLROD/MLROD_test_gabbro_0.pkl")
    test_set_gabbro_50 = MLROD_dataset("datasets/MLROD/MLROD_test_gabbro_50.pkl")

    X_train = [train_set[i][0].squeeze() for i in range(len(train_set))]
    y_train = [train_set[i][1] for i in range(len(train_set))]
    best_accuracy = 0
    best_hyper = ""
    best_final_model_name = ""
    all_best_test_acc = [[],[],[],[],[]]

    for C in Cs:
        for kernel in kernels:
            iteration = f"For C {C} and kernel {kernel}"
            with open(filename,"a", encoding="utf-8") as f:
                f.write("\n"+iteration+"\n")

            model = SVC(C=C, random_state=42, kernel=kernel)
            model.fit(X_train,y_train)

            accuracy, _, _, _ = test_f1(model,val_set)
            with open(filename,"a", encoding="utf-8") as f:
                f.write(f"The validation accuracy is {round(accuracy,2)}\n")

            if accuracy>best_accuracy:
                best_accuracy = accuracy
                all_best_test_acc[0] = test_f1(model,test_set_granite_0)
                all_best_test_acc[1] = test_f1(model,test_set_granite_50)
                all_best_test_acc[2] = test_f1(model,test_set_gabbro_0)
                all_best_test_acc[3] = test_f1(model,test_set_gabbro_50)
                all_best_test_acc[4] = test_f1(model,test_set)
                best_hyper = iteration

                #Deleting the previous best model
                if len(best_final_model_name)!=0:
                    os.remove(best_final_model_name)

                #Saving the current model
                best_final_model_name = f"results/hyperparameter_tuning/trained_models/MLROD_SVC_{round(accuracy,2)}_.pkl"
                with open(best_final_model_name, 'wb') as file:
                    pickle.dump(model, file)
    
    with open(filename,"a", encoding="utf-8") as f:
        f.write("For Granite 0\n")
        f.write(f"The best test accuracy is {round(all_best_test_acc[0][0],2)}\n")
        f.write(f"The best test Precision is {round(all_best_test_acc[0][1],4)}\n")
        f.write(f"The best test Recall is {round(all_best_test_acc[0][2],4)}\n")
        f.write(f"The best test F1_score is {round(all_best_test_acc[0][3],4)}\n \n")

        f.write("For Granite 50\n")
        f.write(f"The best test accuracy is {round(all_best_test_acc[1][0],2)}\n")
        f.write(f"The best test Precision is {round(all_best_test_acc[1][1],4)}\n")
        f.write(f"The best test Recall is {round(all_best_test_acc[1][2],4)}\n")
        f.write(f"The best test F1_score is {round(all_best_test_acc[1][3],4)}\n \n")

        f.write("For Gabbro 0\n")
        f.write(f"The best test accuracy is {round(all_best_test_acc[2][0],2)}\n")
        f.write(f"The best test Precision is {round(all_best_test_acc[2][1],4)}\n")
        f.write(f"The best test Recall is {round(all_best_test_acc[2][2],4)}\n")
        f.write(f"The best test F1_score is {round(all_best_test_acc[2][3],4)}\n \n")

        f.write("For Gabbro 50\n")
        f.write(f"The best test accuracy is {round(all_best_test_acc[3][0],2)}\n")
        f.write(f"The best test Precision is {round(all_best_test_acc[3][1],4)}\n")
        f.write(f"The best test Recall is {round(all_best_test_acc[3][2],4)}\n")
        f.write(f"The best test F1_score is {round(all_best_test_acc[3][3],4)}\n \n")

        f.write("For Overall\n")
        f.write(f"The best test accuracy is {round(all_best_test_acc[4][0],2)}\n")
        f.write(f"The best test Precision is {round(all_best_test_acc[4][1],4)}\n")
        f.write(f"The best test Recall is {round(all_best_test_acc[4][2],4)}\n")
        f.write(f"The best test F1_score is {round(all_best_test_acc[4][3],4)}\n \n")


        f.write(best_hyper)


if __name__ == "__main__":
    main()








