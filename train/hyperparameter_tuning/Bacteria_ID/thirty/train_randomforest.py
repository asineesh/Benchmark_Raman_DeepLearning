import torch

from torch.utils.data import Dataset, random_split, ConcatDataset
from torchvision import transforms

import pickle
import random

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from sklearn.ensemble import RandomForestClassifier
from datasets.Bacteria_ID.config import ATCC_GROUPINGS

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
    filename = "results/hyperparameter_tuning/Bacteria_ID/thirty/results_RandomForest.txt"
    os.makedirs("results/hyperparameter_tuning/Bacteria_ID/thirty", exist_ok=True)
    os.makedirs("results/hyperparameter_tuning/trained_models/", exist_ok=True)

    n_estimators = [100, 300, 500] #Number of trees in the forest
    max_depths = [None, 10, 20] #Maximum depth of the tree
    combinations = ["fine","ref","ref and fine"]

    generator = torch.manual_seed(42)
    random.seed(42)

    ref_set = Bacteria_Dataset("datasets/Bacteria_ID/X_reference.npy","datasets/Bacteria_ID/y_reference.npy",30)
    fine_set = Bacteria_Dataset("datasets/Bacteria_ID/X_finetune.npy","datasets/Bacteria_ID/y_finetune.npy",30)
    test_set = Bacteria_Dataset("datasets/Bacteria_ID/X_test.npy","datasets/Bacteria_ID/y_test.npy",30)
    ref_train_set, ref_val_set = random_split(ref_set,[0.8,0.2],generator=generator)
    fine_train_set, fine_val_set = random_split(fine_set,[0.8,0.2],generator=generator)

    best_accuracy = 0
    best_hyper = ""
    best_final_model_name = ""
    all_best_test_acc = []

    for combination in combinations:
        if combination == "fine":
            X_train = [fine_train_set[i][0].squeeze() for i in range(len(fine_train_set))]
            y_train = [fine_train_set[i][1] for i in range(len(fine_train_set))]


        elif combination == "ref":
            X_train = [ref_train_set[i][0].squeeze() for i in range(len(ref_train_set))]
            y_train = [ref_train_set[i][1] for i in range(len(ref_train_set))]
        
        else:
            X_train = [fine_train_set[i][0].squeeze() for i in range(len(fine_train_set))]
            X_train.extend([ref_train_set[i][0].squeeze() for i in range(len(ref_train_set))])

            y_train = [fine_train_set[i][1] for i in range(len(fine_train_set))]
            y_train.extend([ref_train_set[i][1] for i in range(len(ref_train_set))])

        val_set = fine_val_set

        for n_estimator in n_estimators:
            for max_depth in max_depths:
                iteration = f"For Combination {combination}, Number of estimators {n_estimator} and Max depth {max_depth}"
                with open(filename,"a", encoding="utf-8") as f:
                    f.write("\n"+iteration+"\n")

                model = RandomForestClassifier(max_depth=max_depth, random_state=42, n_estimators=n_estimator)
                model.fit(X_train,y_train)

                accuracy, _, _, _ = test_f1(model,val_set)
                with open(filename,"a", encoding="utf-8") as f:
                    f.write(f"The validation accuracy is {round(accuracy,2)}\n")

                if accuracy>best_accuracy:
                    best_accuracy = accuracy
                    all_best_test_acc = test_f1(model,test_set)
                    best_hyper = iteration

                    #Deleting the previous best model
                    if len(best_final_model_name)!=0:
                        os.remove(best_final_model_name)

                    #Saving the current model
                    best_final_model_name = f"results/hyperparameter_tuning/trained_models/Bacteria_thirty_RandomForest_{round(accuracy,2)}_.pkl"
                    with open(best_final_model_name, 'wb') as file:
                        pickle.dump(model, file)
    
    with open(filename,"a", encoding="utf-8") as f:
        f.write(f"\nThe best test accuracy is {round(all_best_test_acc[0],2)}\n")
        f.write(f"The best test Precision is {round(all_best_test_acc[1],4)}\n")
        f.write(f"The best test Recall is {round(all_best_test_acc[2],4)}\n")
        f.write(f"The best test F1_score is {round(all_best_test_acc[3],4)}\n")
        f.write(f"The best hyperparameters are {best_hyper}")


if __name__ == "__main__":
    main()








