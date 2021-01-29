import pandas as pd
from joblib import load
import os
import PerformanceMetricsEvaluation as PME
import numpy as np
import csv
import re
import random as rd

from sklearn.model_selection import KFold

def indices_for_ncv(outer_cv,data_pth):
    y = pd.read_csv(data_pth).y
    n = len(y)
    all = range(n)
    folds = [rd.sample(all,round((1/outer_cv)*n))]
    all = [x for x in all if x not in folds]

    for i in range(1,outer_cv):
        new = rd.sample([x for x in all if x not in folds[:i]],round((1/outer_cv)*n))
        folds.append(new)
        all = [x for x in all if x not in new]
    return folds

def build_data_sets(ncv, indices_path):
    indice_groups = []
    all_indices = []
    for i in range(ncv):
        indices_fold = []
        for j in range(ncv):
            if not i == j:
                indices_fold.append(list(pd.read_csv(indices_path + 'fold' + str(i+1) + '.csv').iloc[:,0]))
        indice_groups.append(indices_fold)
        all_indices.append(list(pd.read_csv(indices_path + 'fold' + str(i+1) + '.csv').iloc[:,0]))
    all_indices = [item for one_fold_indices in all_indices for item in one_fold_indices]
    return indice_groups, all_indices

def final_ncv_eval(ncv, base_path, models_path, indices_path, all_final_encos, save_path):
    print("Starting eval")
    save_path = base_path + save_path
    models_path = base_path+models_path
    cv = 3
    data_indis, all_indis = build_data_sets(ncv,indices_path)
    print(models_path)
    if "amp" in indices_path:
        data_typ = "amp/"
    elif "imo" in indices_path:
        data_typ = "imo/"
    fold_scores = []

    for i in range(ncv):
        print("fold"+str(i+1))

        fold_encos = all_final_encos[i]
        clf_names = [x for x in os.listdir(models_path) if (('fold' + str(i+1)) in x)]

        times = [re.findall("1\d+\.\d+",clf_names[i])[0] for i in range(len(clf_names))]
        clf_names = [x[0] for x in sorted(zip(clf_names,times),key=lambda x: x[1] )]
        kf = KFold(cv,shuffle=True)
        try:
            y = pd.read_csv('data/'+data_typ+'structure_based/'+fold_encos[0]).y
        except FileNotFoundError:
            y = pd.read_csv('data/' + data_typ + 'sequence_based/' + fold_encos[0]).y
        training_indices = [item for one_fold_indices in data_indis[i] for item in one_fold_indices]
        testing_indis = [x for x in all_indis if x not in training_indices]
        predictions = []
        scores = []

        for j in range(cv):
            y_test = y.iloc[testing_indis]
            for k in range(len(clf_names)):
                try:
                    clf = load(models_path+clf_names[0])
                    for encoding in fold_encos:
                        try:
                            df = pd.read_csv('data/'+data_typ+'structure_based/'+encoding)
                        except FileNotFoundError:
                            df = pd.read_csv('data/' + data_typ + 'sequence_based/' + encoding)
                        #X_train = df.iloc[training_indices].iloc[split_idx[0], 1:-1]
                        #y_train = y.iloc[split_idx[0]]
                        X_test = df.iloc[testing_indis, 1:-1]
                        #clf.fit(X_train,y_train)
                        individual_predictions = clf.predict_proba(X_test)
                        predictions.append([x[1] for x in individual_predictions])
                        print(k)
                except ValueError:
                    ignore = True
                break
            p = [0 if x < 0.5 else 1 for x in np.mean(predictions, axis=0)]           # Fusion method: mean
            scores.append(PME.final_eval(y_test, p))
        fold_scores.append(scores)

    mean_score = []
    fold_scores = [item for one_fold_score in fold_scores for item in one_fold_score]
    print(fold_scores)
    for j in range(len(fold_scores[0])):
        mean_score.append(0)
        for i in range(len(fold_scores)):
            mean_score[j] = mean_score[j] + fold_scores[i][j]
        mean_score[j]=mean_score[j]/len(fold_scores)
    print(mean_score)
    if os.path.exists(save_path):
        file_exists = True
    else:
        file_exists = False
    with open(save_path, 'a') as fd:
        fields = mean_score +np.std(mean_score[0])
        writer = csv.writer(fd)
        if not file_exists:
            writer.writerow(["F1", "Accuracy", "Precision", "Recall", "MCC","F1 Std"])
        writer.writerow(fields)
    exit()