import argparse

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef

import Ensemble
import pandas as pd

def convert_to_ensemble(row):
    individual = []
    for i in range(N_ENCODINGS):
        cls = row['cls{}'.format(i)]
        encoding = row['encoding{}'.format(i)]
        if cls == 'rf':
            individual.append(
                {'cls': cls, 'encoding': encoding, 'params': {'n_estimators': row['RF_n{}'.format(i)]}})
        if cls == 'svm':
            gamma = row['SVM_gamma{}'.format(i)]
            gamma = 'auto' if gamma == 0 else gamma
            individual.append({'cls': cls, 'encoding': encoding,
                               'params': {'probability': True, 'gamma': gamma, 'C': row['SVM_C{}'.format(i)]}})
    return Ensemble(individual)

def final_eval(y_test, predictions):
    scores = []
    scores.append(f1_score(y_test.values, predictions))
    scores.append(accuracy_score(y_test.values, predictions))
    scores.append(precision_score(y_test.values, predictions))
    scores.append(recall_score(y_test.values, predictions))
    scores.append(matthews_corrcoef(y_test.values,predictions))
    return scores



def argparser():
    """
    Command line arguments are specified here.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', action="store", type=str, default='populations', help='File path to load')
    return parser.parse_args()

if __name__=='__main__':
    filename = argparser().__dict__.get('filename')
    df = pd.read_csv(filename)
    top_20_individuals = df[(df.index.levels[0].max(), 0):(df.index.levels[0].max(), len(df.index.levels[1]))]. \
                             sort_values('fitness', ascending=False)[:20]
    N_ENCODINGS = df.columns.map(lambda x: 1 if x.startswith('cls') else 0).values.sum()

    res = pd.DataFrame()
    for i in range(20):
        individual = top_20_individuals.iloc[i]
        e = convert_to_ensemble(individual)
        res = res.append(e.performance_metrics())
    res.to_csv('Results/'+filename+'_performance')