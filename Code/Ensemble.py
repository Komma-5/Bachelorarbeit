

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from mlxtend.classifier import StackingCVClassifier, StackingClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pandas as pd
import numpy as np
import DisagreementMeasurement as DisMeas
import PerformanceMetricsEvaluation as PME
import random
import re



class Ensemble:
    KNOWN_CLASSIFIERS = {'rf':RandomForestClassifier, 'svm':SVC}

    def __init__(self, classifiers, div_frac, pre_div, struc_seq_pairs, train_fit_compo, final_eval, training_indices,
                 test_indices, pre_div_path, dataset_path, final_classifier_data = None, stacking_cv = False, meta_cls = LogisticRegression(),
                 meta_clf_fixed = None):
        self.classifiers = classifiers
        self.div_frac = div_frac
        self.pre_div = pre_div
        self.struc_seq_pairs = struc_seq_pairs
        self.train_fit_compo = train_fit_compo
        self.final_eval = final_eval
        self.training_indices = training_indices
        self.test_indices = test_indices
        self.all_indices = test_indices+training_indices
        self.pre_div_path = pre_div_path
        self.final_classifier_data = final_classifier_data
        self.stacking_cv = stacking_cv
        self.meta_cls = meta_cls
        self.all_meta_cls = [LogisticRegression(), RandomForestClassifier(), SVC(probability=True), MLPClassifier()]
        self.meta_cls_fixed = meta_clf_fixed
        self.fixed = self.meta_cls_fixed is not None
        self.data_path = dataset_path

    def mean_evaluate(self, n=3):
        """
        To repetitively evaluate n times.
        :param n:
        :return: mean of evaluation metric (F1-score), standard deviation
        """
        res = []
        for i in range(n):
            res.append(self.evaluate()[0])
        return np.mean(res), np.std(res)

    def evaluate(self, metric = f1_score, cv = 3, classifiers = None):
        """
        Evaluate the ensemble using 'cv' fold cross validation
        :param classifiers:
        :param metric: evaluation metric
        :param cv: Number of splits achieved in cross validation
        :return: mean cross validation score according to the metric
                 (-> fitness value)
        """

        if classifiers is None:
            classifiers= self.classifiers

        data_path = self.data_path
        y = pd.read_csv(data_path+'sequence_based/aac.csv').y.iloc[self.training_indices]
        kf = KFold(cv, shuffle=True)
        split_indices = kf.split(y)
        if self.final_eval:
            if len(self.test_indices) == 0:
                test_indi = random.sample(self.training_indices, round(len(self.training_indices) * (1/cv)))
            else:
                test_indi = self.test_indices
        metric_scores = []
        disagreements = []
        fitted_models = []
        final_scores = []
        for split_idx in split_indices:
            if not self.final_eval:
                y_test = y.iloc[split_idx[1]]
            else:
                y_test = pd.read_csv(data_path+'sequence_based/aac.csv').y.iloc[test_indi]
            preds = []
            predictions = []
            encoding_names = []
            for j, classifier in enumerate(classifiers):
                encoding = classifier.get('encoding')
                encoding_names.append(encoding)
                try:
                    df = pd.read_csv(data_path+'sequence_based/' + encoding)
                except FileNotFoundError:
                    df = pd.read_csv(data_path+'structure_based/' + encoding)
                df2 = df.iloc[self.training_indices]                                        #if error here: check for hidden files in directory
                X_train = df2.iloc[split_idx[0], 1:-1]
                y_train = y.iloc[split_idx[0]]
                if not self.final_eval:
                    X_test = df2.iloc[split_idx[1],1:-1]
                else:
                    X_test = df.iloc[test_indi,1:-1]
                clf = self.KNOWN_CLASSIFIERS.get(classifier.get('cls'))(**classifier.get('params'))
                clf.fit(X_train, y_train)
                fitted_models.append(clf)
                individual_predictions = clf.predict_proba(X_test)
                predictions.append([x[1] for x in individual_predictions])
            if (not self.pre_div) or 't9_st-lambda-correlation_rt-9_ktu-1_la-1' in encoding:
                self.pre_div = False
                for model in predictions:
                    preds.append([0 if x < 0.5 else 1 for x in model])
                disagreements.append(np.mean(list(DisMeas.calc_disagreements(preds, True)[0].values())))          #disagreement computation if not precomputed

            p = [0 if x < 0.5 else 1 for x in np.mean(predictions, axis = 0)]                                      # Fusion method: mean
            metric_scores.append(metric(y_test.values, p))
            if self.final_eval:
                final_scores.append(PME.final_eval(y_test, p))
        print(encoding_names)
        if self.pre_div:
            try:
                disagreements = DisMeas.ensemble_dist_matr(classifiers, self.struc_seq_pairs,self.pre_div_path)       #disagreement grabbing if precomputed
            except KeyError:
                print("No diversity value found. Div set to 0")
                disagreements = 0
        disagreement = np.mean(disagreements)       #takes mean of all disagreements between classifier_data in ensemble TODO: maybe min max diff or sth else
        mean_metric = np.mean(metric_scores)
        div_frac = self.get_div_frac()
        fitness = (1-div_frac) * mean_metric + div_frac * disagreement
        if not self.final_eval:
            return fitness, mean_metric, div_frac, fitted_models, disagreement, np.std(metric_scores), encoding_names,fitted_models

            """Final Eval"""
        else:
            final_f1 = np.mean([x[0] for x in final_scores])
            final_acc = np.mean([x[1] for x in final_scores])
            final_prec = np.mean([x[2] for x in final_scores])
            final_rec = np.mean([x[3] for x in final_scores])
            final_mcc = np.mean([x[4] for x in final_scores])
            final_div_frac = self.final_classifier_data[0]
            final_div = self.final_classifier_data[1]
            final_encodings = self.final_classifier_data[2]
            return [final_f1, final_acc, final_prec, final_mcc, final_rec, final_div_frac, final_div, final_encodings, fitted_models]

    def stacking_evaluate(self, metric ='f1', outer_cv = 3, inner_cv = 3, fold_sets_size = 0.75):
        """
        Evaluate the ensemble using a stacking classifier_data and nested cross validation (outer outer_cv = 'outer_cv', inner outer_cv = 'inner_cv'  .
        :param learn_meta:
        :param fold_sets_size:
        :param inner_cv:
        :param metric:
        :param outer_cv:
        :return: evaluation measures
        """
        data= self.data_path
        y = pd.read_csv(data+'sequence_based/aac.csv').y
        data_frame, pipelines, models, classifier_names, encoding_names = self.build_pipe_and_data()
        print(encoding_names)
        div_frac = self.get_div_frac()
        metric_scores = []
        if self.fixed:
            meta_cls = self.all_meta_cls[self.meta_cls_fixed]
        elif self.meta_cls is not None:
            meta_cls = self.meta_cls
        else:
            meta_cls = LogisticRegression()

        if not self.final_eval:
            if (not self.fixed) and random.random() < 0.15:
                meta_cls = random.sample(self.all_meta_cls, 1)[0]
            fold_indices = random.sample(self.training_indices, round(fold_sets_size*len(self.training_indices)))
            vali_indices = [x for x in self.training_indices if x not in fold_indices]

            y_tr = y.iloc[fold_indices]
            y_vali = y.iloc[vali_indices]
            disagreements = []
            preds = []
            x_fold = data_frame.iloc[fold_indices,:]
            x_vali = data_frame.iloc[vali_indices,:]

            if self.stacking_cv:
                sclf = StackingCVClassifier(classifiers=pipelines, meta_classifier=meta_cls, cv=inner_cv,n_jobs=-1)
            else:
                sclf = StackingClassifier(classifiers=pipelines, meta_classifier=meta_cls)
            sclf.fit(x_fold, y_tr)
            """
            #uncomment to see level 0 classifier performance
            scores = []
            for clf, label in zip(models+[sclf],classifier_names+['StackingClassifier']):
                score = model_selection.cross_val_score(clf, x_vali, y_vali, cv=outer_cv, scoring=metric, n_jobs=-1)
                scores.append(score.mean())
                print(score.mean())
            """
            score = model_selection.cross_val_score(sclf, x_vali, y_vali, cv=outer_cv, scoring=metric, n_jobs=-1)
            if self.pre_div:
                disagreements = DisMeas.ensemble_dist_matr(self.classifiers, self.struc_seq_pairs, self.pre_div_path)  # disagreement-grabbing if precomputed
            else:
                print("calculating disagreement value")
                print("dis calc for stacking not yet implemented")
                disagreements.append(np.mean(list(DisMeas.calc_disagreements(preds, True)[0].values())))  # disagreement computation if not precomputed
            disagreement = np.mean(disagreements)  # takes mean of all disagreements between classifier_data in ensemble TODO: maybe min max diff or sth else
            mean_metric = score.mean()
            fitness = (1 - div_frac) * mean_metric + div_frac * disagreement
            return fitness, mean_metric, div_frac, np.std(metric_scores), disagreement, sclf, meta_cls, encoding_names

            """final evaluation"""
        else:
            if len(self.test_indices) != 0:
                train_indi = self.training_indices
                test_indi = self.test_indices
            else:
                n = round(len(self.training_indices)*fold_sets_size)
                train_indi = self.training_indices[:n]
                test_indi = self.training_indices[n:]
            meta_cls = self.final_classifier_data[2]
            if self.stacking_cv:
                sclf = StackingCVClassifier(classifiers=pipelines, meta_classifier=meta_cls, cv=inner_cv, n_jobs=-1, use_probas=True)
            else:
                sclf = StackingClassifier(classifiers=pipelines, meta_classifier=meta_cls, use_probas=True)
            sclf.fit(data_frame.iloc[train_indi, :], y.iloc[train_indi])
            # sclf = self.final_classifier_data[0]
            p = sclf.predict_proba(data_frame.iloc[test_indi, :])
            p = [x[1] for x in p]
            x_pred = [0 if x < 0.5 else 1 for x in p]
            y_test = y.iloc[test_indi]
            final_diversity = self.final_classifier_data[1]
            return PME.final_eval(y_test, x_pred) + [div_frac, final_diversity, encoding_names, meta_cls,[sclf]]

    def build_pipe_and_data(self):
        data_frame = None
        pipelines = []
        models = []
        classifier_names = []
        encoding_names = []
        start_index = 1
        data = self.data_path

        for j, classifier in enumerate(self.classifiers):
            encoding = classifier.get('encoding')
            cls_name = classifier.get('cls')
            cls = self.KNOWN_CLASSIFIERS.get(cls_name)(**classifier.get('params'))
            classifier_names.append(cls_name)
            try:
                df = pd.read_csv(data+'sequence_based/' + encoding)
            except FileNotFoundError:
                df = pd.read_csv(data+'structure_based/' + encoding)
            #df = df.iloc[self.all_indices,:]
            df.set_index('Sequence_index', inplace=True)
            df.drop(['y'], axis=1, inplace=True)

            encoding_names.append(re.sub('\.csv','',encoding))
            end_index = start_index + len(df.columns) - 1
            pipe = make_pipeline(ColumnSelector(cols=range(start_index, end_index)), cls)
            models.append(cls)
            start_index = end_index
            pipelines.append(pipe)
            if data_frame is None:
                data_frame = df
            else:
                data_frame = data_frame.join(df, on='Sequence_index', how='left', rsuffix=encoding)
        return data_frame, pipelines, models, classifier_names, encoding_names


    def get_div_frac(self):
        """
        Returns fixed diversity fraction if diversity is not to be learned.
        Else: Returns random diversity fraction normally distributed around the diversity that the
                ensemble had in the previous generation; starting with given initial diversity.
        :return: diversity fraction
        """
        prev_frac = self.extract_div_frac()
        if self.train_fit_compo:
            new_div_frac = np.random.normal(loc=prev_frac, scale=0.015)
            return abs(new_div_frac)
        else:
            return prev_frac

    def extract_div_frac(self):
        """
        Extracts diversity fraction value. Important if it's of type list.
        :return:
        """
        try:
            return self.div_frac[0]
        except TypeError:
            return self.div_frac

if __name__=='__main__':
    e = Ensemble([{'cls':'rf','encoding':'ifeature_aaindexencoder_aaindex-NAKH900112_interpol-33.csv','params': {'n_estimators':88}},
                  {'cls': 'svm', 'encoding': 'ifeature_aaindexencoder_aaindex-YUTK870101_interpol-33.csv',
                   'params': {'C':1,'probability': True, 'gamma':'auto'}}])

    print(e.performance_metrics())
