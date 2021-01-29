import argparse
import operator
import re
from multiprocessing.spawn import freeze_support

from deap import base, creator, tools
from random import choice, random, gauss
import numpy as np

from Ensemble import Ensemble
import DataManipulation as DM
import NestedCV as NCV
import pandas as pd
from time import time
import os
import sys
import signal
import threading
import csv
import multiprocessing
import random as rd
from joblib import dump
from sklearn.metrics import f1_score


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class GA:
    def __init__(self, test_indices = [], fold_indices = None, **kwargs):

        self.toolbox = base.Toolbox()
        self.kwargs = kwargs
        self.setup()
        self.pop = self.toolbox.population(n=self.kwargs.get('popsize'))
        self.ended = False
        self.test_indices = test_indices
        self.fold_indices = fold_indices
        self.training_indices, self.vali_indices = self.get_indices()
        if self.kwargs.get("import_ind") is None:
            self.id = ""
        else:
            self.id = re.sub("\.csv", "", self.kwargs.get("import_ind"))+'_'
        self.gens = args.__dict__.get('gens')

    def setup(self):
        """
        Evolutionary algorithm setup according to ../Documentation/Genetic Algorithm Design.md
        Functions are registered to the toolbox without instantiating anything yet.
        Functions are then used throughout the algorithm.
        :return:
        """
        creator.create("F1Max", base.Fitness, weights=(1.0,))
        creator.create("DiversityMax", base.Fitness, weights=(1.0,))
        if self.kwargs.get('div_excl'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax, f1score=creator.F1Max)
        else:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox.register("attr_rf_n", lambda: 1 + round(abs(gauss(140, 10))))
        self.toolbox.register("attr_svm_C", lambda: abs(gauss(0.5, 0.1)) + 0.000001)


        sequence_encodings = os.listdir("./data/"+self.kwargs.get('dataset')+"sequence_based")
        structure_encodings = os.listdir("./data/"+self.kwargs.get('dataset')+"structure_based")
        self.toolbox.register("attr_sequence_encoding", choice, sequence_encodings)
        self.toolbox.register("attr_structure_encoding", choice, structure_encodings)


        self.toolbox.register("attr_structure_RF", lambda: {'cls': 'rf', 'encoding': self.toolbox.attr_structure_encoding(), 'encoding_type':'structure',
                                                  'params': {'n_estimators': self.toolbox.attr_rf_n()}})

        self.toolbox.register("attr_structure_SVM", lambda: {'cls': 'svm', 'encoding': self.toolbox.attr_structure_encoding(),'encoding_type':'structure',
                                                    'params': {'C': self.toolbox.attr_svm_C(),'probability': True}})

        self.toolbox.register("attr_sequence_RF", lambda: {'cls': 'rf', 'encoding': self.toolbox.attr_sequence_encoding(),'encoding_type':'sequence',
                                                  'params': {'n_estimators': self.toolbox.attr_rf_n()}})

        self.toolbox.register("attr_sequence_SVM", lambda: {'cls': 'svm', 'encoding': self.toolbox.attr_sequence_encoding(),'encoding_type':'sequence',
                                                   'params': {'C': self.toolbox.attr_svm_C(),'probability': True}})

        self.toolbox.register("attr_structure_classifier", lambda: choice([self.toolbox.attr_structure_RF(), self.toolbox.attr_structure_SVM()]))
        self.toolbox.register("attr_sequence_classifier", lambda: choice([self.toolbox.attr_sequence_RF(), self.toolbox.attr_sequence_SVM()]))

        self.toolbox.register("attr_div_frac", lambda: {'div_frac':self.kwargs.get('div_frac')})

        if self.kwargs.get("enco_method") is None:
            self.toolbox.register("individual", tools.initCycle, creator.Individual,
                                  (self.toolbox.attr_structure_classifier, self.toolbox.attr_sequence_classifier),
                                  int(np.floor(self.kwargs.get('n_classifiers')/2)))
        elif self.kwargs.get("enco_method") == "struc":
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_structure_classifier,
                                  self.kwargs.get('n_classifiers'))
        elif self.kwargs.get("enco_method") == "seq":
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_sequence_classifier,int(self.kwargs.get('n_classifiers')))

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        path = str(self.kwargs.get('data_path'))+str(self.kwargs.get('dataset'))

        if self.kwargs.get('stacking'):
            self.toolbox.register("evaluate", lambda ind,div=self.kwargs.get('div_frac'), final_eval=False, data = [], meta_cls = None:
            Ensemble(ind, div, self.kwargs.get('pre_div'), self.kwargs.get('struc_seq_pairs'), self.kwargs.get('train_fit_compo'),
                     final_eval=final_eval, training_indices=self.training_indices, test_indices=self.vali_indices,
                     pre_div_path=path+'encoding_distances/'+self.kwargs.get('pre_div_path'),dataset_path=path,
                     final_classifier_data=data, stacking_cv=self.kwargs.get('stacking_cv'), meta_cls=meta_cls, meta_clf_fixed=self.kwargs.get('fixed_meta_clf')).stacking_evaluate())
        else:
            self.toolbox.register("evaluate", lambda ind,div=self.kwargs.get('div_frac'),final_eval=False ,data = [], meta_cls = None:
            Ensemble(ind, div, self.kwargs.get('pre_div'), self.kwargs.get('struc_seq_pairs'), self.kwargs.get('train_fit_compo'),
                     final_eval, self.training_indices, self.vali_indices, final_classifier_data= data,
                     pre_div_path=path+'encoding_distances/'+self.kwargs.get('pre_div_path'),dataset_path=path).evaluate())
        self.toolbox.register("recombine", tools.cxOnePoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selBest)

    def get_indices(self):
        """
        Returns Training and validation set indices.
        :return:
        """
        y = pd.read_csv('data/'+self.kwargs.get('dataset')+'sequence_based/aac.csv').y
        if self.fold_indices is None:
            all = [x for x in range(len(y)) if x not in self.test_indices]
            training_set_indices = rd.sample(all, round((len(all) * self.kwargs.get('inner_train_size'))))
            vali_set_indices = [x for x in all if x not in training_set_indices]
        else:
            training_set_indices = [x for x in range(len(y)) if x not in self.fold_indices]
            vali_set_indices = self.fold_indices
        return training_set_indices, vali_set_indices

    def mutate(self, individual):
        """
        Mutates individuals of the form [{'cls':'rf','params':{'n_estimators':100},'encoding':'some encoding33.csv'}...]

        :param individual:
        :return:
        """
        for c in range(len(individual)):
            if individual[c].get('cls') == 'rf':
                individual[c]['params']['n_estimators'] = 1 + round(
                    abs(gauss(individual[c]['params']['n_estimators'], 3) - 1))
            elif individual[c].get('cls') == 'svm':
                x = individual[c]['params']['C']
                individual[c]['params']['C'] = abs(gauss(x, x * 1 / 30))
            if random() < 0.1:
                individual[c]['encoding'] = self.mutate_encoding(individual[c])
            if random() < 0.04:
                if individual[c]['cls'] == 'svm':
                    if individual[c]['encoding_type'] == 'structure':
                        individual[c] = self.toolbox.attr_structure_RF()
                    else:
                        individual[c] = self.toolbox.attr_sequence_RF()
                else:
                    if individual[c]['encoding_type'] == 'structure':
                        individual[c] = self.toolbox.attr_structure_SVM()
                    else:
                        individual[c] = self.toolbox.attr_sequence_SVM()
        return individual

    @staticmethod
    def mutate_encoding(ind):
        """
        Mutate an encoding
        :return: encoding of the mutated encoding
        """
        args = argparser()
        base_path=args.__dict__.get('data_path')+args.__dict__.get('dataset')
        enc_type = ind["encoding_type"]

        if enc_type=='structure':
            encoding_group=os.listdir(base_path+"structure_based")
        elif enc_type=='sequence':
            encoding_group=os.listdir(base_path+"sequence_based")
        encoding_group = [re.sub("\.csv","",x) for x in encoding_group]
        n_encodings = len(encoding_group)

        encoding = re.sub("\.csv","",ind['encoding'])
        div_matrix =pd.read_csv(base_path+"encoding_distances/all_vs_all_div.csv")

        if 't9_st-lambda-correlation_rt-9_ktu-1_la-1' not in encoding:
            sorted_divs = div_matrix.sort_values(encoding)
        else:
            sorted_divs = div_matrix

        divs = [x for x in sorted_divs.iloc[:,0] if x in encoding_group]
        def exponential_k():
            if enc_type == 'structure':
                z = 2
            elif enc_type == 'sequence':
                z = 10
            res = round(np.random.exponential(z)) + 1
            while res >= n_encodings:
                round(np.random.exponential(z)) + 1
            return res
        k = exponential_k()
        new = divs[k]+".csv"
        return new



    def evolve(self):
        """
        Evolution limited to the maximum of kwargs.gens and kwargs.runtime.
        Variation is achieved via Mutation and Recombination.

        :return:
        """

        self.all_pops = []
        print("---------- start of evolution ----------")
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            if self.kwargs.get('div_excl'):
                ind.fitness.values = [fit[0]]
                ind.f1score.values = [fit[1]]
            else:
                ind.fitness.values = [fit[0], fit[4]]
        start_time = time()
        self.g = 0
        #meta_cls = None
        while time() - start_time < (self.kwargs.get('runtime')*3600) and self.gens - self.g > 0:
            self.classifier_data = []

            self.g += 1
            print('\ngen:', self.g)
            offspring = self.offspring_selection(self.kwargs.get('selection'))
            offspring = list(map(self.toolbox.clone, offspring))

            for mutant in offspring[int(len(offspring)*self.kwargs.get('selection'))::2]:
                self.toolbox.mutate(mutant)
            parents = offspring[:int(len(self.pop) / 2)]
            parents.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
            parents = parents[:4]

            for i in range(0, len(parents), 2):
                self.toolbox.recombine(parents[i], parents[i + 1])

            for individual in offspring:
                fit = self.toolbox.evaluate(individual)
                #individual.f1score.values = [fit[1]]
                #individual.diversity.values = [fit[4]]
                fitness  = fit[0]
                f1  = fit[1]
                if self.kwargs.get('div_excl'):
                    individual.fitness.values = [fitness]
                    individual.f1score.values = [f1]
                else:
                    individual.fitness.values = [fitness, fit[4]]
                if self.kwargs.get('stacking'):
                    meta_cls = fit[6]
                    scls = fit[5]
                    self.classifier_data.append([fitness, fit[5], fit[4], meta_cls,[scls]])

                    if self.kwargs.get('fixed_meta_clf') is None:
                        self.save_meta_cls(self.g,meta_cls,fit[7])
                else:
                    encodings = fit[6]
                    fitted_models = fit[7]
                    div_frac = fit[2]
                    div = fit[4]
                    self.classifier_data.append([fitness, div_frac, div, encodings, fitted_models])

            self.pop = offspring
            if self.kwargs.get('div_excl'):
                #self.all_pops.append(list(zip(offspring, [(i.fitness.values[0], i.f1score.values[0]) for i in offspring])))

                pop = (list(zip(offspring, [(indi.fitness.values[0], indi.f1score.values[0]) for indi in offspring])))
                #self.pops_to_df().to_csv(self.kwargs.get('savepath') + self.id + 'population_details.csv')

            else:
                #self.all_pops.append(list(zip(offspring, [(i.fitness.values[0],i.fitness.values[1]) for i in offspring])))
                pop = list(zip(offspring, [(indi.fitness.values[0],indi.fitness.values[1]) for indi in offspring]))



            self.write_pop(pop, self.g)

        self.final_report(self.get_best_model_data())


    def get_best_model_data(self):
        sorted_by_f1 = sorted(self.classifier_data, key=operator.itemgetter(0))
        return sorted_by_f1[0][1:]

    def offspring_selection(self, frac_survivors):
        """
        Selects the top frac_survivors fraction of the population according to the selection method in toolbox.
        Repeats that selection until an offspring population of same cardinality is established.
        :param frac_survivors:
        :return:
        """
        n = len(self.pop)
        total_survivors = int(n * frac_survivors)
        copy_iterations = int(n / total_survivors)
        total_fill_up_individuals = n - copy_iterations * total_survivors
        offspring = self.toolbox.select(self.pop, total_survivors) * copy_iterations + self.toolbox.select(self.pop,
                                                                                                           total_fill_up_individuals)
        return offspring

    def save_meta_cls(self,g ,cls, encodings):
        filename=self.kwargs.get('savepath')+self.id+"meta_cls.csv"
        file_exists = False
        if os.path.exists(filename):
            file_exists = True
        with open(filename, 'a') as fd:
            fields = [g,cls,encodings]
            writer = csv.writer(fd)
            if not file_exists:
                writer.writerow(["Generation","Meta_Classifier","encodings"])
            writer.writerow(fields)
            fd.close()

    def pops_to_df(self):
        """
        Transforms the all_populations data into a pd.DataFrame for later analysis [e.g. in a jupyter notebook].
        Saves to kwargs.savepath
        :return:
        """
        cols = ['fitness'] + ['cls{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['RF_n{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['SVM_C{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['SVM_gamma{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['encoding{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))]

        df = pd.DataFrame(columns=cols, index=pd.MultiIndex.from_tuples(
            [(i, j) for i in range(len(self.all_pops)) for j in range(len(self.all_pops[0]))],
            names=['gen', 'individual']))
        for i, pop in enumerate(self.all_pops):
            for j, individual in enumerate(pop):
                df.loc[(i, j), :] = [individual[1]] \
                                    + [individual[0][j]['cls'] for j in
                                       range(self.kwargs.get('n_classifiers'))] \
                                    + [individual[0][j]['params'].get('n_estimators') for j in
                                       range(self.kwargs.get('n_classifiers'))] \
                                    + [individual[0][j]['params'].get('C', 0) for j in
                                       range(self.kwargs.get('n_classifiers'))] \
                                    + [individual[0][j]['params'].get("gamma", 0) for j in
                                       range(self.kwargs.get('n_classifiers'))] \
                                    + [individual[0][j]['encoding'] for j in range(self.kwargs.get('n_classifiers'))]
        return df

    def write_pop(self,pop,gen):
        cols = ['fitness'] + ['cls{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['RF_n{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['SVM_C{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['SVM_gamma{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))] + \
               ['encoding{}'.format(j) for j in range(self.kwargs.get('n_classifiers'))]

        df = pd.DataFrame(columns=cols, index=pd.MultiIndex.from_tuples(
            [(1, j) for j in range(len(pop))],
            names=['gen', 'individual']))
        for j, individual in enumerate(pop):
            df.loc[(gen, j), :] = [individual[1]] \
                                + [individual[0][j]['cls'] for j in
                                   range(self.kwargs.get('n_classifiers'))] \
                                + [individual[0][j]['params'].get('n_estimators') for j in
                                   range(self.kwargs.get('n_classifiers'))] \
                                + [individual[0][j]['params'].get('C', 0) for j in
                                   range(self.kwargs.get('n_classifiers'))] \
                                + [individual[0][j]['params'].get("gamma", 0) for j in
                                   range(self.kwargs.get('n_classifiers'))] \
                                + [individual[0][j]['encoding'] for j in range(self.kwargs.get('n_classifiers'))]
        df.to_csv(self.kwargs.get('savepath')+self.id+"population_details.csv",mode="a", header=False)

    def final_report(self, classifier_data=None):
        """
        A printed command line overview of the results.
        :return:
        """
        best_ind = tools.selBest(self.pop, 1)[0]
        print('---------- end of evolution ----------')

        print('\nResults for best individuum:')

        imp = self.kwargs.get('import_ind') is not None
        mean_data = self.toolbox.evaluate(best_ind, final_eval=True, data=classifier_data)

        """save fitted model (of whole set or imported fold)"""
        fitted_classifiers = classifier_data[-1] if imp else mean_data[-1]
        if self.kwargs.get('save_m'):
            for clf in fitted_classifiers:
                dump(clf, self.kwargs.get('savepath') + self.id + "_" + str(time()) + 'fitted_model.joblib')
        if (self.kwargs.get('stacking')):
            mean_data = mean_data[:-1]
        else:
            mean_data = mean_data[:-1] + ["-"]

        filename = self.kwargs.get('savepath') + self.id + 'eval_scores.csv'
        imported_fold = self.kwargs.get("import_ind")
        file_exists = False
        if os.path.exists(filename):
            file_exists = True
        with open(filename, 'a') as fd:
            fields = mean_data + [self.g, self.kwargs.get('popsize'), self.kwargs.get('train_fit_compo'),
                                  self.kwargs.get('stacking'), imported_fold, self.kwargs.get('dataset')]
            writer = csv.writer(fd)
            if not file_exists:
                writer.writerow(
                    ["F1", "Accuracy", "Precision", "Recall", "MCC", "Div Frac", "Final Div Measure", "Encodings",
                     "Meta Classifier", "Generations", "Pop Size", "Learned Div", "Stacking", "Import","Dataset"])
            writer.writerow(fields)
        if self.kwargs.get('div_excl'):
            print('One-time F1-score:', best_ind.f1score.values[0])
            print('One-time Fitness-score:', best_ind.fitness.values[0])
            print('One-time diversity:', mean_data[6])
            print('One-time diversity fraction:', mean_data[5])
        else:
            print('One-time F1-score:', best_ind.fitness.values[0] - (mean_data[5] * best_ind.fitness.values[1]))
            print('One-time Fitness-score:', best_ind.fitness.values[0])
            print('One-time diversity:', best_ind.fitness.values[1])


def signal_handler():
    """
    Handles system interrupts.
    Specifically if the script is terminated [Ctrl + C], the results are saved.
    :param signal:
    :param frame:
    :return:
    """
    print('You pressed Ctrl+C!')
    print('Exiting.')
    exit()

def argparser():
    """
    Command line arguments are specified here.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-runtime', action="store", type=int, default=20, help='Runtime in hours')
    parser.add_argument('-gens', action="store", type=int, default=-1, help='Number of maximum generations')
    parser.add_argument('-n_classifiers', action="store", type=int, default=4, help='Number of classifier_data in ensemble. Must be even, since the individual is a composition of structure and sequence based classifier_data.')
    parser.add_argument('-n_structure_classifiers', action="store", type=int, default=2, help='Number of structure classifier_data in ensemble')
    parser.add_argument('-n_sequence_classifiers', action="store", type=int, default=2, help='Number of sequence classifier_data in ensemble')
    parser.add_argument('-savepath', action="store", type=str, default='Results/', help='Path for saving results and performance')
    parser.add_argument('-data_path', action="store", type=str, default='data/', help='Path for saving results and performance')
    parser.add_argument('-dataset', action="store", type=str, default='amp/', help='Path to dataset folder')
    parser.add_argument('-popsize', action="store", type=int, default=40, help='Population size')
    parser.add_argument('-selection', action='store', type=float, default=0.5, help='Percentage of offspring that continues to the next generation')
    parser.add_argument('-div_excl', action="store_true", default=False, help='If set, diversity is excluded from optimisation process. Note: Div can still be included in fitness via -div_frac.')
    parser.add_argument('-div_frac', action='store', type=float, default=0, help='Composition of diversity and F1 Score: (1-div_frac) * F1 + div_frac * diversity. Default: 0')
    parser.add_argument('-train_fit_compo', action="store_true", default=False, help='If true, the GA will try to learn the best fitness composition of fitness and f1 score. Else: use fixed "div_frac" diversity composition. Default: False.')
    parser.add_argument('-pre_div', action="store_true", default=False, help='If true, diversity values are taken from a precomputed disagreement matrix.')
    parser.add_argument('-pre_div_path', action="store", type=str, default='seq_vs_struc_div.csv', help='Path to diversity matrix. Only use with \'-pre_div\' argument.')
    parser.add_argument('-stacking', action="store_true", default=False, help='If true, the ensemble will be combinded through a meta classifier_data (stacking). Else: mean. Default: false.')
    parser.add_argument('-stacking_cv', action="store_true", default=False, help='If true, the ensemble will be combinded through a meta classifier_data (stacking) using cross validation. Else: mean. Default: false.')
    parser.add_argument('-fixed_meta_clf', action='store', type=int, default=None, help='Indexed of fixed meta classifier for stacking')
    parser.add_argument('-inner_train_size', action='store', type=float, default=0.8, help='Size of training set size. e.g. 0.9 = 90% of whole dataset. default: 0.8')
    parser.add_argument('-outer_cv', action='store', type=int, default=1, help='Number of folds of outer CV (for Nested CV). Default: 1=No NCV')
    parser.add_argument('-import_ind', action="store",type=str, default=None, help='If set, testset indices will be imported_fold from data/test_set_indices/ import_ind. Default: false.')
    parser.add_argument('-enco_method', action="store",type=str, default=None, help='Sets construction of encoding methods used in ensemble. If \'None\' (default): structured and sequence based; if \'struc\' only structure based; if \'seq\' only sequence based encodings.')
    parser.add_argument('-struc_seq_pairs', action="store", type=bool, default=True, help='If true, there\'s one sequence classifier_data for every structure classifier_data in an ensemble (structure,sequence pairs).')
    parser.add_argument('-save_m', action="store_true", default=False, help='If true, the fitted models will be saved as ".joblib".')
    parser.add_argument('-ncvEval', action="store_true", default=False)

    return parser.parse_args()

def eval_ncv():

    NCV.final_ncv_eval(3, 'Results/NoStacking/Run7/', 'fitted_models/', 'data/imo/test_set_indices/',
                       [['ssec.csv', 'waac_aaindex_BUNA790103.csv', 'delaunay_average_distance.csv', 'fft_aaindex_GEOR030103.csv'],
                        ['electrostatic_hull_0.csv', 'fft_aaindex_QIAN880102.csv', 'electrostatic_hull_9.csv', 'dde.csv'],
                        ['sseb.csv', 'dist_freq_dn_10_dc_5.csv', 'ta.csv', 'ngram_a2_20.csv']],
                       'final_eval_ncv.csv')

if __name__ == '__main__':
    args = argparser()

    if args.__dict__.get('ncvEval'):
        eval_ncv()

    signal.signal(signal.SIGINT, signal_handler)
    forever = threading.Event()
    path = args.__dict__.get('data_path')+args.__dict__.get('dataset')
    y = pd.read_csv(path+'sequence_based/aac.csv').y
    print('PARAMETERS')
    for key in args.__dict__.keys():
        print('{}: {}'.format(key, args.__dict__.get(key)))
    print()
    outer_cv = args.__dict__.get('outer_cv')
    file = args.__dict__.get('import_ind')
    if file is None:
        fold_ind = None
    else:
        fold_ind = DM.import_fold_indices(file, path)
    if outer_cv <= 1:
        ti = time()
        ga = GA(fold_indices=fold_ind, **args.__dict__)
        ga.evolve()
        print(time() - ti)
    else:
        folds = NCV.indices_for_ncv(outer_cv,'data/'+args.__dict__.get('dataset')+'sequence_based/aac.csv')
        for i in range(outer_cv):
            ti = time()
            ga = GA(folds[i], **args.__dict__)
            ga.evolve()
            print(time()-ti)
    sys.exit(0)

# To run this code, try:
# python3 GA.py -gens 1 -popsize 4 -n_classifiers 2 -div_frac 0 -stacking


"""    def setup(self):
        Evolutionary algorithm setup according to ../Documentation/Genetic Algorithm Design.md
        Functions are registered to the toolbox without instantiating anything yet.
        Functions are then used throughout the algorithm.
        :return:

        creator.create("F1Max", base.Fitness, weights=(1.0,))
        creator.create("DiversityMax", base.Fitness, weights=(1.0,))
        if self.kwargs.get('div_excl'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax, f1score=creator.F1Max)
        else:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox.register("attr_rf_n", lambda: 1 + round(abs(gauss(140, 10))))
        self.toolbox.register("attr_svm_C", lambda: abs(gauss(0.5, 0.1)) + 0.000001)


        sequence_encodings = os.listdir("./data/"+self.kwargs.get('dataset')+"sequence_based")
        structure_encodings = os.listdir("./data/"+self.kwargs.get('dataset')+"structure_based")
        self.toolbox.register("attr_sequence_encoding", choice, sequence_encodings)
        self.toolbox.register("attr_structure_encoding", choice, structure_encodings)


        self.toolbox.register("attr_structure_RF", lambda: {'cls': 'rf', 'encoding': self.toolbox.attr_structure_encoding(), 'encoding_type':'structure',
                                                  'params': {'n_estimators': self.toolbox.attr_rf_n()}})

        self.toolbox.register("attr_structure_SVM", lambda: {'cls': 'svm', 'encoding': self.toolbox.attr_structure_encoding(),'encoding_type':'structure',
                                                    'params': {'C': self.toolbox.attr_svm_C(),'probability': True}})

        self.toolbox.register("attr_sequence_RF", lambda: {'cls': 'rf', 'encoding': self.toolbox.attr_sequence_encoding(),'encoding_type':'sequence',
                                                  'params': {'n_estimators': self.toolbox.attr_rf_n()}})

        self.toolbox.register("attr_sequence_SVM", lambda: {'cls': 'svm', 'encoding': self.toolbox.attr_sequence_encoding(),'encoding_type':'sequence',
                                                   'params': {'C': self.toolbox.attr_svm_C(),'probability': True}})

        self.toolbox.register("attr_structure_classifier", lambda: choice([self.toolbox.attr_structure_RF(), self.toolbox.attr_structure_SVM()]))
        self.toolbox.register("attr_sequence_classifier", lambda: choice([self.toolbox.attr_sequence_RF(), self.toolbox.attr_sequence_SVM()]))

        self.toolbox.register("attr_div_frac", lambda: {'div_frac':self.kwargs.get('div_frac')})



        if self.kwargs.get("enco_method") is None:
            self.toolbox.register("individual", tools.initCycle, creator.Individual,
                                  (self.toolbox.attr_structure_classifier, self.toolbox.attr_sequence_classifier),
                                  int(np.floor(self.kwargs.get('n_classifiers')/2)))
        elif self.kwargs.get("enco_method") == "struc":
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_structure_classifier,
                                  self.kwargs.get('n_classifiers'))
        elif self.kwargs.get("enco_method") == "seq":
            self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                                  self.toolbox.attr_sequence_classifier,int(self.kwargs.get('n_classifiers')))

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        path = str(self.kwargs.get('data_path'))+str(self.kwargs.get('dataset'))

        if self.kwargs.get('stacking'):
            self.toolbox.register("evaluate", lambda ind,div=self.kwargs.get('div_frac'), final_eval=False, data = [], meta_cls = None:
            Ensemble(ind, div, self.kwargs.get('pre_div'), self.kwargs.get('struc_seq_pairs'), self.kwargs.get('train_fit_compo'),
                     final_eval=final_eval, training_indices=self.training_indices, test_indices=self.vali_indices,
                     pre_div_path=path+'encoding_distances/'+self.kwargs.get('pre_div_path'),dataset_path=path,
                     final_classifier_data=data, stacking_cv=self.kwargs.get('stacking_cv'), meta_cls=meta_cls, meta_clf_fixed=self.kwargs.get('fixed_meta_clf')).stacking_evaluate())
        else:
            self.toolbox.register("evaluate", lambda ind,div=self.kwargs.get('div_frac'),final_eval=False ,data = [], meta_cls = None:
            Ensemble(ind, div, self.kwargs.get('pre_div'), self.kwargs.get('struc_seq_pairs'), self.kwargs.get('train_fit_compo'),
                     final_eval, self.training_indices, self.vali_indices, final_classifier_data= data,
                     pre_div_path=path+'encoding_distances/'+self.kwargs.get('pre_div_path'),dataset_path=path).evaluate())
        self.toolbox.register("recombine", tools.cxOnePoint)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selBest)
"""