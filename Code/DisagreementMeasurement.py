import pandas as pd
import numpy as np
import re


def classifier_dist_matr(classifier1, classifier2, struc_seq_pairs, pre_div_path):
    """
    Returns diversity value of two classifier_data from precomputed diversity matrix.
    If classifier_data are given in default mode (classier1 is structure, classier2 is sequence based),
    it's slightly faster.
    :param classifier1:
    :param classifier2:
    :param struc_seq_pairs:
    :param pre_div_path: path of precomputed diversity matrix
    :return:
    """
    diversity_matrix = pd.read_csv(pre_div_path)
    diversity_matrix = diversity_matrix.set_index('Unnamed: 0')
    encoding1 = re.sub("\.csv","",classifier1.get('encoding'))
    encoding2 = re.sub("\.csv","",classifier2.get('encoding'))
    if struc_seq_pairs:
        sequence_encoding = encoding1
        structure_encoding = encoding2
    else:
        if classifier1.get('encoding_type') =='sequence':
            sequence_encoding = encoding1
            structure_encoding = encoding2
        else:
            sequence_encoding = encoding2
            structure_encoding = encoding1
    return diversity_matrix.loc[structure_encoding,sequence_encoding]

def ensemble_dist_matr(classifier_ensemble, struc_seq_pairs, pre_div_path):
    """
    Returns disagreement values of an classifier_data ensemble from a precomputed distance matrix.
    If the classifier_data in the ensemble are ordered in default (classifier1 is structure, classifier2 is sequence based),
    it's slightly faster.
    :argument: classifier_ensemble
    :return:
    """
    disagreements = []
    for i in range(0,len(classifier_ensemble)-1,2):
        disagreements.append(classifier_dist_matr(classifier_ensemble[i], classifier_ensemble[i+1], struc_seq_pairs, pre_div_path))
    return disagreements

def calc_disagreements(classifiers, relative):
    """
    Returns disagreement values as dict
    between each classifier_data output in 'classifier_data'.
    :param classifiers:
    :return:
    """
    n = len(classifiers[0])
    m = len(classifiers)

    disagreements = np.zeros((m, m), dtype=int)
    dict = {}

    # if rel == 1 the absolute value is returned
    rel = 1
    if relative:
        rel = n

    for i in range(0, m):
        for j in range(0, m):
            if i == j:
                break
            for k in range(0, n - 1):
                current_dis = abs(classifiers[i][k] - classifiers[j][k])/rel
                #disagreements[i][j] += current_dis
                key = str(i) + "," + str(j)
                if key not in dict:
                    dict[key] = current_dis
                else:
                    dict[key] += current_dis
    return dict, disagreements

def max_disagreement(c, disagreement_dict):
    """
    Returns indices with the c highest disagreement values, given a dictionary and c
    :param c:
    :param disagreement_dict:
    :return:
    """
    n = len(disagreement_dict)
    if c > n:
        return max_disagreement(c - 1, disagreement_dict)
    if c == n:
        return list(range(0, n))

    indices = set()
    for i in range(0, c - 1):
        index_as_key = max(disagreement_dict, key=disagreement_dict.get)
        for x in re.split(",", index_as_key):
            indices.add(int(x))
        del disagreement_dict[index_as_key]

    return list(indices)

def max_disagreement_from_matrix(disagreement_tri_matrix):
    """
    Not ready
    Max disagreement from matrix.
    :param c:
    :param disagreement_tri_matrix:
    :return:
    """
    result = np.where(disagreement_tri_matrix == np.amax(disagreement_tri_matrix))
    list_of_coordinates = list(zip(result[0], result[1]))
    return list_of_coordinates
