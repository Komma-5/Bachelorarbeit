import os, re
import pandas as pd
import numpy as np
from random import sample

def sort(encoding_type, filename):
    """
    sorts encodings by sequence number, given encoding in 'data/data2' and returns them.
    :param encoding_type:
    :param filename:
    :return:
    """
    encoding_df = pd.read_csv("data/data2/" + encoding_type + filename)
    destination = "data/data2/sorted/" + encoding_type + filename
    indices = encoding_df.iloc[:, 0]
    for i in range(0,len(indices)):
        m = re.search("[0-9]+",encoding_df.iloc[i, 0])
        seq = m.group()
        w = seq
        if len(seq) == 3:
            w = "0" + seq
        elif len(seq)==2:
            w = "00" + seq
        elif len(seq) == 1:
            w = "000" + seq
        encoding_df.iloc[i, 0] = w
    sorted_df = encoding_df.sort_values(by=encoding_df.columns[0],)

    sorted_df.to_csv(destination, index=None)
    exit()
    #return sorted_df

def find_common_seqs():
    """
    Saves number of the sequences that are in all encodings.
    Exits python when done.
    :return:
    """
    init = True
    for encoding_type in ["sequence_based/","structure_based/"]:
        for filename in os.listdir('data/data2/'+ encoding_type):
            print(filename)
            if filename.startswith("."):
                print("skipped")
                continue
            sorted_df = pd.read_csv("data/data2/" + encoding_type + filename)
            print(len(sorted_df))
            if init:
                intersected_array = sorted_df.to_numpy()[:,0]
                init = False
            intersected_array = np.intersect1d(intersected_array, sorted_df.to_numpy()[:,0])
            print(len(intersected_array))

    np.savetxt("data/data2/intersection_indices.csv", intersected_array, delimiter=",", fmt='%s')
    exit()

def find_not_included():
    """
    Returns indices of sequences that are not in all encodings.
    Exits python when done.
    :return:
    """
    included_indices = np.genfromtxt("data/data2/intersection_indices.csv", delimiter=",")
    print(len(included_indices))
    not_included = set(included_indices) ^ set(range(1,688))
    not_included = list(not_included)
    np.savetxt("data/no_intersection_indices.csv", not_included, delimiter=",", fmt='%s')
    print(len(not_included))
    print(len(included_indices)+len(not_included))
    exit()

def get_valid():
    """
    For all encodings the sequences that are not in all encodings are deleted.
    :return:
    """
    source_path = "data/data2/"
    valid_seqs_init = np.genfromtxt(source_path+"intersection_indices.csv", delimiter=",").astype(int)
    for encoding_type in ["sequence_based/", "structure_based/"]:
        done = os.listdir(source_path + "/valid/"+encoding_type)
        for filename in os.listdir(source_path + encoding_type):
            print(filename)
            if filename.startswith(".") or filename in done:
                print("skipped")
                continue
            valid_seqs = valid_seqs_init
            source = source_path+encoding_type+filename
            encoding_df = pd.read_csv(source)
            encoding_df = encoding_df.rename(columns={"Unnamed: 0": "Sequence_index"})
            final_encoding = encoding_df.loc[encoding_df["Sequence_index"].isin(valid_seqs)]
            x = pd.DataFrame(final_encoding)
            x.to_csv(source_path + "/valid/"+encoding_type+filename)
    exit()

def check_disagreement_matrix():
    """
    Checks if all encodings are in encoding distance matrix. If not the names will be returned at "data/not_in_distance_matrix.csv"
    :return:
    """
    data = 'data/amp'
    #diversity_matrix = pd.read_csv("data/encoding_distances/ace_vaxin_diversity_all_vs_all.csv") #pd.read_csv("data/encoding_distances/ace_vaxinpad_diversity.csv")
    diversity_matrix = pd.read_csv("data/amp/encoding_distances/all_vs_all_div.csv")
    diversity_matrix = diversity_matrix.rename(columns={"Unnamed: 0": "Sequence_index"})
    diversity_encodings = np.append(diversity_matrix['Sequence_index'].to_numpy(),diversity_matrix.columns)
    print(diversity_encodings)
    not_in_matrix = []
    for encoding_type in ["sequence_based/", "structure_based/"]:
        for filename in os.listdir('data/amp/' + encoding_type):
            encoding = re.sub("\.csv", "",filename)
            print(encoding)
            if encoding not in diversity_encodings:
                not_in_matrix.append(encoding)
    np.savetxt("data/amp/not_in_distance_matrix.csv",not_in_matrix,delimiter=",",fmt="%s")
    exit()

def move_missing_dis_enc():
    """
    Moves all encodings that are not in distance matrix to "data/not_in_distance_matrix/"
    :return:
    """
    for encoding_type in ["sequence_based/", "structure_based/"]:
        print(encoding_type)
        for filename in os.listdir('data/' + encoding_type):
            encoding = re.sub("\.csv", "", filename)
            if (encoding in pd.read_csv("data/not_in_distance_matrix.csv").to_numpy()) & (encoding != ".DS_Store"):
                os.rename("data/"+encoding_type+filename,"data/not_in_distance_matrix/"+filename)
                print(encoding)
    exit()

def delete_first_row():
    """
    deletes first row of every (header) of every encoding
    :return:
    """
    source_path = "data/data2/"
    for encoding_type in ["sequence_based/", "structure_based/"]:
        print(encoding_type)
        for filename in os.listdir(source_path+'valid/' + encoding_type):
            print(filename)
            if filename.startswith("."):
                print("skipped")
                continue
            df = pd.read_csv(source_path+"valid/"+encoding_type+filename)
            df = df.drop(df.columns[0], axis=1)
            df.to_csv(source_path+"final/"+encoding_type+filename,index=False)
    exit()


def indices_for_ncv(outer_cv, path):
    y = pd.read_csv('data/amp/sequence_based/aac.csv').y
    n = len(y)
    all = range(n)
    folds = []
    all = [x for x in all if x not in folds]
    for i in range(outer_cv):
        new = sample([x for x in all if x not in folds[:i]],round((1/outer_cv)*n))
        folds.append(new)
        pd.DataFrame(new).to_csv(path+'test_set_indices/fold'+str(i+1)+'.csv',index=False,header=False)
        all = [x for x in all if x not in new]
        print("asd")
    exit()

def import_fold_indices(filename, path):
    fold_indices = pd.read_csv(path+'test_set_indices/'+filename).swapaxes(0,1).values
    fold_indices= fold_indices.tolist()[0]
    return fold_indices

def seq_struc_dist_matr():
    diversity_matrix = pd.read_csv("data/encoding_distances/ace_vaxin_diversity_all_vs_all.csv")
    sequence = os.listdir("data/sequence_based")
    struc = os.listdir("data/structure_based")
    seq =[ re.sub("\.csv", "", x) for x in sequence]
    struc =[ re.sub("\.csv", "", x) for x in struc]
    dele_row =[]
    dele_col = []
    for i in range(len(diversity_matrix.iloc[:, 0])):
        if diversity_matrix.iloc[i,0] in struc:
            dele_row.append(i)
        col = diversity_matrix.columns[i]
        if col in seq:
            dele_col.append(col)
    print(dele_row)
    print(dele_col)
    diversity_matrix.drop(dele_row, inplace=True)
    diversity_matrix.drop(dele_col, inplace=True, axis=1)
    print(diversity_matrix)
    exit()

def single_enc_dist_matr(dataset, enco):
    diversity_matrix = pd.read_csv("data/"+dataset+"/encoding_distances/all_vs_all_div.csv")
    #34
    if enco == "seq":
        encos = os.listdir("data/"+dataset+"/structure_based")
    elif enco=="struc":
        encos = os.listdir("data/"+dataset+"/sequence_based")
    encos =[re.sub("\.csv", "", x) for x in encos]
    dele_row =[]
    dele_col = []
    for i in range(1,len(diversity_matrix.iloc[:, 0])):
        #print(diversity_matrix.iloc[i, 0])
        if str(diversity_matrix.iloc[i,0]) in encos:
            print(diversity_matrix.iloc[i, 0])
            dele_row.append(i)
            col = diversity_matrix.columns[i]
            dele_col.append(col)
    print(len(encos))
    print(len(dele_row))
    df = diversity_matrix.iloc[dele_row,:]
    df.index= diversity_matrix.columns[dele_row]
    print(df)
    print([x for x in encos if x not in diversity_matrix.columns[dele_row]])
    diversity_matrix.drop(dele_row, inplace=True)
    diversity_matrix.drop(dele_col, inplace=True, axis=1)
    #print(diversity_matrix)
    exit()

"""
def testing():
    df = pd.read_csv('data/sequence_based/aac.csv').y
    top = df.groupby('gen').apply(lambda group: list(group.astype(float).nlargest(2)))
    N_ENCODINGS= 4
    encoding_counts = pd.Series.value_counts(df[[('encoding{}'.format(i)) for i in range(N_ENCODINGS) ] ].values.ravel())

"""