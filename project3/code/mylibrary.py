import numpy as np
from io import StringIO

class NominalFeature:
    def __init__(self, col_id, members):
        self.col_id = col_id
        self.members = members.copy()
    def __repr__(self):
        return "col_id: " + str(self.col_id) + "\n members: " + str(self.members)

def get_set(filename):
    """
    Purpose:
        Get raw set from given path
    """
    with open(filename) as f:
        raw_set = np.genfromtxt(StringIO(f.read()), delimiter="\t", dtype='str')
    return raw_set

def n_fold(n, pos, raw_set):
    block_size = int(len(raw_set) / n)
    training_set = np.vstack((raw_set[0:block_size*pos], raw_set[block_size*(pos+1):]))
    test_set = raw_set[block_size*pos: block_size*(pos+1)]
    return training_set, test_set

def get_data_label(the_set):
    return the_set[:,:-1], the_set[:,-1].astype(int)

def confusion_matrix(true_label, res_label):
    """
    Purpose:
        get the confusion_matrix
        res[0][0] is the True Postive(TP)
        res[0][1] is the False Negative(FN)
        res[1][0] is the False Positive(FP)
        res[1][1] is the True Negative(TN)
    """
    res = [[0,0],[0,0]]
    for i, val in enumerate(true_label):
        if val == 1:
            if val == res_label[i]:
                res[0][0] += 1
            else:
                res[0][1] += 1
        else:
            if val != res_label[i]:
                res[1][0] += 1
            else:
                res[1][1] += 1
    return np.asarray(res)

def get_accuracy(confusion):
    """
    Purpose:
        accuracy = (TP + TN)/(TP+TN+FP+FN)
    """
    return (confusion[0,0] + confusion[1,1]) / np.sum(confusion)

def get_precision(confusion):
    """
    PRECISION = TP / (TP + FP)
    """
    return confusion[0,0] / (confusion[0,0] + confusion[1,0])

def get_recall(confusion):
    """
    Recall = TP / (TP + FN)
    """
    return confusion[0,0] / (confusion[0,0] + confusion[0,1])

def get_f1_score(confusion):
    """
    F1 score = 2*TP / (2*TP + FN + FP)
    """
    return 2 * confusion[0,0] / (2 * confusion[0,0] + confusion[0,1] + confusion[1,0])


def error(div_arr):
    div_arr = np.asarray(div_arr)
    t = np.sum(div_arr)
    return 1 - np.max(div_arr / t)

def gini(div_arr):
    div_arr = np.asarray(div_arr)
    t = np.sum(div_arr)
    return 1 - np.sum(np.square(div_arr/t))

def entropy(div_arr):
    div_arr = np.asarray(div_arr)
    t = np.sum(div_arr)
    div_arr = div_arr / t
    res = 0
    for elem in div_arr:
        if elem != 0:
            res += elem * np.log2(elem)
    return -res

def gain(parent, children, impurity_fun):
    parent = np.asarray(parent)
    children = np.asarray(children)
    
    n = np.sum(parent)
    children_weight = []
    for child in children:
        children_weight.append(np.sum(child))
    children_weight = np.asarray(children_weight) / n
    children_impurity = np.asarray([impurity_fun(child) for child in children])
    return impurity_fun(parent) - np.sum(children_weight * children_impurity)

def convert_label(label, new_neg=-1, new_pos=1, old_neg=0, old_pos=1):
    new_label = np.zeros(label.shape)
    new_label[label == old_neg] = new_neg
    new_label[label == old_pos] = new_pos
    return new_label

def sign(val):
    if val > 0:
        return 1
    return -1



