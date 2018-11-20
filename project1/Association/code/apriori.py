#!/usr/bin/python3

import sys
import pickle
import numpy as np
from io import StringIO
from collections import deque
import time


def get_data(filename):
    """
    Purpose:
        Read file, and extract data and label from it.
    Input:
        filename: String
    Output:
        data: a matrix of string
        label: a vector of string
    """    
    with open(filename) as f:
        raw_data = np.genfromtxt(StringIO(f.read()), delimiter="\t",dtype='str')
        data = raw_data[:,:-1]
        label = raw_data[:,-1]
    return data, label


def pre_process(data):
    """
    Purpose:
        preprocees data in place
    Input:
        data: a matrix of string
    output:
        None
    """

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i,j] = 'G' + str(j+1) + '_' + data[i,j]


def get_C1(data):
    """
    Purpose:
        Get all the length one itemsets of data
    input: 
        data: a matrix of string
    output: 
        res: a set contains frozenset type elements.
    """
    res = set()
    for row in data:
        for item in row:
            res.add(frozenset([item]))
    return res

def get_freqI(data, Ck, min_support, record):
    """
    Purpose:
        Generate the frequent and the unfrequent items from the candidate itemsets.
    input:
        data: a matrix of string 
        Ck : set of frozenset, current candidate frequent itemsets
        min_support: float, the minimum support
        record: a dictionary. Key type is frozenset, value type is float.
    Output:
        Fq: a list of frozenset, contains all the frequent itemsets.
        UnFq: a list of frozenset, contains all the unfrequent itemsets.
    """

    N = len(data)
    Fq = []
    UnFq = []
    for item in Ck:
        count = 0
        for row in data:
            if item.issubset(row):
                count += 1
        support = count / N
        if support >= min_support:
            Fq.append(item)
        else:
            UnFq.append(item)
        record[item] = support
    return Fq, UnFq  


def get_Ck(Fq):
    """
    Purpose:
        Gnenrate the candidate itemsets by using previous freqent itemsets
    input:
        Fq: list of frozenset, previous frequent itemsets
    output:
        Ck: set of frozenset, current candidate itemsets
    """
    C = Fq
    C_level = len(C[0])
    C1 = set()
    Ck = set()

    for item in C:
        for elem in item:
            C1.add(frozenset([elem]))
        
    for item in C:
        for elem in C1:
            check = item | elem
            if(len(check) - C_level == 1):
                Ck.add(check)
    return Ck


def eliminate_infeq(Ck, Ck_pre_unfq):
    """
    Purpose:
        prune the candidate itemsets by the previous unfrequent itemsets.
    Input:
        Ck: a set of frozenset, current candidate itemsets
        unfq: a list of frozenset, previous unfrequent itemsets.
    Output:
        res: a set of frozenset, the prunded current candidate itemsets
    """
    res = set()
    for i in Ck:
        for j in Ck_pre_unfq:
            if j.issubset(i):
                break
        else:
            res.add(i)
    return res


def apriori(filename, support=0.5):
    """
    Purpose:
        Doing apriori minging on given data.
    Input:
        filename: String, the filename of data
        Support: float, the minimum support, default 0.5
    Output:
        fq_list: a two dimensions list of frozenset. Store the frequent itemsets.
        record: a dictionary. Key type is frozenset, value type is float.    
    """
    data, label = get_data(filename)
    pre_process(data)
    record = {}
    data = list(map(set, data))
    C1 = get_C1(data)
    fq_list = []
    fq, unfq = get_freqI(data, C1, support, record)
    fq_list.append(fq)
    while len(fq_list[-1]) != 0:
        Ck = get_Ck(fq_list[-1])
        ## right now eliminate is speend much more time than without it.
        #Ck = eliminate_infeq(Ck, unfq)
        fq, unfq = get_freqI(data, Ck, support, record)
        fq_list.append(fq)
    return fq_list, record


def count(fq_list, support):
    count = 0
    print("Support is set to be {}%".format(support * 100))
    for i in range(len(fq_list) - 1):
        count += len(fq_list[i])
        print("number of length-{} frequent itemsets: {}".format(i+1, len(fq_list[i])))
    print("number of all lengths frequent itemsets: {}".format(count))

if __name__ == "__main__":
	filename = sys.argv[1]
	support = float(sys.argv[2])
	fq_list, record = apriori(filename, support)
	dump_filename = "../data/support_" + str(int(support * 100)) +".p"
	pickle.dump((fq_list, record), open(dump_filename, "wb" ))
	count(fq_list, support)
	








