#!/usr/bin/python3

import sys
import numpy as np
from io import StringIO
from collections import deque
from apriori import get_C1
from apriori import get_Ck
import pickle


def rule_gen(itemset, record, min_conf, res):
    """
    Purpose:
        generate the association rule from the given itemset
    Input:
        itemset: frozenset of itemset
        record: dictionary, Key is the frozenset of itemsets , value is the support
        min_conf: float, the minimum confidence
        res: dictionary, Key is the frozenset of itemsets , value is the confidence
    Output:
        res: dictionary, Key is the frozenset of itemsets , value is the confidence
    """
    queue = deque()
    queue.append(itemset)
    while len(queue) != 0:
        item = queue.popleft()
        conf = record[itemset] / record[item]
        if  conf >= min_conf:
            res[(item, itemset - item)] = conf
            if(len(item) > 1):
                c1 = [frozenset([elem]) for elem in item]
                for elem in c1:
                    queue.append(item - elem)
    res.pop((itemset, frozenset()))
    return res


def get_rule(combinations, min_conf, record, res):
    """
    Purpose:
        get the association rule of all the itemsets from the combinations list. And
        save the rule in the res dictionray.
    Input:
        combinations: a list of frozenset of itemsets. 
        min_conf: float, the minimum confidence
        record: dictionary, Key is the frozenset of itemsets , value is the support
        res: dictionary, Key is the frozenset of itemsets , value is the confidence 
    Output:
        None
    """
    for row in combinations:
        for elem in row:
            rule_gen(elem, record, min_conf, res)



def item_combinations(items):
    """
    Purpose:
        get all the combinations of items from the list.
    Input:
        items: list of string. Each element in list represent an itemset
    Output:
        res[:-1]: a list of frozenset. 
    """
    res = []
    res.append(get_C1([items]))
    while len(res[-1]) != 0 :
        Ck = get_Ck([i for i in res[-1]])
        res.append(Ck)
    return res[:-1]


def combination_any(items, fq_list):
    """
    Input:
        items: list of string. Each element in list represent an itemset
        fq_list: a two dimension list of the frozenset of itemsets.
    Output:
        combinations: a list of frozenset. 
    """
    combinations = []
    itemset = frozenset(items)
    for row in fq_list[1:-1]:
        combinations.append([])
        for item in row:
            if len(item) != len(item - itemset):
                combinations[-1].append(item)
    return combinations


def combination_none(items, fq_list):
    """
    Input:
        items: list of string. Each element in list represent an itemset
        fq_list: a two dimension list of the frozenset of itemsets.
    Output:
        combinations: a list of frozenset. 
    """
    combinations = []
    itemset = frozenset(items)
    for row in fq_list[1:-1]:
        combinations.append([])
        for item in row:
            if len(item) == len(item - itemset):
                combinations[-1].append(item)
    return combinations



def combination_num(cmd_2, items, fq_list):
    """
    Input:
        cmd_2: int, target number
        items: list of string. Each element in list represent an itemset
        fq_list: a two dimension list of the frozenset of itemsets.
    Output:
        combinations: a list of frozenset. 
    """
    layer = int(cmd_2)
    combinations = []
    itemsets = item_combinations(items)[layer - 1]
    non_itemsets = [(frozenset(items) - item) for item in itemsets]
    for item in non_itemsets:
        sub_all = combination_none([i for i in item], fq_list)
        check = frozenset(items) - item
        row = []
        for i in sub_all:
            for j in i:
                if check.issubset(j):
                    row.append(j)
        combinations.append(row)
    return combinations



def head_body_num_dic(cmd_1, cmd_2, items, rule_any_dic):
    """
    Purpose:
        use the rule_any_dic to generated the association rules which has limited 
        number on HEAD or BODY
    Input:
        cmd_1: String, command 1
        cmd_2: String or int, used to limited number of itemsets in HEAD or BODY
        items: list of string. Each element in list represent an itemset
        rule_any_dic: dictionary, Key is the frozenset of itemsets , value is 
                      the confidence. 
    Output:
        res: dictionary, Key is the frozenset of itemsets , value is the confidence.
    """
    res = {}
    index = 0
    num = int(cmd_2)
    if cmd_1 == "HEAD": index = 0
    elif cmd_1 == "BODY": index = 1
    check = frozenset(items)
    for k, v in rule_any_dic.items():
        if len(check - k[index]) == len(check) - num :
            res[k] = v
    return res;


def head_body_any_dic(cmd_1, items, rule_any_dic):
    """
    Purpose:
        use the rule_any_dic to generated the association rules which has any number of 
        items on HEAD or BODY
    """
    res = {}
    index = 0
    if cmd_1 == "HEAD": index = 0
    elif cmd_1 == "BODY": index = 1
    check = frozenset(items)
    for k, v in rule_any_dic.items():
        if len(k[index] - check) != len(k[index]):
            res[k] = v
    return res;



def head_body_none_from_rule_any(cmd_1, items, rule_any_dic):
    """
    Purpose:
        use the rule_any_dic to generated the association rules which has none of 
        items on HEAD or BODY
    """
    res = {}
    index = 0
    if cmd_1 == "HEAD": index = 0
    elif cmd_1 == "BODY": index = 1
    check = frozenset(items)
    for k, v in rule_any_dic.items():
        if len(k[index] - check) == len(k[index]):
            res[k] = v
    return res;



def print_dic(dic):
    for k, v in dic.items():
        print(k[0],"-->",k[1]," : ", v)


def template1(cmd_1, cmd_2, items, min_conf, record, fq_list):
    """
    Purpose:
        mining association rules by given query from template1
    Input :
        cmd_1: string
        cmd_2: string
        items: list of string. Each element in list represent an itemset
        min_conf: float, the minimum confidence
        record: dictionary, Key is the frozenset of itemsets , value is the support
        fq_list: a two dimension list of the frozenset of itemsets.
    Output :
        res: dictionary, Key is the frozenset of itemsets , value is the confidence
    """
    
    res = {}
    if cmd_1 == "RULE":
        if cmd_2 == "ANY":
            combinations = combination_any(items, fq_list)
            get_rule(combinations, min_conf, record, res)
        elif cmd_2 == "NONE":
            combinations = combination_none(items, fq_list)
            get_rule(combinations, min_conf, record, res)
        else:
            combinations = combination_num(cmd_2, items, fq_list)
            get_rule(combinations, min_conf, record, res)
    elif cmd_1 == "HEAD":
        if cmd_2 == "ANY":
            rule_any_dic = template1("RULE", "ANY", items, min_conf, record, fq_list)
            res = head_body_any_dic(cmd_1, items, rule_any_dic)
        elif cmd_2 == "NONE" : 
            rule_any_dic = template1("RULE", "ANY", items, min_conf, record, fq_list)
            rule_none_dic = template1("RULE", "NONE", items, min_conf, record, fq_list)
            dic_1 = head_body_none_from_rule_any(cmd_1, items, rule_any_dic)
            res = {**dic_1, **rule_none_dic}
        else :
            rule_any_dic = template1("RULE", "ANY", items, min_conf, record, fq_list)
            res = head_body_num_dic(cmd_1, cmd_2, items, rule_any_dic)        
    elif cmd_1 == "BODY" :
        if cmd_2 == "ANY" :
            rule_any_dic = template1("RULE", "ANY", items, min_conf, record, fq_list)
            res = head_body_any_dic(cmd_1, items, rule_any_dic)
        elif cmd_2 == "NONE" : 
            rule_any_dic = template1("RULE", "ANY", items, min_conf, record, fq_list)
            rule_none_dic = template1("RULE", "NONE", items, min_conf, record, fq_list)
            dic_1 = head_body_none_from_rule_any(cmd_1, items, rule_any_dic)
            res = {**dic_1, **rule_none_dic}
        else :
            rule_any_dic = template1("RULE", "ANY", items, min_conf, record, fq_list)
            res = head_body_num_dic(cmd_1, cmd_2, items, rule_any_dic)
    
    return res



def template2(cmd_1, cmd_2, min_conf, record, fq_list):
    res = {}
    num = int(cmd_2)
    combinations_list = []
    for item in fq_list[num - 1]:
        combinations_list.append([i for i in item])
    if cmd_1 == "RULE":
        for combinations in combinations_list:
            res = {**res, **template1(cmd_1, cmd_2, combinations, min_conf, record, fq_list)}
    elif cmd_1 == "HEAD":
        for combinations in combinations_list:
            res = {**res, **template1(cmd_1, cmd_2, combinations, min_conf, record, fq_list)}
    elif cmd_1 == "BODY":
        for combinations in combinations_list:
            res = {**res, **template1(cmd_1, cmd_2, combinations, min_conf, record, fq_list)}
    else:
        print("RULE|HEAD|BODY")
    return res


def template3(cmd, *cmds):
    res = {}
    if cmd == "1or1" or cmd == "1and1":
        cmd1_1, cmd1_2, items1, cmd2_1, cmd2_2, items2, min_conf, record, fq_list = cmds
        dic1 = template1(cmd1_1, cmd1_2, items1, min_conf, record, fq_list)
        dic2 = template1(cmd2_1, cmd2_2, items2, min_conf, record, fq_list)
        if cmd == "1or1":
            res ={**dic1, **dic2}
        else :
            for k, v in dic1.items():
                if k in dic2:
                    res[k] = v
    elif cmd == "1or2" or cmd == "1and2":
        cmd1_1, cmd1_2, items1, cmd2_1, cmd2_2, min_conf, record, fq_list = cmds
        dic1 = template1(cmd1_1, cmd1_2, items1, min_conf, record, fq_list)
        dic2 = template2(cmd2_1, cmd2_2, min_conf, record, fq_list)
        if cmd == "1or2":
            res = {**dic1, **dic2}
        else :
            for k, v in dic1.items():
                if k in dic2:
                    res[k] = v
    elif cmd == "2or2" or cmd == "2and2" :
        cmd1_1, cmd1_2, cmd2_1, cmd2_2, min_conf, record, fq_list = cmds
        dic1 = template2(cmd1_1, cmd1_2, min_conf, record, fq_list)
        dic2 = template2(cmd2_1, cmd2_2, min_conf, record, fq_list)
        if cmd == "2or2":
            res = {**dic1, **dic2}
        else :
            for k, v in dic1.items():
                if k in dic2:
                    res[k] = v
    return res



if __name__ == "__main__":
	filename = sys.argv[1]
	confidence = float(sys.argv[2])

	fq_list, record = pickle.load( open(filename, "rb" ) )

	print("template1 result:")

	result11 = template1("RULE", "ANY", ['G59_Up'], confidence, record, fq_list)
	print("result11 : ", len(result11))

	result12 = template1("RULE", "NONE", ['G59_Up'], confidence,record, fq_list)
	print("result12 : ", len(result12))

	result13 = template1("RULE", 1, ['G59_Up', 'G10_Down'], confidence, record, fq_list)
	print("result13 : ",len(result13))

	result14 = template1("HEAD", "ANY", ['G59_Up'], confidence,record, fq_list)
	print("result14 : ",len(result14))

	result15 = template1("HEAD", "NONE", ['G59_Up'], confidence,record, fq_list)
	print("result15 : ", len(result15))

	result16 = template1("HEAD", "1", ['G59_Up', 'G10_Down'], confidence,record, fq_list)
	print("result16 : ", len(result16))

	result17 = template1("BODY", "ANY", ['G59_Up'], confidence,record, fq_list)
	print("result17 : ", len(result17))

	result18 = template1("BODY", "NONE", ['G59_Up'], confidence,record, fq_list)
	print("result18 : ", len(result18))

	result19 = template1("BODY", "1", ['G59_Up', 'G10_Down'], confidence,record, fq_list)
	print("result19 : ", len(result19))


	print("******************")
	print("template2 result:")

	result21 = template2("RULE", 3, confidence,record, fq_list)
	print("result21 : ", len(result21))

	result22 = template2("HEAD", 2, confidence,record, fq_list)
	print("result22 : ", len(result22))

	result23 = template2("BODY", 1, confidence,record, fq_list)
	print("result23 : ", len(result23))


	print("******************")
	print("template3 result:")

	result31 = template3("1or1", "HEAD", "ANY", ['G10_Down'], "BODY", 1, ['G59_Up'], confidence,record, fq_list)
	print("result31 : ", len(result31))

	result32 = template3("1and1", "HEAD", "ANY", ['G10_Down'], "BODY", 1, ['G59_Up'], confidence,record, fq_list)
	print("result32 : ", len(result32))

	result33 = template3("1or2", "HEAD", "ANY", ['G10_Down'], "BODY", 2, confidence,record, fq_list)
	print("result33 : ", len(result33))

	result34 = template3("1and2", "HEAD", "ANY", ['G10_Down'], "BODY", 2, confidence,record, fq_list)
	print("result34 : ", len(result34))

	result35 = template3("2or2", "HEAD", 1, "BODY", 2, confidence,record, fq_list)
	print("result35 : ", len(result35))

	result36 = template3("2and2", "HEAD", 1, "BODY", 2, confidence,record, fq_list)
	print("result36 : ", len(result36))




















