import numpy as np
import mylibrary as mylib
from mylibrary import NominalFeature
from collections import Counter

class LeafNode:
    def __init__(self, label):
        self.label = label
        
class BodyNode:
    def __init__(self, feature_id, children):
        self.feature_id = feature_id
        self.children = children
        
    def add_child(self, name, child):
        self.children[name] = child
    def set_id(self, f_id):
        self.feature_id = f_id

class TreeFactory:
    """
    Purpose:
        The factory to generate Decision tree machine.
    Initialized by :
        training data and label
    """
    
    def __init__(self, training_data, training_label):
        self.training_data = training_data
        self.training_label = training_label
    
    def scale_features(self, gate_num):
        """
        Purpose:
            to discretize the training data.
            For continuous data, it will be divided to labels range from 1 to gate_num.
            For nominal data, it will be labeled by the index of a member range from 0 to len(members)-1
        Input:
            gate_num: int, numbers of gates for continuours data.
        Ouput:
            The normized data
            gates: a list of gates for all features.
            NominalFeatures: a list of NominalFeature which save nominal features's colum id and members
        """
        gates = []
        NominalFeatures = []
        data_t = []
        for i, row in enumerate(self.training_data.T):
            try:
                float(row[0])
                row = row.astype(float)
                old_min = np.min(row)
                old_max = np.max(row)
                gate = np.linspace(old_min, old_max, num=gate_num + 1)
                gates.append(gate)
                new_row = np.zeros(row.shape)
                for j in range(1, len(gate) - 1):
                    if j == 1:
                        new_row[row < gate[j]] = j
                    elif j < len(gate) - 2:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] <= row, row < gate[j])]] = j
                    else:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] <= row, row < gate[j])]] = j
                        new_row[row >= gate[j]] = j+1
                data_t.append(new_row)
            except ValueError:
                members = list(set(row))
                NominalFeatures.append(NominalFeature(i, members))
                new_row = np.asarray([members.index(x) for x in row])
                old_min = 0
                old_max = len(members) - 1
                gate = np.linspace(old_min, old_max, num=len(members))
                gates.append(gate)
                data_t.append(new_row)
                
        return np.asarray(data_t).T, gates, NominalFeatures
        
    def best_split(self, check_data, check_label, impurity_fun):
        """
        Purpose:
            find the the feature that best split data by get maximum gain.
        Input:
            check_data: real, 2-D numpy array
            check_label: real, 1-D numpy array.
            impurity_fun: function, the function to measure impurity, 
                          including error, entropy, gini. 
        Output: the column id of check_data which label the best feature.
        """
        count = Counter(check_label)
        parent_stat = [count[-1], count[1]]
        gains = []
        for row in check_data.T:
            stat = {}
            for i, member in enumerate(row):
                if member not in stat:
                    stat[member] = [0,0]
                if check_label[i] == -1:
                    stat[member][0] += 1
                else:
                    stat[member][1] += 1
            gains.append(mylib.gain(parent_stat ,list(stat.values()), impurity_fun))
        gains = np.asarray(gains)
        
        return np.argmax(gains)
    
    
    def create_node(self, check_data, check_label):
        """
        Purpose:
            create Tree node. It will retun LeafNode or BodyNode.
        """
        label_set = set(check_label)
        #all training samples have same label
        if len(label_set) == 1:
            return LeafNode(check_label[0])
        # identical features
        if check_data.shape[1] == 1:
            return LeafNode(max(check_label, key=Counter(check_label).get))
        
        return BodyNode(-1, dict())
    
    def get_children(self, check_data, feature_id):
        """
        Purpose:
            Get the children and their corresponding samples id(row id) from the best feature.
        Input:
            check_data: 2D numpy array.
            feature_id: the id of the feature which best split check_data.
        Output:
            member_dict: dictionary. keys are children names. values are rows id.
        """
        feature = check_data.T[feature_id]
        members = set(feature)
        member_dict = dict()
        for member in members:
            member_dict[member] = np.where(feature == member)[0] 
        return member_dict
        
    def tree_growth(self, data, label,Es, Fs, gate_num, impurity_fun, sub_space_fun, rand):
        """
        Purpose:
            The main function that recursively build the decision tree.
        Input:
            data: 2-D numpy array, the normized data
            label: 1-D numpy array, the normized label. Negative:-1, Postive: +1
            Es: The rows indexs of data, indexs for samples.
            Fs: The colums indexs of data, indexs for features.
            gate_num: int, numbers of gates for continuours data. We will use it to do majority
                      vote for unseen samples.
            impurity_fun: function, the function to measure impurity.
            sub_space_fun: function, the function to determine the percentage of features used 
                           for build tree. 
            rand: numpy random object with seed.
        Output:
            node: the root node, which usually is BodyNode.
        """
        Es_list = np.asarray(list(Es))
        Fs_list = np.asarray(list(Fs))
        #chose a subset of features 
        sub_Fs_num = sub_space_fun(len(Fs_list))
        Fs_list = Fs_list[rand.permutation(len(Fs_list))[0:sub_Fs_num]]
        
        check_data = data[Es_list, :][:, Fs_list]
        check_label = label[Es_list]
        node = self.create_node(check_data, check_label)
        if type(node) is LeafNode:
            return node
        else:
            feature_virtual_id = self.best_split(check_data, check_label, impurity_fun)
            feature_actual_id = Fs_list[feature_virtual_id]
            node.set_id(feature_actual_id)
            new_Fs = Fs - set([feature_actual_id])
            children_dict = self.get_children(check_data, feature_virtual_id)
            
            #majority vote for unseen samples
            feature_members = set(data[:, feature_actual_id])
            # not nominal feature and the the feature members not completed
            if min(list(feature_members)) != 0 and len(feature_members) != gate_num:
                feature_members = set(range(1, gate_num + 1))
            c_feature_members = set(children_dict.keys())
            uncovered_members = feature_members - c_feature_members
            guess_label = max(check_label, key=Counter(check_label).get)
            for name in uncovered_members:
                node.add_child(name, LeafNode(guess_label))
    
            del check_data
            del check_label
            for name, Evs_vitual_list in children_dict.items():
                Evs = set(Es_list[Evs_vitual_list])
                child = self.tree_growth(data, label, Evs, new_Fs, gate_num, impurity_fun, sub_space_fun, rand)
                node.add_child(name, child)
        return node
    
    def get_DT_machine(self, gate_num, impurity_fun, sub_space_fun, seed):
        """
        Purpose:
            To build DT_machine object. 
        Input:
            You can see parameters description in tree_growth function.
        Output:
            A DT_machine object.
        """
        data, gates, nominal_features = self.scale_features(gate_num)
        label = mylib.convert_label(self.training_label)
        rand = np.random.RandomState(seed)
        Es = set(range(data.shape[0]))
        Fs = set(range(data.shape[1]))
        root = self.tree_growth(data, label, Es, Fs, gate_num, impurity_fun, sub_space_fun, rand)
        
        return DT_machine(root, gates, nominal_features)


class DT_machine:
    """
    Purpose:
        The factory to generate Decision tree machine.
    Initialized by :
        root: [BodyNode(likely) or LeafNode] the root of the decision tree.
        gates: a list of gates for all features of training data. used to normalized test data.
        nominal_features: a list of NominalFeature of training data. used to normalized test data.                    
    """
    def __init__(self, root, gates, nominal_features):
        self.root = root
        self.gates = gates.copy()
        self.nominal_features = nominal_features.copy()
    
    def preprocess(self, the_data):
        """
        Purpose:
            use gates and nominal_features to normalize test data.
        Input: 
            the_data: test data.
        Output:
            normalized test data.
        """
        data_t = []
        nominal_ids = [obj.col_id for obj in self.nominal_features]
        nominal_loc = 0
        
        for i, row in enumerate(the_data.T):
            if i not in nominal_ids:
                row = row.astype(float)
                new_row = np.zeros(row.shape)
                gate = self.gates[i]
                for j in range(1, len(gate) - 1):
                    if j == 1:
                        new_row[row < gate[j]] = j
                    elif j < len(gate) - 2:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] <= row, row < gate[j])]] = j
                    else:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] <= row, row < gate[j])]] = j
                        new_row[row >= gate[j]] = j+1
                data_t.append(new_row)
            else:
                nominal = self.nominal_features[nominal_loc]
                new_row = np.asarray([nominal.members.index(x) for x in row])
                data_t.append(new_row)
                nominal_loc += 1
                
        return np.asarray(data_t).T
    
    def predict_aux(self, node, entry):
        """
        the aux funcion to swim down in decision tree. Here I use normal recusion.
        Actually, the tail recursion can be more efficient. But it seems python3 not 
        support tail recursion.
        """
        if type(node) is LeafNode:
            return node.label
        return self.predict_aux(node.children[entry[node.feature_id]], entry)
    
    def predict(self, test_data):
        """
        Purpose:
            to classify test_data
        """
        data_test = self.preprocess(test_data)
        res_label = []
        for row in data_test:
            res_label.append(self.predict_aux(self.root, row))
        return np.asarray(res_label)


def sub_p(method=1):
    """
    Purpose:
        The high order function to return sub_space_fun for decision tree construction.
    recommend:
        method_1 for single Tree
        method_2 for Classification problem
        method_3 for regression problem
        from wiki https://en.wikipedia.org/wiki/Random_forest
    """
    def method_1(p):
        return p
    def method_2(p):
        return int(np.log2(p))
    def method_3(p):
        return int(p/3)
    
    if method == 2:
        return method_2
    elif method == 3:
        return method_3
    else:
        return method_1


def show_res(raw_set, n, gate_num, sub_space_fun, seed):
    
    for i in range(n):
        training_set, test_set = mylib.n_fold(n ,i, raw_set)
        training_data, training_label = mylib.get_data_label(training_set)
        factory = TreeFactory(training_data, training_label)
        
        dtree = factory.get_DT_machine(gate_num, mylib.entropy, sub_space_fun, seed)
        test_data, test_label = mylib.get_data_label(test_set)
        true_label = mylib.convert_label(test_label)
        res_label = dtree.predict(test_data)
        confusion = mylib.confusion_matrix(true_label, res_label)
        accuracy = mylib.get_accuracy(confusion)
        precision = mylib.get_precision(confusion)
        recall = mylib.get_recall(confusion)
        f1_score = mylib.get_f1_score(confusion)
        print("**************itr: ", i," **************")
        print("confusion matrix:")
        print(confusion)
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1_score: ", f1_score)

if __name__ == "__main__":
    n = 10
    branch_num = 4
    sub_space_fun = sub_p(2)
    seed = 20
    print("***************project3_dataset1*****************")
    raw_set= mylib.get_set("../data/project3_dataset1.txt")
    show_res(raw_set, n, branch_num, sub_space_fun, seed)
    print("\n\n***************project3_dataset2*****************")
    raw_set= mylib.get_set("../data/project3_dataset2.txt")
    show_res(raw_set, n, branch_num, sub_space_fun, seed)

