import numpy as np
import mylibrary as mylib
from mylibrary import NominalFeature
from collections import Counter

class NaiveBayesFactory:
    """
    Purpose:
        The factory to generate NaiveBayes machine.
    Initialized by :
        training data and label
    
    """
    def __init__(self, training_data, training_label):
        self.training_data = training_data
        self.training_label = training_label
    
    def scale_features(self, section_num):
        """
        Purpose:
            to discretize the training data.
            For continuous data, it will be divided to section_num sections.
            For nominal data, it will be labeled by the index of a member list.
        Input:
            section_num: int, numbers of discrete sections for continuours data.
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
                gate = np.linspace(old_min, old_max, num=section_num-1)
                gates.append(gate)
                new_row = np.zeros(row.shape)
                for j in range(len(gate)):
                    if j == 0:
                        new_row[row < gate[j]] = j
                    elif j == 1:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] <= row, row <= gate[j])]] = j
                    elif j < len(gate) - 1:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] < row, row <= gate[j])]] = j
                    else:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] < row, row <= gate[j])]] = j
                        new_row[row > gate[j]] = j+1
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
    
    def get_naiveBayes_machine(self, sect_num):
        """
        Purpose:
            To generate NaiveBayes_Machine object.
        Input:
            sect_num: int, numbers of discrete sections for continuours data.
        Output:
            NaiveBayes_Machine object.
        """
        data, gates, nominal_features = self.scale_features(sect_num)
        nominal_ids = [obj.col_id for obj in nominal_features]
        
        n = len(self.training_label)
        counter = Counter(self.training_label)
        n1 = counter[1]
        n0 = counter[0]
        p0 = n0 / n
        p1 = n1 / n
        features = []
        for i, row in enumerate(data.T):
            count = {}
            for j, member in enumerate(row):
                if member not in count:
                    count[member] = [0,0]
                if self.training_label[j] == 1:
                    count[member][1] += 1
                elif self.training_label[j] == 0:
                    count[member][0] += 1
            
            ##probability for out range data
            if i not in nominal_ids:
                if 0 not in count:   
                    count[0] = [0,0]
                if len(count) not in count:
                    count[len(count)] = [0,0]
                
            #Avoiding the Zero-Probability Problem
            new_n0 = n0 + len(count)
            new_n1 = n1 + len(count)
            for k, v in count.items():
                v[0] = (v[0] + 1) / new_n0
                v[1] = (v[1] + 1) / new_n1
                
            features.append(count)
            
        return NaiveBayes_Machine(p0, p1, features, gates, nominal_features)

class NaiveBayes_Machine:
    """
    Purpose:
        To generate objects to do NaiveBayes classification task.
    Initialized by:
        p0: real, probability of negative samples in training data.
        p1: real, probability of postive samples in training data.
        features_stat: a list of dictionary. features_stat[0] saves the numbers of the positve 
                       and the negaive labels for each members in column 0 feature.
        gates: save the gates using to discretize continuous features.
        nominal_features: a list of NominalFeature which save nominal features's colum id and members
    
    """
    def __init__(self, p0, p1, features_stat, gates, nominal_features):
        self.p0 = p0
        self.p1 = p1
        self.features_stat = features_stat.copy()
        self.gates = gates.copy()
        self.nominal_features = nominal_features.copy()
        
    def preprocess(self, the_data):
        """
        Purpose:
            Use the gates and nominal_features to discretize test data.
        """
        data_t = []
        nominal_ids = [obj.col_id for obj in self.nominal_features]
        nominal_loc = 0
        
        for i, row in enumerate(the_data.T):
            if i not in nominal_ids:
                row = row.astype(float)
                new_row = np.zeros(row.shape)
                gate = self.gates[i]
                for j in range(len(gate)):
                    if j == 0:
                        new_row[row < gate[j]] = j
                    elif j == 1:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] <= row, row <= gate[j])]] = j
                    elif j < len(gate) - 1:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] < row, row <= gate[j])]] = j
                    else:
                        new_row[[x[0] and x[1] for x in zip(gate[j-1] < row, row <= gate[j])]] = j
                        new_row[row > gate[j]] = j+1
                data_t.append(new_row)
            else:
                nominal = self.nominal_features[nominal_loc]
                new_row = np.asarray([nominal.members.index(x) for x in row])
                data_t.append(new_row)
                nominal_loc += 1
                
        return np.asarray(data_t).T   
    
    def predict(self, test_data):
        """
        Purpose:
            to classify the test_data using bayes rule.
        """
        data_test = self.preprocess(test_data)
        res_label = []
        for row in data_test:
            p_x_given_0 = 1
            p_x_given_1 = 1
            for i, feature in enumerate(row):
                p_x_given_0 *= self.features_stat[i][feature][0]
                p_x_given_1 *= self.features_stat[i][feature][1]
            check = np.asarray([self.p0 * p_x_given_0, self.p1 * p_x_given_1])
            res_label.append(np.argmax(check))
            
        return np.asarray(res_label)

def show_res(raw_set, n, sec_num):
    
    for i in range(n):
        training_set, test_set = mylib.n_fold(n ,i, raw_set)
        training_data, training_label = mylib.get_data_label(training_set)
        factory = NaiveBayesFactory(training_data, training_label)
        bayes_5 = factory.get_naiveBayes_machine(sec_num)
        test_data, test_label = mylib.get_data_label(test_set)
        res_label = bayes_5.predict(test_data)
        confusion = mylib.confusion_matrix(test_label, res_label)
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
    sec_num = 5
    print("***************project3_dataset1*****************")
    raw_set= mylib.get_set("../data/project3_dataset1.txt")
    show_res(raw_set, n, sec_num)
    print("\n\n***************project3_dataset2*****************")
    raw_set= mylib.get_set("../data/project3_dataset2.txt")
    show_res(raw_set, n, sec_num)