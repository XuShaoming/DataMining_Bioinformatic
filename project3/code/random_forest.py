import numpy as np
import mylibrary as mylib
import decision_tree as DT
from collections import Counter

class ForestFactory:
    """
    Purpose:
        The factory to generate Random forest machine.
    Initialized by :
        training data and label
    """
    def __init__(self, training_data, training_label):
        self.training_data = training_data
        self.training_label = training_label
    
    def get_RF(self, k, branch_num=4, impurity_fun=mylib.entropy, sub_space_fun=DT.sub_p(method=2), seed=20):
        """
        Purpose:
            get random RF_machine object.
        Input:
            k: int, the number of trees in forest.
            branch_num: int, the numbers of branch for continuous features in decision tree.
            impurity_fun: function, the function to measure impurity, 
                          including error, entropy, gini. 
            sub_space_fun: function, the function to determine the percentage of features used 
                           for build tree. 
            seed: int, use to get random object.
        
        Output:
            a RF_machine object.
        """
        rand = np.random.RandomState(seed)
        n,d = self.training_data.shape
        
        forest = []
        for i in range(k):
            index = np.asarray([rand.randint(n) for i in range(n)])
            data = self.training_data[index, :]
            label = self.training_label[index]
            factory = DT.TreeFactory(data, label)
            forest.append(factory.get_DT_machine(branch_num, impurity_fun, sub_space_fun, seed))  
        return RF_machine(forest)


class RF_machine:
    """
    Purpose:
        Generate object to do random forest classification..
    Initialized by :
        forest: a list of decision trees.
    """
    def __init__(self, forest):
        self.forest = forest.copy()
    
    def predict(self, test_data):
        """
        Purpose:
            To classify the test_data.
            Here, I use the majority vote trees as the label.
        """
        labels = []
        for tree in self.forest:
            labels.append(tree.predict(test_data))
        labels = np.asarray(labels)
        res = []
        for row in labels.T:
            res.append(max(row, key=Counter(row).get))
        return np.asarray(res)


def show_res(raw_set, n, k, branch_num, impurity_fun, sub_space_fun, seed):
    
    for i in range(n):
        training_set, test_set = mylib.n_fold(n ,i, raw_set)
        training_data, training_label = mylib.get_data_label(training_set)
        factory = ForestFactory(training_data, training_label)
        randForest = factory.get_RF(k, branch_num, impurity_fun, sub_space_fun, seed)
        test_data, test_label = mylib.get_data_label(test_set)
        true_label = mylib.convert_label(test_label)
        res_label = randForest.predict(test_data)
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
    k = 30
    branch_num = 4
    impurity_fun = mylib.entropy
    sub_space_fun = DT.sub_p(method=2)
    seed = 20

    print("***************project3_dataset1*****************")
    raw_set= mylib.get_set("../data/project3_dataset1.txt")
    show_res(raw_set, n, k, branch_num, impurity_fun, sub_space_fun, seed)
    print("\n\n\n***************project3_dataset2*****************")
    raw_set= mylib.get_set("../data/project3_dataset2.txt")
    show_res(raw_set, n, k, branch_num, impurity_fun, sub_space_fun, seed)



