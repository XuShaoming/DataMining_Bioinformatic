import numpy as np
import mylibrary as mylib
import decision_tree as DT

class AdaFactory:
    
    def __init__(self, training_data, training_label):
        self.training_data = training_data
        self.training_label = training_label
    
    def get_AdaMachine(self, k, branch_num=4, impurity_fun=mylib.entropy, sub_space_fun=DT.sub_p(method=2), seed=20):
        rand = np.random.RandomState(seed)
        n,d = self.training_data.shape
        true_label = mylib.convert_label(self.training_label)
        vec = np.arange(n)
        weights = np.asarray([1/n for i in range(n)])
        classifiers = []
        importances = []
        i = 0
        while i < k:
            index = rand.choice(vec, n, replace=True, p=weights)
            data = self.training_data[index, :]
            label = self.training_label[index]
            factory = DT.TreeFactory(data, label)
            dtree = factory.get_DT_machine(branch_num, impurity_fun, sub_space_fun, seed)
            res_label = dtree.predict(self.training_data)
            error_vect = res_label != true_label
            error = np.sum(weights[error_vect])
            if error > 0.5:
                print("error greater than 0.5")
                #reset weights and go back the head of loop
                weights = np.asarray([1/n for i in range(n)])
                continue
            
            ai = 1/2 * np.log((1-error) / error)
            
            #update weights
            X1 =  weights * np.exp(-ai * true_label * res_label)
            norm = np.sum(X1)
            weights = X1 / norm
            
            classifiers.append(dtree)
            importances.append(ai)
            i += 1
        
        return AdaMachine(classifiers, np.asarray(importances))

class AdaMachine:
    def __init__(self, classifiers, importances):
        self.classifiers = classifiers.copy()
        self.importances = importances.copy()
    
    def predict(self, test_data):
        labels = []
        for classifier in self.classifiers:
            labels.append(classifier.predict(test_data))
        labels = np.asarray(labels)
        res = []
        for row in labels.T:
            res.append(mylib.sign(np.sum(row * self.importances)))
        return np.asarray(res)

def show_res(raw_set, n, k, branch_num, impurity_fun, sub_space_fun, seed):
    
    for i in range(n):
        training_set, test_set = mylib.n_fold(n ,i, raw_set)
        training_data, training_label = mylib.get_data_label(training_set)
        factory = AdaFactory(training_data, training_label)
        adaForest = factory.get_AdaMachine(k, branch_num, impurity_fun, sub_space_fun, seed)
        test_data, test_label = mylib.get_data_label(test_set)
        true_label = mylib.convert_label(test_label)
        res_label = adaForest.predict(test_data)
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