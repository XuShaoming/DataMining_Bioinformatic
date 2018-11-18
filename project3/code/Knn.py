import numpy as np
from io import StringIO
import mylibrary as mylib
from mylibrary import NominalFeature
import heapq
from collections import Counter
        
class KnnFactory:
    
    def __init__(self, training_data, training_label):
        self.training_data = training_data
        self.training_label = training_label
    
    def get_KnnMachine(self, k, new_min, new_max):
        old_min_max_list = []
        NominalFeatures = []
        data_t = []
        new_range = new_max - new_min
        
        for i, row in enumerate(self.training_data.T):
            try:
                float(row[0])
                row = row.astype(float)
                old_min = np.min(row)
                old_max = np.max(row)
                old_min_max_list.append((old_min,old_max))
                old_range = old_max - old_min
                data_t.append((((row - old_min) * new_range) / old_range) + new_min)
            except ValueError:
                members = list(set(row))
                NominalFeatures.append(NominalFeature(i, members))
                new_row = np.asarray([members.index(x) for x in row])
                old_min = 0
                old_max = len(members) - 1
                old_min_max_list.append((old_min,old_max))
                old_range = old_max - old_min
                data_t.append((((new_row - old_min) * new_range) / old_range) + new_min)
                
        return KnnMachine(k, np.asarray(data_t).T, self.training_label, new_min, new_max, np.asarray(old_min_max_list), NominalFeatures)

class KnnMachine:
    def __init__(self, k, data, label, new_min, new_max, min_max_arr, nominal_features):
        self.k = k
        self.training_data = data
        self.training_label = label
        self.min_max_arr = min_max_arr.copy()
        self.new_min = new_min
        self.new_max = new_max
        self.nominal_features = nominal_features.copy()
    
    def preprocess(self, test_data):
        data_t = []
        new_range = self.new_max - self.new_min
        j = 0
        
        for i, row in enumerate(test_data.T):
            try:
                float(row[0])
                row = row.astype(float)
                old_min = self.min_max_arr[i][0]
                old_max = self.min_max_arr[i][1]
                old_range = old_max - old_min
                data_t.append((((row - old_min) * new_range) / old_range) + self.new_min)
            except ValueError:
                the_nominal = self.nominal_features[j]
                members = the_nominal.members
                new_row = np.asarray([members.index(x) for x in row])
                old_min = self.min_max_arr[i][0]
                old_max = self.min_max_arr[i][1]
                old_range = old_max - old_min
                data_t.append((((new_row - old_min) * new_range) / old_range) + self.new_min)
                j += 1
        return np.asarray(data_t).T
 

    def euclidean_distance(self, pt1, pt2):
        if self.nominal_features != None and len(self.nominal_features) > 0:
            cols_id = np.asarray([obj.col_id for obj in self.nominal_features])
            total = 0
            j = 0
            for i, val in enumerate(pt1):
                if j < len(cols_id) and i == cols_id[j]:
                    if pt1[i] != pt2[i]:
                        total += 1
                    j += 1
                else:
                    total += np.square(pt1[i] - pt2[i])
            return np.sqrt(total)
        return np.sqrt(np.sum(np.square(pt1 - pt2)))

    def predict(self, test_data):
        data_test = self.preprocess(test_data)
        label_res = []
        for row in data_test:
            heap = []
            for i, check in enumerate(self.training_data):
                distance = self.euclidean_distance(row, check)
                heapq.heappush(heap, Record(self.training_label[i], distance, i))
                if(len(heap) > self.k):
                    heapq.heappop(heap)
            vote_list = [obj.label_id for obj in heap]
            label_res.append(max(vote_list, key=Counter(vote_list).get))
            #label_res.append(Counter(vote_list).most_common(1)[0][0])
        return np.asarray(label_res)

class Record:
    """
    Purpose:
        Use to save the label_id of the nearest neighbor of given data entry.
        self.label_id save a neighbor's label
        self.distance save the distance between the neighbor and the data entry
        self.row_id save the neighbor's row_id in training data. For debug purpose
    """
    def __init__(self, label_id, distance, row_id):
        self.label_id = label_id
        self.distance = distance
        self.row_id = row_id  #for debug
    def __lt__(self, other):
        return self.distance - other.distance > 0
    def __repr__(self):
        return "row_id: " + str(self.row_id) + "  distance: " + str(self.distance)


def show_res(raw_set, n, k, new_min, new_max):
    for i in range(n):
        training_set, test_set = mylib.n_fold(n ,i, raw_set)
        training_data, training_label = mylib.get_data_label(training_set)
        factory = KnnFactory(training_data, training_label)
        knn_machine_5 = factory.get_KnnMachine(k,new_min,new_max)
        test_data, test_label = mylib.get_data_label(test_set)
        res_label = knn_machine_5.predict(test_data)
        confusion = mylib.confusion_matrix(test_label, res_label)
        accuracy = mylib.get_accuracy(confusion)
        precision = mylib.get_precision(confusion)
        recall = mylib.get_recall(confusion)
        f1_score = mylib.get_f1_score(confusion)
        print("***itr: ", i," ***")
        print("confusion matrix:")
        print(confusion)
        print("accuracy: ", accuracy)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f1_score: ", f1_score)

if __name__ == "__main__":
	n=10
	k=5
	new_min = 0
	new_max = 1

	print("***************project3_dataset1*****************")
	raw_set= mylib.get_set("../data/project3_dataset1.txt")
	show_res(raw_set, n, k, new_min, new_max)
	print("\n\n***************project3_dataset2*****************")
	raw_set= mylib.get_set("../data/project3_dataset2.txt")
	show_res(raw_set, n, k, new_min, new_max)




