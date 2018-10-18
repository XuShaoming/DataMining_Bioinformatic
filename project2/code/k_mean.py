import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mylibrary as mylib
from mylibrary import euclidean_distance

def get_centers_bank(data, k, seed=41):
    """
    Purpose:
        recore and return the results of each step of k-mean algorithm
    Input:
        data: a two dimension matrix
        k: int, number of clusters
        seed: seed for random number generator
    """
    init_centers = data[np.random.RandomState(seed=seed).permutation(data.shape[0])[0:k]]
    centers_bank = [[[center, np.array([])] for center in init_centers]]
    itr = 1
    while True:
        if(itr % 5 == 0):
            print("itr:", itr)
        dis_matrix = np.empty((0,data.shape[0]))
        for row in centers_bank[-1]:
            dis_matrix = np.vstack((dis_matrix, np.sum(np.square(data - row[0]), axis=1)))
        belongs = np.argmin(dis_matrix, axis=0)
        centers = []
        check_same = True
        for i in range(k):
            index = np.where(belongs == i)[0]
            centers_bank[-1][i][1] = index
            center = np.mean(data[index],axis=0)
            check_same  = check_same and np.all(center == centers_bank[-1][i][0])
            centers.append([center, np.array([])])
        if check_same == True:
            break
        centers_bank.append(centers)
        itr += 1
    print("total itr:",itr)
    return centers_bank


def get_centers(centers_bank, itr=-1):
    """
    Purpose:
        Return all centers for specific iteration of K-mean
    Input:
        centers_bank: the database to store all iterations information for K-mean
        itr: int, iteration number.
    """
    centers_matrix = np.empty((0,data.shape[1]))
    for row in centers_bank[itr]:
        centers_matrix = np.vstack((centers_matrix, row[0]))
    return centers_matrix

def plot_k_min(data, centers, itr):
    label_set = set(np.arange(len(centers)))
    color_map = dict(zip(label_set, cm.rainbow(np.linspace(0, 1, len(label_set)))))
    for label in label_set:
        index = centers[label][1]
        plt.scatter(data[index][:,0], data[index][:,1], s=20, c=color_map[label],
                    alpha=0.3, label=label)
        plt.scatter(centers[label][0][0], centers[label][0][1], s=100, c=color_map[label],
                    alpha=1.0, marker='x')
    plt.title("iteration: "+ str(itr))
    plt.legend(loc='best')
    plt.show()
    plt.close()


X = mylib.generate_data()
#X = generate_circle()
k = 4
centers_bank = get_centers_bank(X, k, 20)
for i in range(len(centers_bank)):
    plot_k_min(X,centers_bank[i], i+1)

data, labels = mylib.get_data("../data/cho.txt")
k = 5
data_pca = mylib.pca(data)
centers_bank = get_centers_bank(data_pca, k,20)
plot_k_min(data_pca,centers_bank[-1],len(centers_bank))

data, labels = mylib.get_data("../data/iyer.txt")
k = 5
data_pca = mylib.pca(data)
centers_bank = get_centers_bank(data, k,20)
plot_k_min(data_pca,centers_bank[-1], len(centers_bank))

