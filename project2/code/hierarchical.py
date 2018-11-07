import numpy as np
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mylibrary as mylib
from mylibrary import euclidean_distance

def init_clusters_bank(data):
    """
    Purpose:
        init layer 0 of the clusters_bank at where each points is a cluster
    Input: 
        data : two dimension matrix
    output:
        a list of set of frozenset. each frozenset is a cluster.
    """
    return [set([frozenset([i+1]) for i in np.arange(len(data))])]

def init_distance_matrix(clusters_bank, fun):
    """
    Purpose:
        init distance marix. distance matrix is a dictionary which key is two cluster,
        value is the distance between the two cluster.
    Input:
        clusters_bank: a list of set of frozenset. each frozenset is a cluster.
        fun: the function to calculate distance
    Output:
        distance_matrix:  a dictionary
    """
    distance_matrix = {}
    for cluster in combinations(clusters_bank[0], 2):
        distance_matrix[frozenset(cluster)] = fun(data[np.asarray(list(cluster[0])) - 1], data[np.asarray(list(cluster[1])) - 1])
    return distance_matrix

def get_min_set(distance_matrix):
    """
    Purpose:
        Get two clusters which has least distance in distance matrix.
    Input: 
        distance_matrix: a dictionary which key is two cluster, value is the distance between the two cluster.
    Output:
        The key which map to least value in distance_matrix.
    """
    return min(distance_matrix, key=distance_matrix.get)

def merge_min_set(min_set):
    """
    Purpose:
        merge two clusters in one
    Input:
        min_set: frozenset contains two frozensets. the two frozensets are two clusters.
    Output:
        merged_min_set a frozenset of cluster.
    """
    merged_min_set = frozenset()
    for val in min_set:
        merged_min_set = merged_min_set | val
    merged_min_set = frozenset([merged_min_set])
    return merged_min_set

def get_overlap_dict(distance_matrix, min_set):
    """
    Purpose:
        remove the items which keys has overlap with min_set in distance_matrix. Store these
        items in overlap_dict
    Input:
        distance_matrix: a dictionary which key is two cluster, value is the distance between the two cluster.
        min_set: frozenset contains two frozensets. the two frozensets are two clusters.
    Output:
        overlap_dict: a dictionary contans removed items from distance_matrix.
    """
    overlap_dict = {}
    for key in list(distance_matrix.keys()):
        if(len(min_set & key) != 0):
            overlap_dict[key] = distance_matrix.pop(key)
    del overlap_dict[min_set]
    return overlap_dict

def update_distance_matrix(distance_matrix, overlap_dict, min_set):
    """
    Purpose:
        Update distance matrix by Min method of Hierarchical clustering. Take the min distance
        of a cluster to two clusters in min_set as the distance between the cluster and new cluster.
    Input:
        distance_matrix: a dictionary which key is two cluster, value is the distance between the two cluster.
        overlap_dict: a dictionary contans removed items from distance_matrix.
        min_set: frozenset contains two frozensets. the two frozensets are two clusters.
    Output:
        updated distance_matrix
    """
    merged_min_set = merge_min_set(min_set)
    for cluster_1 in overlap_dict.keys():
        cluster_2 = (cluster_1 | min_set) - (cluster_1 & min_set)
        new_cluster = merged_min_set | (cluster_1 & cluster_2)
        distance_matrix[new_cluster] = min(overlap_dict[cluster_1], overlap_dict[cluster_2])
    return distance_matrix

def update_clusters_bank(clusters_bank, distance_matrix):
    """
    Purpose:
        add new clusters as the new row of clusters_bank
    input:
        clusters_bank: a list of set of frozenset. each frozenset is a cluster.
        distance_matrix: a dictionary which key is two cluster, value is the distance between the two cluster.
    Output:
        updated clusters_bank
    """
    row = set()
    for key in distance_matrix.keys():
        row |= key
    clusters_bank.append(row)
    return clusters_bank

def plot_hierarchical(data, clusters):
    clusters = list(map(lambda x: np.asarray(list(x)), clusters))
    label_set = set(np.arange(len(clusters)))
    color_map = dict(zip(label_set, cm.rainbow(np.linspace(0, 1, len(label_set)))))
    for label in label_set:
        index = clusters[label] - 1
        plt.scatter(data[index][:,0], data[index][:,1], s=20, c=color_map[label],
                    alpha=0.3, label=label)
    plt.title("clusters number: "+ str(len(clusters)))
    plt.legend(loc='best')
    plt.show()
    plt.close()

if __name__ == "__main__":
    #data = mylib.generate_data()
    data = mylib.generate_circle()
    clusters_bank = init_clusters_bank(data)
    distance_matrix = init_distance_matrix(clusters_bank, euclidean_distance)
    while(len(distance_matrix) != 0):
        print(len(distance_matrix))
        min_set = get_min_set(distance_matrix)
        overlap_dict = get_overlap_dict(distance_matrix, min_set)
        distance_matrix = update_distance_matrix(distance_matrix, overlap_dict, min_set)
        clusters_bank = update_clusters_bank(clusters_bank, distance_matrix)

    plot_hierarchical(data, clusters_bank[-2])

    # data, labels = mylib.get_data("../data/cho.txt")
    # data = mylib.pca(data)
    # clusters_bank = init_clusters_bank(data)
    # distance_matrix = init_distance_matrix(clusters_bank)
    # while(len(distance_matrix) != 0):
    #     #print(len(distance_matrix))
    #     min_set = get_min_set(distance_matrix)
    #     overlap_dict = get_overlap_dict(distance_matrix, min_set)
    #     distance_matrix = update_distance_matrix(distance_matrix, overlap_dict, min_set)
    #     clusters_bank = update_clusters_bank(clusters_bank, distance_matrix)






