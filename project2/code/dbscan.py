import numpy as np
from io import StringIO
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mylibrary as mylib
from mylibrary import euclidean_distance

class Point:
    def __init__(self, loc):
        self.loc = loc
        self.is_visited = False
        self.is_noise = False
        self.cluster_id = -1
        
    def label_noise(self, is_noise=True):
        self.is_noise = is_noise
        
    def label_cluster_id(self, c_id = -1):
        self.cluster_id = c_id
    def label_is_visited(self, is_visited=True):
        self.is_visited = True
    
    def __repr__(self):
         return str(self.loc) + " in cluster: " + str(self.cluster_id) 


def preprocess(data):
    """
    Purpose:
        transfer each record of data into the object of point
    Input:
        data: a two dimension matrix
    Output:
        a list of Point objects
    """
    return list(map(lambda x: Point(x), data))

def region_query(data, pt, eps, fun):
    """
    Purpose:
        get all neighbor of point pt within eps distance
    Input:
        data: a list of Point object
        pt: a Point
        eps: the epsilon distance
        fun: the function to calculate distance
    Output:
        neighbors: a list of Points
    """
    neighbors = []
    for neighbor in data:
        if fun(neighbor.loc, pt.loc) <= eps:
            neighbors.append(neighbor)
    return neighbors

def expand_cluster(data, pt, neighbors, cluster_id, eps, min_pts):
    """
    Purpose:
        generate a cluster from data
    Input:
        data: a list of Points
        pt: a Point
        neighbors: a list of Points
        cluster_id: int, current cluster id
        eps: the epsilon distance
        min_pts: the minimum number of neighbors.
    Output:
        the updated data
    """
    pt.label_cluster_id(cluster_id)
    neighbors.remove(pt)
    queue = deque(neighbors)
    
    while len(queue) != 0:
        neighbor = queue.popleft()
        if neighbor.is_noise:
            neighbor.label_noise(False)
            neighbor.label_cluster_id(cluster_id)
        if neighbor.cluster_id == -1:
            neighbor.label_cluster_id(cluster_id)
            neighbor_neighbors = region_query(data, neighbor, eps, euclidean_distance)
            if len(neighbor_neighbors) >= min_pts:
                queue.extend(neighbor_neighbors)
    return data

def dbscan(data, eps, min_pts):
    """
    Purpose:
        dbscan algorithm which is to do density base clustering.
    Input:
        data: a list of Points
        eps: the epsilon distance
        min_pts: the minimum number of neighbors.
    Output:
        cluster_id: the last cluster_id which is same as the number of iterations.
    """
    cluster_id = 0
    for pt in data:
        if pt.cluster_id == -1:
            neighbors = region_query(data, pt, eps, euclidean_distance)
            if len(neighbors) < min_pts:
                pt.label_noise(True)
            else:
                cluster_id += 1
                expand_cluster(data, pt, neighbors, cluster_id, eps, min_pts)
    return cluster_id

def plot_dbscan(data, clusters_num):
    label_set = set(np.append(-2, np.arange(clusters_num))+1)
    color_map = dict(zip(label_set, cm.rainbow(np.linspace(0, 1, len(label_set)))))
    for label in label_set:
        cluster = np.asarray(list(map(lambda y: y.loc, filter(lambda x: x.cluster_id == label, data))))
        if len(cluster) != 0:
            plt.scatter(cluster[:,0], cluster[:,1], s=20, c=color_map[label],
                        alpha=0.3, label=label)
    plt.title("clusters number: " + str(clusters_num))
    plt.legend(loc='best')
    plt.show()
    plt.close() 


if __name__ == "__main__":
    data = mylib.generate_data()
    data = preprocess(data)
    eps = 0.5
    pt = data[0]
    min_pts = 5
    clusters_num = dbscan(data, eps, min_pts)
    plot_dbscan(data, clusters_num)

    data = mylib.generate_circle()
    data = preprocess(data)
    eps = 0.15
    pt = data[0]
    min_pts = 5
    clusters_num = dbscan(data, eps, min_pts)
    plot_dbscan(data, clusters_num)






