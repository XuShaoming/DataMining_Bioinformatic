import io
import sys
import re
import numpy as np
from io import StringIO
from sklearn import datasets as ds
from functools import reduce

def generate_circle():
    # generate data
    X,c = ds.make_circles(n_samples=500, factor=.5,noise=.05,random_state=10)
    return X

def generate_data():
    sigma = np.array([[0.2,0],[0,0.2]])
    n = 100
    mu1 = np.array([1,1])
    mu2 = np.array([3,4])
    mu3 = np.array([4,9])
    mu4 = np.array([1,8])
    x11 = np.random.multivariate_normal(mu1,sigma,n)
    x15 = np.random.multivariate_normal(mu2,sigma,n)
    x51 = np.random.multivariate_normal(mu3,sigma,n)
    x55 = np.random.multivariate_normal(mu4,sigma,n)
    X = np.vstack([x11,x15,x51,x55])  
    return X

def get_data(filename):
    with open(filename) as f:
        raw_data = np.genfromtxt(StringIO(f.read()), delimiter="\t")
    label = raw_data[:,0:2].astype(int)
    data = raw_data[:,2:]
    return data, label

def pca(data):
    data_adjust = data - np.mean(data, axis=0)
    w, v = np.linalg.eig(np.cov(data_adjust.T))
    return data_adjust.dot(v[np.argsort(w)[-2:]].T)

def euclidean_distance(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

def get_pca_file(filename):
    data, label = get_data(filename)
    data_pca = pca(data)
    m = re.search(r'(\w+).txt', filename)
    filename = m.group(1) + "_pca.txt"
    with open(filename, 'w') as f:
        for i, line in enumerate(data_pca):
            line = np.append(label[i], line)
            row = reduce(lambda a,b: str(a) +'\t' + str(b), line) + "\n"
            f.write(row)  

def get_random_center(inputfile, k, seed=20, filename="KmeanCenter.txt"):
    data, label = get_data(inputfile)
    centers = data[np.random.RandomState(seed=seed).permutation(data.shape[0])[0:k]]
    with open(filename, 'w') as f:
        for i, center in enumerate(centers):
            row = str(i) + "\t" + reduce(lambda a,b: str(a) +'\t' + str(b), center) + "\n"
            f.write(row) 


#if __name__ == "__main__":