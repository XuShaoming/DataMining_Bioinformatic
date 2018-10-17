import numpy as np
from sklearn import datasets as ds


def generate_circle():
    # generate data
    X,c = ds.make_circles(n_samples=1000, factor=.5,noise=.05)
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

def euclidean_distance(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


#if __name__ == "__main__":