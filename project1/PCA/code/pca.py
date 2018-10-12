import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.cm as cm
import re

def pca(filename):
    with open(filename) as f:
        raw_data = np.genfromtxt(StringIO(f.read()), delimiter="\t",dtype='str')
        data = raw_data[:,:-1].astype(float)
        label = raw_data[:,-1]
    data_adjust = data - np.mean(data, axis=0)
    w, v = np.linalg.eig(np.cov(data_adjust.T))
    return data_adjust.dot(v[np.argsort(w)[-2:]].T), label


def svd(filename):
    with open(filename) as f:
        raw_data = np.genfromtxt(StringIO(f.read()), delimiter="\t",dtype='str')
        data = raw_data[:,:-1].astype(float)
        label = raw_data[:,-1]
    U, s, V = np.linalg.svd(data)
    S = np.zeros((U.shape[0],2))
    S[:2, :2] = np.diag(s[:2])
    return U.dot(S), label


def tsne(filename):
    with open(filename) as f:
        raw_data = np.genfromtxt(StringIO(f.read()), delimiter="\t",dtype='str')
        data = raw_data[:,:-1].astype(float)
        label = raw_data[:,-1]
    return TSNE(n_components=2).fit_transform(data), label


def plot_data(*param):
    data, labels, algo, dataset = param
    data_t = data.T
    label_set = set(labels)
    color_map = dict(zip(label_set, cm.rainbow(np.linspace(0, 1, len(label_set)))))
    for label in label_set:
        index = np.where(labels == label)
        plt.scatter(data_t[0][index], data_t[1][index], s=20, c=color_map[label],
                    alpha=0.5, label=label)  
    m = re.search(r'(\w+).txt', dataset)
    plt.title(algo + " on " + m.group(1))
    plt.legend(loc='best')
    plt.savefig('../img/' + algo + "_on_"+ m.group(1))
    plt.close()

if __name__ == "__main__":

	### Plot datasets by PCA
	data, labels = pca('../data/pca_a.txt')
	data_b, labels_b = pca('../data/pca_b.txt')
	data_c, labels_c = pca('../data/pca_c.txt')

	plot_data(data, labels, 'pca', 'pca_a.txt')
	plot_data(data_b, labels_b, 'pca', 'pca_b.txt')
	plot_data(data_c, labels_c, 'pca', 'pca_c.txt')

	### Plot datasets by SVD
	data_svd, labels_svd = svd('../data/pca_a.txt')
	data_b_svd, labels_b_svd = svd('../data/pca_b.txt')
	data_c_svd, labels_c_svd = svd('../data/pca_c.txt')

	plot_data(data_svd, labels_svd, 'svd', 'pca_a.txt')
	plot_data(data_b_svd, labels_b_svd, 'svd', 'pca_b.txt')
	plot_data(data_c_svd, labels_c_svd, 'svd', 'pca_c.txt')

	### Plot datasets by T-SNE
	data_tsne, labels_tsne = tsne('../data/pca_a.txt')
	data_b_tsne, labels_b_tsne = tsne('../data/pca_b.txt')
	data_c_tsne, labels_c_tsne = tsne('../data/pca_c.txt')

	plot_data(data_tsne, labels_tsne, 't-sne', 'pca_a.txt')
	plot_data(data_b_tsne, labels_b_tsne, 't-sne', 'pca_b.txt')
	plot_data(data_c_tsne, labels_c_tsne, 't-sne', 'pca_c.txt')




