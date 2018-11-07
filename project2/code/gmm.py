import numpy as np
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import os

UBIT = '50247057'
np.random.seed(sum([ord(c) for c in UBIT]))

class Point:
    """
    Purpose:
        each point save a record of data, and a list ot r_k. each value in r_k represents the 
        possibilitie that it belogins to one distribution. the length of r_ks shows the
        number of distributions. the id of the distributions varing from 0 to len(r_ks)-1.
    """
    def __init__(self, loc, r_ks):
        self.loc = loc
        self.r_ks = r_ks.copy()
        
    def set_r_ks(self, r_ks):
        self.r_ks = r_ks.copy()
        
    def set_r_k(self, k, val):
        self.r_ks[k] = val
    
    def show_r_ks(self):
        res = "\n "
        for i, val in enumerate(self.r_ks):
            res += "r_" + str(i) + "=" + str(val) +"\n "
        return res + "\n\n"
    
    def __repr__(self):
        return "pts:" + str(self.loc) + "\nr_ks is:" + self.show_r_ks()

class Distribution:
    """
    Purpose:
        each object represents a gaussisn distribution. it stores the mean, sigma(convariance matrix),
        and the pi_k. The pi_k is a float value that represents the possibility that the distribution 
        been picked in given data.
    """
    def __init__(self, mean, sigma, pi_k):
        self.mean = mean.copy()
        self.sigma = sigma.copy()
        self.pi_k = pi_k
    def set_mean(self, mean):
        self.mean = mean.copy()
        
    def set_sigma(self, sigma):
        self.sigma = sigma.copy()   
    
    def set_pi_k(self, pi_k):
        self.pi_k = pi_k
        
    def __eq__(self, other): 
        return np.all(abs(self.mean - other.mean) < 0.001)
    
    def __repr__(self):
        return "mean:\n" + str(self.mean) + "\nsigma:\n" + str(self.sigma) +"\npi_k:\n" + str(self.pi_k) + "\n\n\n"

def preprocess(data_given, means, sigmas, pi_k):
    """
    Purpose:
        preprocess the data to init the distributions and data list
    Input:
        data_given: a two dimension matrix
        means: a two dimension matrix
        sigmas: a three dimension matrix, each row represents a covariance matrix
        pi_k: float, the possibility that a distribution in given data. 
    Output:
        data: a list of Point object.
        distributions: a list of Distribution object.
    """
    if len(means) != len(sigmas):
        raise Exception("means number must equal to sigmas number!")
    r_ks = np.zeros(len(means))
    distributions = list(map(lambda x: Distribution(x[0], x[1], pi_k), zip(means,sigmas)))
    data = list(map(lambda x: Point(x, r_ks), data_given))
    return data, distributions

def e_step(data, distributions):
    """
    Purpose:
        Do the E step of Expectation-Maximizaion algorithm. The inner function set_r_ks is 
        to update the r_ks list of the Point object given the distributions list. So the 
        e_step function is to update all the Point objects in data on r_ks list given the 
        distributions list.
    Input:
        data: a list of Point object.
        distributions: a list of Distribution object.
    Output:
        data: updated data.
    """
    def set_r_ks(point, distributions):
        a_s = []
        b = 0
        for distribution in distributions:
            a = distribution.pi_k * multivariate_normal.pdf(point.loc, distribution.mean, distribution.sigma)
            a_s.append(a)
            b += a
        a_s = np.asarray(a_s)
        r_ks = a_s / b
        point.set_r_ks(r_ks)
        return point

    return list(map(lambda x: set_r_ks(x, distributions), data))

def m_step(data, distributions):
    """
    Purpose:
        Do the M step of Expectation-Maximizaion algorithm which updates the pi_ks, means, and sigmas
        Then use them to get a new list of Distribution objects.
    Input:
        data: a list of Point object.
        distributions: a list of Distribution object.
    Output:
        a list of Distribution object updated by pi_ks, means, and sigmas
    """
    r = np.asarray(list(map(lambda x: x.r_ks, data)))
    a = np.sum(r,axis=0)
    #get pi_ks
    pi_ks = a / r.shape[0]
    
    #get means
    locs = np.asarray(list(map(lambda x: x.loc, data)))
    means = []
    for r_k in r.T:
        means.append(np.sum(np.asarray(list(map(lambda x : x[0] * x[1], zip(locs, r_k)))), axis=0))
    means = np.asarray(means)
    means = np.asarray(list(map(lambda x : x[0] / x[1], zip(means, a))))
    
    # get sigmas
    sigmas = []
    for k, mean in enumerate(means):
        x_u = locs - mean
        v_1 = x_u.T @ np.asarray(list(map(lambda x: x[0] * x[1], zip(x_u, r.T[k]))))
        v_2 = a[k]
        sigmas.append(v_1 / v_2)
    sigmas = np.asarray(sigmas)
    
    # return new list of Distribution objects
    return list(map(lambda x: Distribution(x[0],x[1],x[2]), zip(means, sigmas, pi_ks)))

def det(A):
    """
    Purpose:
        Compute the determinant of a matrix.
    Input:
        A: a square matrix
    Output:
        res: real, the determinant of a matrix
    """
    if A.shape[0] != A.shape[1]:
        raise Exception("must be square matrix")
    if A.shape == (2,2):
        return A[0,0] * A[1,1] - A[0,1] * A[1,0]
    res = 0
    for i, row in enumerate(A.T):
        res += row[0] * pow(-1,i) * det(np.delete(np.delete(A, 0, 0), i, 1))
    return res

def gaussian_mixture_model(data_given, init_fun, max_itr=100, trace_plot=False, data_name="anony", export_path="../task3_img/"):
    """
    Purpose:
        the main funcion for Expectation-Maximizaion.
    Input:
        data_given: two dimension matrix saving the data.
        init_fun: a function which provides the init means, sigmas, and pi_k for distributions
        max_itr: int, the number of maximum iteration. default 100.
        trace_plot: boolean, set true will save plot of each iterations in given save-path.
        data_name, export_path: string, export_path+data_name generate the save_path. Default:
        data_name="anony", export_path="../task3_img/"
    Output:
        data: a list of Point object in the final iteration
        distributions: a list of Distribution object in the final iteration.
    """
    means, sigmas, pi_k = init_fun()
    data, distributions = preprocess(data_given, means, sigmas, pi_k)
    
    if trace_plot:
        save_path = export_path + data_name
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print("use existing folder:", save_path)
    
    for i in np.arange(max_itr):
        print("iteration:", i)
        if i == 1 and data_name == "bonus_a":
            print("means")
            for val in [distribution.mean for  distribution in distributions]:
                print(val)
        if trace_plot:
            plot_gmm(data_given, distributions, i, save_path)
        try:
            data = e_step(data, distributions)
        except Exception as inst:
            print("\nEncounter exception")
            print(inst)
            break
        new_distributions = m_step(data, distributions)
        if new_distributions == distributions:
            break
        if not np.all(np.asarray(list(map(lambda x: det(x.sigma), new_distributions))) != 0):
            print("\n\nEncounter singular matrix exception!!!")
            break
        distributions = new_distributions
        
    return data, distributions

def plot_gmm(data_given, distributions, itr,save_path="../task3_img/anony"):
    """
    Purpose:
        use to plot the result of EM algorithm. it will save the result in given save_path.
    Input:
        data_given: a two dimension matrix which save the data.
        distributions: a list of Distribution objects.
        itr: int, the number of iteration.
        save_path: string, the place the save the image. Default: ../task3_img/anony
    Output:
        None.
    """
    # Plot the raw points...
    x, y = data_given.T
    plt.plot(x, y, 'ro')

    label_set = set(np.arange(len(distributions)))
    color_map = dict(zip(label_set, cm.rainbow(np.linspace(0, 1, len(label_set)))))

    # Plot a transparent 3 standard deviation covariance ellipse
    for k, label in enumerate(label_set):
        plot_cov_ellipse(distributions[k].sigma, distributions[k].mean, 2, None, alpha=0.2,color=color_map[label], )    
    plt.title("iteration: "+ str(itr))
    plt.savefig(save_path+"/iteration_" + str(itr) + ".png")
    plt.close()

def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    """
    adopt from: https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def get_faithful_data(filename):
    """
    Purpose:
        Read file, and extract data and label from it.
    Input:
        filename: String
    Output:
        data: a matrix of float
        label: a vector of int
    """
    with open(filename) as f:
        raw_data = np.genfromtxt(StringIO(f.read()), dtype='str')
        data = raw_data[:,1:].astype("float")
        label = raw_data[:,0].astype("int")
    return data, label

def init_means_sigmas_piks_a():
    """
    Purpose:
        For task3 part a
        init_fun for gaussian_mixture_model function. which provide the means, sigmas and pi_ks for 
        distribution. 
    """
    means = np.array([[6.2, 3.2],
                  [6.6, 3.7],
                  [6.5, 3.0]])
    sigmas = np.array([[[0.5,0],[0,0.5]],
                   [[0.5,0],[0,0.5]],
                   [[0.5,0],[0,0.5]]])
    pi_k = 1 / 3
    return means, sigmas, pi_k

def init_means_sigmas_piks_faithful():
    """
    Purpose:
        For task3 part b
        init_fun for gaussian_mixture_model function. which provide the means, sigmas and pi_ks for 
        distribution. 
    """
    means = np.array([[4.0, 81],
                  [2.0, 57],
                  [4.0, 71]])
    sigmas = np.array([[[1.30,13.98],[13.98,184.82]],
                       [[1.30,13.98],[13.98,184.82]],
                       [[1.30,13.98],[13.98,184.82]]
                      ])
    pi_k = 1 / 3
    return means, sigmas, pi_k


if __name__ == "__main__":
    print("************ question a ***************")
    data_a = np.array([[5.9,3.2],
                     [4.6,2.9],
                     [6.2,2.8],
                     [4.7,3.2],
                     [5.5,4.2],
                     [5.0,3.0],
                     [4.9,3.1],
                     [6.7,3.1],
                     [5.1,3.8],
                     [6.0,3.0]])

    data, distributions = gaussian_mixture_model(data_a, init_means_sigmas_piks_a, trace_plot=True, data_name="bonus_a")

    print("************ question b ***************")
    faithful_data, _ = get_faithful_data("../data/faithful.dat")
    data, distributions = gaussian_mixture_model(faithful_data, init_means_sigmas_piks_faithful, 
    	max_itr=10, trace_plot=True, data_name="bonus_b_faithful_data")

