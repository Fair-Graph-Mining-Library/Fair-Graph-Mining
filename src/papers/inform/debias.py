import time
import pickle
import os
from copy import deepcopy

import numpy as np
import networkx as nx

from scipy.sparse import csc_matrix, identity, diags
from scipy.sparse import diags, isspmatrix_coo, triu
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh, svds

import sklearn.preprocessing as skpp

import sio


class Debias:
    def __init__(self):
        return

    def __init__(self, adjacency, similarity):
        """
        :param adjacency: initial adjacency matrix of input graph
        :param sim: initial similarity matrix of input graph
        """
        self.init_adj = adjacency
        self.sim = similarity

    def __init__(self, filename):
        data = self.load_graph(filename)
        self.init_adj = data["adjacency_train"]
        self.sim = self.filter_similarity_matrix(
            self.get_similarity_matrix(self.init_adj, metric='jaccard'), sigma=0.75
        )

    def fit(self):
        return

    def load_graph(self, name, is_directed=False):
        if name.endswith(".pickle"):
            return self.read_pickle(name)
        elif name.endwith(".mat"):
            return self.read_mat(name)
        else:
            return self.read_graph(name, is_directed=is_directed)

    # The following methods are util-methods used with attribution from: https://github.com/jiank2/InFoRM/blob/main/utils.py

    @staticmethod
    def read_graph(name, is_directed=False):
        """
        read graph data from edge list file
        :param name: name of the graph
        :param is_directed: if the graph is directed or not
        :return: adjacency matrix, Networkx Graph (if undirected) or DiGraph (if directed)
        """
        if is_directed:
            PATH = os.path.join("data", "directed", "{}.txt".format(name))
            G = nx.read_edgelist(
                PATH, create_using=nx.DiGraph(), nodetype=int, data=(("weight", float),)
            )
        else:
            PATH = os.path.join("data", "{}.txt".format(name))
            G = nx.read_edgelist(
                PATH, create_using=nx.Graph(), nodetype=int, data=(("weight", float),)
            )
        return G

    @staticmethod
    def read_mat(name):
        """
        read .mat file
        :param name: dataset name
        :return: a dict containing adjacency matrix, nx.Graph() and its node labels
        """
        result = dict()
        PATH = os.path.join("data", "{}.mat".format(name))
        matfile = sio.loadmat(PATH)
        result["adjacency"] = matfile["network"]
        result["label"] = matfile["group"]
        result["graph"] = nx.from_scipy_sparse_matrix(
            result["adjacency"], create_using=nx.Graph(), edge_attribute="weight"
        )
        return result

    @staticmethod
    def read_pickle(name):
        """
        read .pickle file
        :param name: dataset name
        :return: a dict containing adjacency matrix, nx.Graph() and its node labels
        """
        with open(name, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def trace(mat):
        """
        calculate trace of a sparse matrix
        :param mat: scipy.sparse matrix (csc, csr or coo)
        :return: Tr(mat)
        """
        return mat.diagonal().sum()

    @staticmethod
    def row_normalize(mat):
        """
        normalize a matrix by row
        :param mat: scipy.sparse matrix (csc, csr or coo)
        :return: row-normalized matrix
        """
        degrees = np.asarray(mat.sum(axis=1).flatten())
        degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
        degrees = diags(np.asarray(degrees)[0, :])
        return degrees @ mat

    @staticmethod
    def column_normalize(mat):
        """
        normalize a matrix by column
        :param mat: scipy.sparse matrix (csc, csr or coo)
        :return: column-normalized matrix
        """
        degrees = np.asarray(mat.sum(axis=0).flatten())
        degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
        degrees = diags(np.asarray(degrees)[0, :])
        return mat @ degrees

    @staticmethod
    def symmetric_normalize(mat):
        """
        symmetrically normalize a matrix
        :param mat: scipy.sparse matrix (csc, csr or coo)
        :return: symmetrically normalized matrix
        """
        degrees = np.asarray(mat.sum(axis=0).flatten())
        degrees = np.divide(1, degrees, out=np.zeros_like(degrees), where=degrees != 0)
        degrees = diags(np.asarray(degrees)[0, :])
        degrees.data = np.sqrt(degrees.data)
        return degrees @ mat @ degrees

    @staticmethod
    def jaccard_similarity(mat):
        """
        get jaccard similarity matrix
        :param mat: scipy.sparse.csc_matrix
        :return: similarity matrix of nodes
        """
        # make it a binary matrix
        mat_bin = mat.copy()
        mat_bin.data[:] = 1

        col_sum = mat_bin.getnnz(axis=0)
        ab = mat_bin.dot(mat_bin.T)
        aa = np.repeat(col_sum, ab.getnnz(axis=0))
        bb = col_sum[ab.indices]
        sim = ab.copy()
        sim.data /= aa + bb - ab.data
        return sim

    def cosine_similarity(self, mat):
        """
        get cosine similarity matrix
        :param mat: scipy.sparse.csc_matrix
        :return: similarity matrix of nodes
        """
        mat_row_norm = skpp.normalize(mat, axis=1)
        sim = mat_row_norm.dot(mat_row_norm.T)
        return sim

    @staticmethod
    def filter_similarity_matrix(sim, sigma):
        """
        filter value by threshold = mean(sim) + sigma * std(sim)
        :param sim: similarity matrix
        :param sigma: hyperparameter for filtering values
        :return: filtered similarity matrix
        """
        sim_mean = np.mean(sim.data)
        sim_std = np.std(sim.data)
        threshold = sim_mean + sigma * sim_std
        sim.data *= sim.data >= threshold  # filter values by threshold
        sim.eliminate_zeros()
        return sim

    def get_similarity_matrix(self, mat, metric=None):
        """
        get similarity matrix of nodes in specified metric
        :param mat: scipy.sparse matrix (csc, csr or coo)
        :param metric: similarity metric
        :return: similarity matrix of nodes
        """
        if metric == "jaccard":
            return self.jaccard_similarity(mat.tocsc())
        elif metric == "cosine":
            return self.cosine_similarity(mat.tocsc())
        else:
            raise ValueError("Please specify the type of similarity metric.")

    @staticmethod
    def power_method(G, c=0.85, maxiter=100, tol=1e-3, personalization=None):
        """
        r = cWr + (1-c)e
        :param G: Networkx DiGraph created by transition matrix W
        :param c: damping factor
        :param maxiter: maximum number of iterations
        :param tol: error tolerance
        :param personalization: personalization for teleporation vector, uniform distribution if None
        :param print_msg: boolean to check whether to print number of iterations for convergence or not.
        :return: PageRank vector
        """
        nnodes = G.number_of_nodes()
        if personalization is None:
            e = dict.fromkeys(G, 1.0 / nnodes)
        else:
            e = dict.fromkeys(G, 0.0)
            for i in e:
                e[i] = personalization[i, 0]

        r = deepcopy(e)
        for niter in range(maxiter):
            rlast = r
            r = dict.fromkeys(G, 0)
            for n in r:
                for nbr in G[n]:
                    r[n] += c * rlast[nbr] * G[n][nbr]["weight"]
                r[n] += (1.0 - c) * e[n]
            err = sum([abs(r[n] - rlast[n]) for n in r])
            if err < tol:
                return r

        return r

    @staticmethod
    def revised_power_method(
        G, c=0.85, alpha=1.0, maxiter=100, tol=1e-3, personalization=None
    ):
        """
        r = Wr + (1-c)/(1+alpha) e
        :param G: Networkx DiGraph created by transition matrix W
        :param c: damping factor
        :param maxiter: maximum number of iterations
        :param tol: error tolerance
        :param personalization: personalization for teleporation vector, uniform distribution if None
        :return: PageRank vector
        """
        nnodes = G.number_of_nodes()
        if personalization is None:
            e = dict.fromkeys(G, 1.0 / nnodes)
        else:
            e = dict.fromkeys(G, 0.0)
            for i in e:
                e[i] = personalization[i, 0]

        r = deepcopy(e)
        for niter in range(maxiter):
            rlast = r
            r = dict.fromkeys(G, 0)
            for n in r:
                for nbr in G[n]:
                    r[n] += rlast[nbr] * G[n][nbr]["weight"]
                r[n] += (1.0 - c) * e[n] / (1.0 + alpha)
            err = sum([abs(r[n] - rlast[n]) for n in r])
            if err < tol:
                return r
        return r

    @staticmethod
    def reverse_power_method(G, c=0.85, maxiter=100, tol=1e-3, personalization=None):
        """
        r = cr'W + (1-c)e
        :param G: Networkx DiGraph created by transition matrix W
        :param c: damping factor
        :param maxiter: maximum number of iterations
        :param tol: error tolerance
        :param personalization: personalization for teleporation vector, uniform distribution if None
        :param print_msg: boolean to check whether to print number of iterations for convergence or not.
        :return: PageRank vector
        """
        nnodes = G.number_of_nodes()
        if personalization is None:
            e = dict.fromkeys(G, 1.0 / nnodes)
        else:
            e = dict.fromkeys(G, 0.0)
            for i in e:
                e[i] = personalization[i, 0]

        r = deepcopy(e)
        for niter in range(maxiter):
            rlast = r
            r = dict.fromkeys(G, 0)
            for n in r:
                for nbr in G[n]:
                    r[nbr] += c * rlast[n] * G[n][nbr]["weight"]
                r[n] += (1.0 - c) * e[n]
            err = sum([abs(r[n] - rlast[n]) for n in r])
            if err < tol:
                return r
        return r

    @staticmethod
    def alias_setup(probs):
        """
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return J, q


@staticmethod
def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
