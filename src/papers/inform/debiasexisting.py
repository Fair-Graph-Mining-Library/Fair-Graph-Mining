from debias import *


class DebiasExisting(Debias):
    def pagerank(self, alpha, c=0.85):
        """
        individually fair PageRank
        :param alpha: regularization parameter
        :param c: damping factor
        """
        mat = ((c / (1 + alpha)) * self.adj) + (((alpha) / (1 + alpha)) * self.sim)
        graph = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())
        return self.revised_power_method(graph, c=c, alpha=alpha)

    def spectral_clustering(self, alpha, ncluster=10, v0=None):
        """
        individually fair spectral clustering
        :param alpha: regularization parameter
        :param ncluster: number of clusters
        :param v0: starting vector for eigen-decomposition
        :return: soft cluster membership matrix of fair spectral clustering
        """
        lap = -1 * (laplacian(self.adj) + alpha * laplacian(self.sim))
        _, u = eigsh(lap, which="LM", k=ncluster, sigma=1.0, v0=v0)
        return u

    def line(
        self,
        alpha,
        dimension=128,
        ratio=3200,
        negative=5,
        init_lr=0.025,
        batch_size=1000,
        seed=None,
    ):
        """
        individually fair LINE
        :param graph: networkx nx.Graph()
        :param alpha: regularization hyperparameter
        :param dimension: embedding dimension
        :param ratio: ratio to control edge sampling #sampled_edges = ratio * #nodes
        :param negative: number of negative samples
        :param init_lr: initial learning rate
        :param batch_size: batch size of edges in each training iteration
        :param seed: random seed
        :return: debiased node embeddings
        """
        if seed is not None:
            np.random.seed(seed)
            
        graph = nx.from_scipy_sparse_matrix(self.init_adj, create_using=nx.Graph(), edge_attribute='weight')

        def _update(vec_u, vec_v, vec_error, label, u, v, lr):
            f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
            g = (lr * (label - f)).reshape((len(label), 1))
            vec_error += g * vec_v
            vec_v += g * vec_u
            if label[0] == 1:
                arr = np.asarray(self.sim[u, v].transpose())
                vec_error -= 2 * lr * alpha * (vec_u - vec_v) * arr
                vec_v -= 2 * lr * alpha * (vec_v - vec_u) * arr

        def _train_line():
            nbatch = int(nsamples / batch_size)
            for iter_num in range(nbatch):
                lr = init_lr * max((1 - iter_num * 1.0 / nbatch), 0.0001)
                u, v = [0] * batch_size, [0] * batch_size
                for i in range(batch_size):
                    edge_id = alias_draw(edges_table, edges_prob)
                    u[i], v[i] = edges[edge_id]
                    if not directed and np.random.rand() > 0.5:
                        v[i], u[i] = edges[edge_id]

                vec_error = np.zeros((batch_size, dimension))
                label, target = np.asarray([1 for _ in range(batch_size)]), np.asarray(
                    v
                )
                for j in range(negative + 1):
                    if j != 0:
                        label = np.asarray([0 for _ in range(batch_size)])
                        for k in range(batch_size):
                            target[k] = alias_draw(nodes_table, nodes_prob)
                    _update(
                        emb_vertex[u],
                        emb_vertex[target],
                        vec_error,
                        label,
                        u,
                        target,
                        lr,
                    )
                emb_vertex[u] += vec_error

        directed = nx.is_directed(graph)

        nnodes = graph.number_of_nodes()
        node2id = dict([(node, vid) for vid, node in enumerate(graph.nodes())])

        edges = [[node2id[e[0]], node2id[e[1]]] for e in graph.edges()]
        edge_prob = np.asarray(
            [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
        )
        edge_prob /= np.sum(edge_prob)
        edges_table, edges_prob = self.alias_setup(edge_prob)

        degree_weight = np.asarray([0] * nnodes)
        for u, v in graph.edges():
            degree_weight[node2id[u]] += graph[u][v].get("weight", 1.0)
            if not directed:
                degree_weight[node2id[v]] += graph[u][v].get("weight", 1.0)
        node_prob = np.power(degree_weight, 0.75)
        node_prob /= np.sum(node_prob)
        nodes_table, nodes_prob = self.alias_setup(node_prob)

        nsamples = ratio * nnodes
        emb_vertex = (np.random.random((nnodes, dimension)) - 0.5) / dimension

        # train
        _train_line()

        # normalize
        embeddings = skpp.normalize(emb_vertex, "l2")
        return embeddings
