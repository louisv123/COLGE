import torch
import torch.nn.functional as F
import random
#from gensim.models import Word2Vec
import networkx as nx
import numpy as np


class S2V_QN(torch.nn.Module):
    def __init__(self, embed_dim, len_pre_pooling, len_post_pooling, T):

        super(S2V_QN, self).__init__()
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Linear(1, embed_dim)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim)
        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            self.list_pre_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            self.list_post_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim)
        self.q = torch.nn.Linear(2 * embed_dim, 1)

    def forward(self, xv, adj, mu_init):

        for t in range(self.T):
            if t == 0:
                mu_1 = self.mu_1(xv)
                mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                mu = torch.add(mu_1, mu_2).clamp(0)

            else:
                mu_1 = self.mu_1(xv)

                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(adj, mu))
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        q = self.q(q_)
        return q


class W2V_QN(torch.nn.Module):
    def __init__(self, G, len_pre_pooling, len_post_pooling, embed_dim, window_size, num_paths, path_length, T):

        super(W2V_QN, self).__init__()
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Linear(1, embed_dim)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim)
        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            self.list_pre_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            self.list_post_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim)
        self.q = torch.nn.Linear(2 * embed_dim, 1)

        walks = self.build_deepwalk_corpus(G, num_paths=num_paths,
                                           path_length=path_length, alpha=0)

        self.model = Word2Vec(walks, size=embed_dim, window=window_size
                              , min_count=0, sg=1, hs=1, iter=1, negative=0, compute_loss=True)

    def random_walk(self, G, path_length, alpha=0, rand=random.Random(), start=None):

        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.nodes()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(list(nx.neighbors(G, cur))))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

    def build_deepwalk_corpus(self, G, num_paths, path_length, alpha=0):
        walks = []

        nodes = list(G.nodes())

        for cnt in range(num_paths):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(G, path_length, alpha=alpha, start=node))

        return walks

    def forward(self, xv, adj, mu_init):

        mu_w2v = torch.from_numpy(
            np.expand_dims(self.model[list(map(str, sorted(list(map(int, list(self.model.wv.vocab))))))], axis=0))

        for t in range(self.T):
            if t == 0:
                mu_1 = self.mu_1(xv)
                mu_2 = self.mu_2(torch.matmul(adj, mu_w2v))
                mu = torch.add(mu_1, mu_2).clamp(0)

            else:
                mu_1 = self.mu_1(xv)

                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(adj, mu))
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        q = self.q(q_)
        return q


class LINE_QN(torch.nn.Module):
    def __init__(self, size, embed_dim=128, order=1):
        super(LINE_QN, self).__init__()

        assert order in [1, 2], print("Order should either be int(1) or int(2)")

        self.embed_dim = embed_dim
        self.order = order
        self.nodes_embeddings = torch.nn.Embedding(size, embed_dim)

        if order == 2:
            self.contextnodes_embeddings = torch.nn.Embedding(size, embed_dim)
            # Initialization
            self.contextnodes_embeddings.weight.data = self.contextnodes_embeddings.weight.data.uniform_(
                -.5, .5) / embed_dim

        # Initialization
        self.nodes_embeddings.weight.data = self.nodes_embeddings.weight.data.uniform_(
            -.5, .5) / embed_dim

    def forward(self, v_i, v_j, negsamples, device):

        v_i = self.nodes_embeddings(v_i).to(device)

        if self.order == 2:
            v_j = self.contextnodes_embeddings(v_j).to(device)
            negativenodes = -self.contextnodes_embeddings(negsamples).to(device)

        else:
            v_j = self.nodes_embeddings(v_j).to(device)
            negativenodes = -self.nodes_embeddings(negsamples).to(device)

        mulpositivebatch = torch.mul(v_i, v_j)
        positivebatch = F.logsigmoid(torch.sum(mulpositivebatch, dim=1))

        mulnegativebatch = torch.mul(v_i.view(len(v_i), 1, self.embed_dim), negativenodes)
        negativebatch = torch.sum(
            F.logsigmoid(
                torch.sum(mulnegativebatch, dim=2)
            ),
            dim=1)
        loss = positivebatch + negativebatch
        return -torch.mean(loss)
