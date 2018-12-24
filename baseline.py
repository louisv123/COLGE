import numpy as np
import torch
import graph
import models

g = graph.Graph(graph_type="erdos_renyi", cur_n=50, p=.09)

X = np.zeros((g.nodes, 3))

for node in range(g.nodes()):
    X[node, 0] = g.degree(node)
    X[node, 1] = g.average_neighbor_degree([node])[node]
    X[node, 2] = np.min([g.degree(i) for i in g.neighbors(node)])

model = models.BASELINE()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1.e-6, momentum=0.9)

y_pred = model(X)
loss = criterion(y_pred, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
